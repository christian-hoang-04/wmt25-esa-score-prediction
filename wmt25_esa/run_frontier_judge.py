"""Run feature-only and context-aware judgments on the frozen 100-item sample.

This script uses the Meddies ProviderPool so credential selection, token usage,
quota, and provider pacing stay under the shared runtime. It writes parsed scores
and sanitized request telemetry only; prompts and raw provider responses are not
persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .prepare_data import NUMERIC_FEATURES


PROMPT_VERSION = "wmt25-esa-frontier-v1"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_INPUT_PRICE = 0.75  # USD / million tokens, official Gemini pricing through 2026-12-31.
DEFAULT_OUTPUT_PRICE = 3.75

FEATURE_ONLY_SCHEMA = {
    "type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["score"], "additionalProperties": False,
}
CONTEXT_AWARE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "explanation": {"type": "string"},
    },
    "required": ["score", "explanation"], "additionalProperties": False,
}


def system_prompt(condition: str) -> str:
    base = (
        "Rate the overall quality of this machine translation on a 0 to 100 scale, "
        "where higher is better. Use only the supplied input. Return valid JSON only. "
    )
    if condition == "feature_only":
        return base + 'The exact schema is {"score": integer from 0 to 100}; no other keys.'
    if condition == "context_aware":
        return base + 'The exact schema is {"score": integer from 0 to 100, "explanation": short string}; no other keys.'
    raise ValueError(f"Unknown prompt condition: {condition}")


def _json_spans(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {
            "start_i": int(item["start_i"]),
            "end_i": int(item["end_i"]),
            "severity": str(item["severity"]),
            "text": str(item.get("text", "")),
        }
        for item in parsed
        if isinstance(item, dict)
        and isinstance(item.get("start_i"), int)
        and isinstance(item.get("end_i"), int)
        and item.get("severity") in {"minor", "major"}
    ]


def build_user_prompt(row: dict[str, Any], condition: str) -> str:
    """Build a prompt from a strict allowlist; labels and IDs cannot enter it."""
    features = {name: float(row[name]) for name in NUMERIC_FEATURES}
    payload: dict[str, Any] = {"error_features": features}
    if condition == "context_aware":
        payload = {
            "source_text_english": str(row["src_text"]),
            "translation_bhojpuri": str(row["target_text"]),
            "input_annotation_errors": _json_spans(row.get("error_spans_a_json")),
            "error_features": features,
        }
    elif condition != "feature_only":
        raise ValueError(f"Unknown prompt condition: {condition}")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_score(content: str) -> int | None:
    """Accept only a JSON object containing exactly one integer 0..100 score."""
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        return None
    score = parsed["score"]
    if type(score) is not int or not 0 <= score <= 100:
        return None
    return score


def parse_response(content: str, condition: str) -> tuple[int, str | None] | None:
    """Validate the exact output object required by a prompt condition."""
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    expected = {"score"} if condition == "feature_only" else {"score", "explanation"}
    if set(parsed) != expected:
        return None
    score = parsed["score"]
    if type(score) is not int or not 0 <= score <= 100:
        return None
    explanation = parsed.get("explanation")
    if condition == "context_aware" and not isinstance(explanation, str):
        return None
    return score, explanation


def write_prompt_artifacts(sample: pd.DataFrame, results_dir: Path) -> list[Path]:
    """Save auditable prompt payloads; identifiers stay outside model messages."""
    prompt_dir = results_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for condition in ("feature_only", "context_aware"):
        path = prompt_dir / f"{condition}.jsonl"
        schema = FEATURE_ONLY_SCHEMA if condition == "feature_only" else CONTEXT_AWARE_SCHEMA
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sample.to_dict("records"):
                record = {
                    "translation_id": str(row["translation_id"]),
                    "condition": condition,
                    "response_schema": schema,
                    "messages": [
                        {"role": "system", "content": system_prompt(condition)},
                        {"role": "user", "content": build_user_prompt(row, condition)},
                    ],
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
    return paths


def _request_reserve_usd(system: str, user: str, max_tokens: int, input_price: float, output_price: float) -> float:
    # UTF-8 byte count is deliberately a conservative upper bound for text tokens.
    input_reserve = len((system + user).encode("utf-8"))
    return input_reserve * input_price / 1_000_000 + max_tokens * output_price / 1_000_000


def _actual_cost(prompt_tokens: int | None, completion_tokens: int | None, input_price: float, output_price: float) -> float:
    if prompt_tokens is None or completion_tokens is None:
        return 0.0
    return prompt_tokens * input_price / 1_000_000 + completion_tokens * output_price / 1_000_000


def _load_frozen_inputs(sample_path: Path, manifest_path: Path, local_path: Path) -> tuple[pd.DataFrame, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if digest != manifest.get("sample_file_sha256"):
        raise ValueError("Frozen sample hash does not match its manifest")
    frozen = pd.read_csv(sample_path, low_memory=False)
    local = pd.read_csv(local_path, low_memory=False)
    if len(frozen) != 100 or frozen["translation_id"].nunique() != 100:
        raise ValueError("Expected exactly 100 distinct frozen translations")
    selected_ids = set(frozen["translation_id"].astype(str))
    local_ids = set(local["translation_id"].astype(str))
    if selected_ids != local_ids:
        raise ValueError("Local baseline rows do not exactly match the frozen translation IDs")
    local = frozen[["translation_id", "directed_pair_id", "pair_id"]].merge(
        local, on=["translation_id", "directed_pair_id", "pair_id"], how="left", validate="one_to_one"
    )
    if local["target_text"].isna().any() or local["score_b"].isna().any():
        raise ValueError("Frozen sample is missing translation context or paired human targets")
    return local, digest


def _error_details(error: BaseException) -> tuple[int | None, str, float | None]:
    context = getattr(error, "context", {})
    status = None
    if isinstance(context, dict):
        for key in ("status_code", "http_status", "status"):
            value = context.get(key)
            if isinstance(value, int) and 100 <= value <= 599:
                status = value
                break
    response = getattr(error, "response", None)
    if status is None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value <= 599:
            status = value
    retry_after = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("Retry-After")
            retry_after = float(value) if value is not None else None
        except (TypeError, ValueError):
            retry_after = None
    category = {
        401: "authentication_or_access",
        403: "authentication_or_access",
        402: "billing_or_quota",
        400: "request_or_model_configuration",
        404: "request_or_model_configuration",
        429: "rate_or_quota_limit",
    }.get(status, "provider_http_error" if status is not None else type(error).__name__)
    return status, category, retry_after


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


async def run(args: argparse.Namespace) -> int:
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    sample, sample_digest = _load_frozen_inputs(args.sample, args.sample_manifest, args.local_baselines)
    if args.prepare_prompts_only:
        paths = write_prompt_artifacts(sample, results_dir)
        for path in paths:
            print(f"Prepared prompt artifact: {path.resolve()}")
        return 0

    # The provider model override is process-local. API credentials still come
    # from the runtime's environment/keyring precedence and are never inspected.
    os.environ["GEMINI_MODEL"] = args.model
    os.environ["AI_STUDIO_MODEL"] = args.model
    from meddies_llm_runtime.provider_pool import open_provider_pool

    request_log_path = results_dir / "frontier_request_log.csv"
    predictions_path = results_dir / "frontier_predictions.csv"
    existing_log = pd.read_csv(request_log_path).to_dict("records") if request_log_path.exists() else []
    existing_predictions = pd.read_csv(predictions_path).to_dict("records") if predictions_path.exists() else []
    valid_done = {
        (str(row.get("translation_id")), str(row.get("condition")), str(row.get("model")), str(row.get("sample_sha256")))
        for row in existing_predictions
        if row.get("score_prediction") is not None and not pd.isna(row.get("score_prediction"))
    }
    log_rows: list[dict[str, Any]] = existing_log
    prediction_rows: list[dict[str, Any]] = existing_predictions
    spent_actual = sum(float(row.get("actual_cost_usd") or 0.0) for row in existing_log)
    projected_total = sum(
        _request_reserve_usd(system_prompt(condition), build_user_prompt(row, condition), args.max_tokens, args.input_price_per_million, args.output_price_per_million)
        for row in sample.to_dict("records") for condition in ("feature_only", "context_aware")
    )
    if projected_total + spent_actual > args.max_cost_usd:
        raise ValueError(
            f"Conservative total request reserve ${projected_total + spent_actual:.4f} exceeds the ${args.max_cost_usd:.2f} cap"
        )

    ledger_path = results_dir / "provider-ledger.json"
    budget_path = results_dir / "provider-budget.json"
    async with open_provider_pool(
        [args.provider], ledger_state_path=ledger_path, budget_state_path=budget_path
    ) as pool:
        if not pool.lanes:
            raise RuntimeError(f"No configured lane for provider {args.provider}; skipped={[(x.lane, x.reason.value) for x in pool.skipped]}")
        lane = pool.lanes[0]
        if lane.model != args.model:
            raise RuntimeError(f"Provider pool resolved model {lane.model!r}, expected requested model {args.model!r}")
        if args.preflight_only:
            start = time.monotonic()
            status = "ok"
            error_category = ""
            http_status = None
            prompt_tokens = completion_tokens = total_tokens = None
            actual_cost = 0.0
            schema_valid = False
            retry_after = None
            try:
                result = await lane.client.chat_completion(
                    "Reply with the exact JSON object {\"score\": 0}.",
                    "Synthetic connectivity check. Return the requested JSON object only.",
                    temperature=0.0, max_tokens=128, max_attempts=1,
                )
                prompt_tokens, completion_tokens, total_tokens = result.prompt_tokens, result.completion_tokens, result.total_tokens
                actual_cost = _actual_cost(prompt_tokens, completion_tokens, args.input_price_per_million, args.output_price_per_million)
                schema_valid = parse_score(result.content) == 0
                if not result.content.strip():
                    status, error_category = "empty_synthetic_response", "empty_content"
            except Exception as exc:  # sanitized: response bodies and request content are not logged
                status, error_category = "request_error", type(exc).__name__
                http_status, error_category, retry_after = _error_details(exc)
            row = {
                "translation_id": "", "condition": "synthetic_preflight", "model": lane.model,
                "provider": lane.provider, "status": status, "error_category": error_category,
                "http_status": http_status, "latency_seconds": round(time.monotonic() - start, 3),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens,
                "actual_cost_usd": actual_cost, "sample_sha256": sample_digest,
                "synthetic_schema_valid": schema_valid, "retry_after_seconds": retry_after,
            }
            log_rows.append(row)
            _write_csv(request_log_path, log_rows)
            (results_dir / "frontier_preflight.json").write_text(json.dumps({
                "provider": lane.provider, "model": lane.model, "status": status,
                "error_category": error_category, "http_status": http_status,
                "latency_seconds": row["latency_seconds"], "sample_sha256": sample_digest,
                "preflight_cost_usd": actual_cost, "synthetic_schema_valid": schema_valid,
                "retry_after_seconds": retry_after,
            }, indent=2), encoding="utf-8")
            print(f"Preflight status={status}; provider={lane.provider}; model={lane.model}; cost=${actual_cost:.6f}")
            return 0 if status == "ok" else 2

        if not args.confirmed_preflight:
            raise ValueError("Run --preflight-only successfully first, then pass --confirmed-preflight")
        preflight_path = results_dir / "frontier_preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
        if preflight.get("status") != "ok" or preflight.get("model") != args.model:
            raise ValueError("A successful synthetic preflight for this exact model is required before data calls")

        tasks = [(row, condition) for row in sample.to_dict("records") for condition in ("feature_only", "context_aware")]
        for index, (row, condition) in enumerate(tasks, start=1):
            identity = (str(row["translation_id"]), condition, lane.model, sample_digest)
            if identity in valid_done:
                continue
            user_prompt = build_user_prompt(row, condition)
            condition_system_prompt = system_prompt(condition)
            reserve = _request_reserve_usd(condition_system_prompt, user_prompt, args.max_tokens, args.input_price_per_million, args.output_price_per_million)
            if spent_actual + reserve > args.max_cost_usd:
                raise RuntimeError(f"Stopping before request {index}: actual spend plus conservative request reserve would exceed ${args.max_cost_usd:.2f}")
            start = time.monotonic()
            request_status = "ok"
            error_category = ""
            http_status = None
            prompt_tokens = completion_tokens = total_tokens = None
            actual_cost = 0.0
            prediction: int | None = None
            try:
                result = await lane.client.chat_completion(
                    condition_system_prompt, user_prompt, temperature=0.0,
                    max_tokens=args.max_tokens, max_attempts=1,
                )
                prompt_tokens, completion_tokens, total_tokens = result.prompt_tokens, result.completion_tokens, result.total_tokens
                parsed = parse_response(result.content, condition)
                prediction = parsed[0] if parsed is not None else None
                actual_cost = _actual_cost(prompt_tokens, completion_tokens, args.input_price_per_million, args.output_price_per_million)
                if prediction is None:
                    request_status, error_category = "invalid_response", "invalid_json_score_schema"
            except Exception as exc:  # one physical attempt; do not broadly retry quota/auth errors
                request_status, error_category = "request_error", type(exc).__name__
                http_status, error_category, _ = _error_details(exc)
            latency = round(time.monotonic() - start, 3)
            log_row = {
                "translation_id": str(row["translation_id"]), "condition": condition,
                "model": lane.model, "provider": lane.provider, "status": request_status,
                "error_category": error_category, "http_status": http_status,
                "latency_seconds": latency, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "total_tokens": total_tokens,
                "explanation": parsed[1] if request_status == "ok" and parsed is not None else None,
                "actual_cost_usd": actual_cost, "projected_reserve_usd": reserve,
                "sample_sha256": sample_digest,
            }
            log_rows.append(log_row)
            prediction_rows.append({
                "translation_id": str(row["translation_id"]), "pair_id": str(row["pair_id"]),
                "directed_pair_id": str(row["directed_pair_id"]), "condition": condition,
                "model": lane.model, "score_prediction": prediction,
                "request_status": request_status, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "total_tokens": total_tokens,
                "actual_cost_usd": actual_cost, "sample_sha256": sample_digest,
            })
            spent_actual += actual_cost
            if prediction is not None:
                valid_done.add(identity)
            _write_csv(request_log_path, log_rows)
            pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
            if request_status == "request_error":
                # Preserve the failed call in the log and stop rather than switching lanes/models.
                raise RuntimeError(f"Frontier request failed with sanitized category {error_category} (HTTP {http_status})")
            if index % 10 == 0 or index == len(tasks):
                print(f"Requests processed {index}/{len(tasks)}; parsed scores={len(valid_done)}; cost=${spent_actual:.4f}", flush=True)

    metadata = {
        "provider": args.provider, "model": args.model, "prompt_version": PROMPT_VERSION,
        "sample_sha256": sample_digest, "sample_size": int(len(sample)),
        "conditions": ["feature_only", "context_aware"],
        "price_usd_per_million_input_tokens": args.input_price_per_million,
        "price_usd_per_million_output_tokens": args.output_price_per_million,
        "max_cost_usd": args.max_cost_usd, "actual_cost_usd": round(spent_actual, 8),
        "request_count": len(log_rows), "prediction_count": len(prediction_rows),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "prompts_or_raw_responses_saved": False,
    }
    (results_dir / "frontier_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frontier run complete; provider={args.provider}; model={args.model}; observed cost upper-bound estimate=${spent_actual:.4f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=Path, default=Path("results/experiment3/sample100.csv"))
    parser.add_argument("--sample-manifest", type=Path, default=Path("results/experiment3/sample100_manifest.json"))
    parser.add_argument("--local-baselines", type=Path, default=Path("results/experiment3/sample100_local_baselines.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiment3"))
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price-per-million", type=float, default=DEFAULT_OUTPUT_PRICE)
    parser.add_argument("--max-cost-usd", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirmed-preflight", action="store_true")
    parser.add_argument("--prepare-prompts-only", action="store_true", help="Write frozen prompt JSONL files without opening a provider pool")
    args = parser.parse_args()
    if args.max_cost_usd <= 0 or args.max_tokens <= 0:
        parser.error("--max-cost-usd and --max-tokens must be positive")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
