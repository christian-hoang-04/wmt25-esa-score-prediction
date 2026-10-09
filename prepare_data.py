"""Flatten WMT25 ESA annotations and derive numeric error-span features."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_URL = (
    "https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/"
    "data/wmt25-genmt-humeval.jsonl"
)
PROTOCOL_URL = "https://www2.statmt.org/wmt25/mteval-subtask2.html"

# WMT25 Subtask 2 explicitly identifies these as ESA-score language pairs.
# MQM-only pairs (en-ko_KR and ja-zh_CN) are intentionally excluded.
ESA_LANG_PAIRS = {
    "cs-uk_UA",
    "cs-de_DE",
    "en-ar_EG",
    "en-zh_CN",
    "en-cs_CZ",
    "en-et_EE",
    "en-is_IS",
    "en-it_IT",
    "en-ja_JP",
    "en-ru_RU",
    "en-sr_Cyrl_RS",
    "en-uk_UA",
    "en-bho_IN",
    "en-mas_KE",
}
MQM_LANG_PAIRS = {"en-ko_KR", "ja-zh_CN"}

NUMERIC_FEATURES = [
    "n_minor",
    "n_major",
    "n_total",
    "target_char_length",
    "source_char_length",
    "target_word_length",
    "source_word_length",
    "minor_coverage",
    "major_coverage",
    "total_error_coverage",
    "max_error_span_length",
    "mean_error_span_length",
    "error_density",
    "major_fraction",
    "has_major_error",
    "has_zero_errors",
]
FEATURE_SETS = {
    "F1": ["n_minor", "n_major"],
    "F2": [
        "n_minor",
        "n_major",
        "minor_coverage",
        "major_coverage",
        "total_error_coverage",
    ],
    "F3": NUMERIC_FEATURES,
}

ANNOTATION_FIELDS = [
    "annotation_id",
    "doc_id",
    "doc_group_id",
    "segment_id",
    "language_pair",
    "domain",
    "system_name",
    "annotator_id",
    "annotation_order",
    "src_text",
    "target_text",
    "source_char_length",
    "target_char_length",
    "source_word_length",
    "target_word_length",
    "n_minor",
    "n_major",
    "n_total",
    "minor_coverage",
    "major_coverage",
    "total_error_coverage",
    "max_error_span_length",
    "mean_error_span_length",
    "error_density",
    "major_fraction",
    "has_major_error",
    "has_zero_errors",
    "score",
    "error_spans_json",
]


def parse_doc_id(doc_id: str) -> dict[str, str]:
    """Parse the WMT ``_#_`` identifier while retaining safe fallbacks."""
    parts = doc_id.split("_#_")
    language_pair = parts[0].strip() if parts else ""
    document_name = ""
    if len(parts) >= 4 and all(parts[:3]):
        domain = parts[1]
        document_name = "_#_".join(parts[2:-1])
        segment_id = parts[-1]
        document_key = f"{language_pair}:::{domain}:::{document_name}"
        parse_status = "documented_format"
    else:
        domain = parts[1] if len(parts) > 1 else "unknown"
        segment_id = parts[-1] if len(parts) > 1 else "unknown"
        # A malformed identifier is isolated rather than merged with unrelated rows.
        document_key = f"{language_pair}:::unparsed:::{doc_id}"
        parse_status = "fallback_unparsed"
    return {
        "language_pair": language_pair,
        "domain": domain,
        "document_name": document_name if len(parts) >= 4 else "",
        "document_key": document_key,
        "segment_id": segment_id,
        "parse_status": parse_status,
    }


def _valid_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _union_span_length(spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start + 1 for start, end in merged)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p10": None,
                "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        index = (len(ordered) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return float(ordered[lower])
        weight = index - lower
        return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "std": float(statistics.pstdev(ordered)),
        "min": float(ordered[0]),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": float(ordered[-1]),
    }


def _annotation_errors(
    errors: Any, target: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return valid spans and invalid span records; offsets are inclusive."""
    if not isinstance(errors, list):
        return [], [{"reason": "errors_not_a_list", "raw": errors}]

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for position, error in enumerate(errors):
        if not isinstance(error, dict):
            invalid.append({"position": position, "reason": "error_not_an_object", "raw": error})
            continue
        start = error.get("start_i")
        end = error.get("end_i")
        severity = str(error.get("severity", "")).strip().lower()
        if start == "missing" or end == "missing":
            reason = "offset_marked_missing"
        elif not _valid_offset(start) or not _valid_offset(end):
            reason = "offset_not_integer"
        elif start < 0 or end < start or end >= len(target):
            reason = "offset_out_of_bounds_or_reversed"
        elif severity not in {"minor", "major"}:
            reason = "unsupported_severity"
        else:
            valid.append(
                {
                    "start_i": start,
                    "end_i": end,
                    "severity": severity,
                    "text": target[start : end + 1],
                }
            )
            continue
        invalid.append(
            {
                "position": position,
                "reason": reason,
                "start_i": start,
                "end_i": end,
                "severity": error.get("severity"),
            }
        )
    return valid, invalid


def _feature_row(
    *,
    doc_id: str,
    id_parts: dict[str, str],
    system_name: str,
    annotator_id: str,
    annotation_order: int,
    src_text: str,
    target: str,
    score: float,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    minor = [e for e in errors if e["severity"] == "minor"]
    major = [e for e in errors if e["severity"] == "major"]
    all_spans = [(e["start_i"], e["end_i"]) for e in errors]
    minor_spans = [(e["start_i"], e["end_i"]) for e in minor]
    major_spans = [(e["start_i"], e["end_i"]) for e in major]
    target_length = len(target)
    n_total = len(errors)
    span_lengths = [e["end_i"] - e["start_i"] + 1 for e in errors]
    coverage_denominator = max(target_length, 1)
    source_words = len(src_text.split())
    target_words = len(target.split())
    return {
        "annotation_id": f"{doc_id}::{system_name}::{annotation_order}",
        "doc_id": doc_id,
        "doc_group_id": id_parts["document_key"],
        "segment_id": id_parts["segment_id"],
        "language_pair": id_parts["language_pair"],
        "domain": id_parts["domain"],
        "system_name": system_name,
        "annotator_id": annotator_id,
        "annotation_order": annotation_order,
        "src_text": src_text,
        "target_text": target,
        "source_char_length": len(src_text),
        "target_char_length": target_length,
        "source_word_length": source_words,
        "target_word_length": target_words,
        "n_minor": len(minor),
        "n_major": len(major),
        "n_total": n_total,
        "minor_coverage": _union_span_length(minor_spans) / coverage_denominator,
        "major_coverage": _union_span_length(major_spans) / coverage_denominator,
        "total_error_coverage": _union_span_length(all_spans) / coverage_denominator,
        "max_error_span_length": max(span_lengths, default=0),
        "mean_error_span_length": statistics.fmean(span_lengths) if span_lengths else 0.0,
        "error_density": (100.0 * n_total / coverage_denominator),
        "major_fraction": len(major) / n_total if n_total else 0.0,
        "has_major_error": int(bool(major)),
        "has_zero_errors": int(n_total == 0),
        "score": score,
        "error_spans_json": json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
    }


def prepare_data(
    input_path: Path,
    output_dir: Path = Path("data/processed"),
    results_dir: Path = Path("results"),
) -> tuple[Path, Path, Path]:
    input_path = input_path.expanduser().resolve()
    annotations_path = output_dir / "annotations.csv.gz"
    summary_path = results_dir / "dataset_summary.json"
    invalid_path = results_dir / "invalid_annotations.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = defaultdict(
        lambda: {
            "records": 0,
            "raw_annotations": 0,
            "retained_annotations": 0,
            "invalid_annotations": 0,
            "invalid_spans": 0,
            "invalid_reason_counts": Counter(),
            "documents": set(),
            "segments": set(),
            "systems": set(),
            "annotators": set(),
            "scores": [],
            "n_minor": [],
            "n_major": [],
            "zero_error_count": 0,
            "fallback_doc_ids": 0,
        }
    )
    protocol_counts: Counter[str] = Counter()
    invalid_reason_counts: Counter[str] = Counter()
    row_count = 0
    invalid_annotation_count = 0

    with input_path.open("r", encoding="utf-8") as source, gzip.open(
        annotations_path, "wt", encoding="utf-8", newline=""
    ) as compressed, invalid_path.open("w", encoding="utf-8", newline="") as invalid_file:
        writer = csv.DictWriter(compressed, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        invalid_writer = csv.DictWriter(
            invalid_file,
            fieldnames=[
                "annotation_id",
                "doc_id",
                "language_pair",
                "system_name",
                "annotator_id",
                "score",
                "invalid_reasons",
                "invalid_spans_json",
            ],
        )
        invalid_writer.writeheader()

        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_number}")

            doc_id = str(record.get("doc_id", ""))
            id_parts = parse_doc_id(doc_id)
            language_pair = id_parts["language_pair"]
            protocol_counts[language_pair] += 1
            if language_pair not in ESA_LANG_PAIRS:
                continue

            lp_stats = stats[language_pair]
            lp_stats["records"] += 1
            lp_stats["segments"].add(doc_id)
            lp_stats["documents"].add(id_parts["document_key"])
            if id_parts["parse_status"] != "documented_format":
                lp_stats["fallback_doc_ids"] += 1

            src_text = record.get("src_text")
            source_text_missing = not isinstance(src_text, str)
            if not isinstance(src_text, str):
                src_text = ""
            targets = record.get("tgt_text") if isinstance(record.get("tgt_text"), dict) else {}
            scores_by_system = record.get("scores") if isinstance(record.get("scores"), dict) else {}

            for system_name, annotations in scores_by_system.items():
                if not isinstance(annotations, list):
                    continue
                lp_stats["systems"].add(str(system_name))
                target = targets.get(system_name)
                for annotation_order, annotation in enumerate(annotations):
                    lp_stats["raw_annotations"] += 1
                    if not isinstance(annotation, dict):
                        annotation = {}
                    annotator_id = annotation.get("annotator")
                    raw_score = annotation.get("score")
                    reasons: list[str] = []
                    if source_text_missing:
                        reasons.append("missing_source_text")
                    try:
                        score = float(raw_score)
                        if not math.isfinite(score) or not 0 <= score <= 100:
                            reasons.append("score_outside_0_100_or_nonfinite")
                    except (TypeError, ValueError):
                        score = float("nan")
                        reasons.append("missing_or_non_numeric_score")
                    if not isinstance(annotator_id, str) or not annotator_id.strip():
                        reasons.append("missing_annotator_id")
                        annotator_id = ""
                    if not isinstance(target, str):
                        reasons.append("missing_target_translation")
                        target_text = ""
                    else:
                        target_text = target

                    valid_errors, invalid_errors = _annotation_errors(
                        annotation.get("errors"), target_text
                    )
                    if invalid_errors:
                        reasons.append("invalid_error_spans_or_severity")
                        for invalid_error in invalid_errors:
                            invalid_reason = str(invalid_error.get("reason", "unknown"))
                            invalid_reason_counts[invalid_reason] += 1
                            lp_stats["invalid_reason_counts"][invalid_reason] += 1
                    if reasons:
                        invalid_annotation_count += 1
                        lp_stats["invalid_annotations"] += 1
                        lp_stats["invalid_spans"] += len(invalid_errors)
                        detailed_reasons = list(dict.fromkeys(reasons))
                        detailed_reasons.extend(
                            f"error_record:{error.get('reason', 'unknown')}"
                            for error in invalid_errors
                            if f"error_record:{error.get('reason', 'unknown')}" not in detailed_reasons
                        )
                        invalid_writer.writerow(
                            {
                                "annotation_id": f"{doc_id}::{system_name}::{annotation_order}",
                                "doc_id": doc_id,
                                "language_pair": language_pair,
                                "system_name": system_name,
                                "annotator_id": annotator_id,
                                "score": raw_score,
                                "invalid_reasons": ";".join(detailed_reasons),
                                "invalid_spans_json": json.dumps(
                                    invalid_errors, ensure_ascii=False, separators=(",", ":")
                                ),
                            }
                        )
                        # Any annotation with malformed error marks is omitted entirely so an
                        # incomplete feature vector cannot be mistaken for a clean annotation.
                        continue

                    row = _feature_row(
                        doc_id=doc_id,
                        id_parts=id_parts,
                        system_name=str(system_name),
                        annotator_id=annotator_id,
                        annotation_order=annotation_order,
                        src_text=src_text,
                        target=target_text,
                        score=score,
                        errors=valid_errors,
                    )
                    writer.writerow(row)
                    row_count += 1
                    lp_stats["retained_annotations"] += 1
                    lp_stats["annotators"].add(annotator_id)
                    lp_stats["scores"].append(score)
                    lp_stats["n_minor"].append(row["n_minor"])
                    lp_stats["n_major"].append(row["n_major"])
                    lp_stats["zero_error_count"] += int(row["n_total"] == 0)

    per_language_pair: dict[str, Any] = {}
    for language_pair in sorted(ESA_LANG_PAIRS):
        lp_stats = stats[language_pair]
        per_language_pair[language_pair] = {
            "records": lp_stats["records"],
            "unique_segments": len(lp_stats["segments"]),
            "unique_documents": len(lp_stats["documents"]),
            "translation_systems": len(lp_stats["systems"]),
            "raw_annotations": lp_stats["raw_annotations"],
            "retained_annotations": lp_stats["retained_annotations"],
            "invalid_annotations": lp_stats["invalid_annotations"],
            "invalid_spans": lp_stats["invalid_spans"],
            "invalid_error_record_reasons": dict(sorted(lp_stats["invalid_reason_counts"].items())),
            "annotators": len(lp_stats["annotators"]),
            "fallback_doc_ids": lp_stats["fallback_doc_ids"],
            "score_distribution": _distribution(lp_stats["scores"]),
            "n_minor_distribution": _distribution(lp_stats["n_minor"]),
            "n_major_distribution": _distribution(lp_stats["n_major"]),
            "percentage_no_marked_errors": (
                100 * lp_stats["zero_error_count"] / lp_stats["retained_annotations"]
                if lp_stats["retained_annotations"]
                else None
            ),
        }

    summary = {
        "dataset": "WMT25 General MT human evaluation",
        "dataset_url": DATASET_URL,
        "protocol_source": PROTOCOL_URL,
        "protocol_filter": "Only the 14 language pairs explicitly designated ESA in WMT25 Subtask 2",
        "included_esa_language_pairs": sorted(ESA_LANG_PAIRS),
        "excluded_mqm_language_pairs": sorted(MQM_LANG_PAIRS),
        "input_path": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "jsonl_records_included": sum(v["records"] for v in stats.values()),
        "annotation_rows_retained": row_count,
        "invalid_annotations_excluded": invalid_annotation_count,
        "invalid_error_record_reasons": dict(sorted(invalid_reason_counts.items())),
        "language_pair_record_counts_all_protocols": dict(sorted(protocol_counts.items())),
        "language_pairs_missing_from_data": sorted(ESA_LANG_PAIRS.difference(stats)),
        "per_language_pair": per_language_pair,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Retained {row_count:,} annotation examples across {len(per_language_pair)} ESA directions")
    print(f"Excluded {invalid_annotation_count:,} annotations with invalid/missing fields or spans")
    print(f"Prepared data: {annotations_path.resolve()}")
    print(f"Dataset summary: {summary_path.resolve()}")
    return annotations_path, summary_path, invalid_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/wmt25-genmt-humeval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    prepare_data(args.input, args.output_dir, args.results_dir)


if __name__ == "__main__":
    main()
