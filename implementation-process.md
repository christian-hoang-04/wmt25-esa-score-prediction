# Implementation process: publish this project

## Outcome

- Created public repository: https://github.com/christian-hoang-04/wmt25-esa-score-prediction
- Pushed the project on `main`, including 233 files: source, docs, ignored raw/prepared data, models, results, plots/reports, and Kaggle execution files and outputs.
- The initial project commit is `976a92d8191ff261cf6c5b17102d594c04ddd63e` (`Publish WMT25 ESA score prediction project`). The branch tracks `origin/main`.
- Both oversized files use Git LFS. The push reported 2/2 LFS objects uploaded (297 MB).

## Commands run

Review and setup:

- `Get-Location`, `Get-ChildItem -Force`, `rg --files`, and reads of `README.md`, `pyproject.toml`, `requirements.txt`, and `.gitignore`.
- `git status --short --branch`, `git remote -v`, `git rev-parse --show-toplevel`, and `git log -1 --oneline` before initialization; these confirmed there was no Git repository.
- `gh auth status`; `gh repo view christian-hoang-04/wmt25-esa-score-prediction --json name,url,isPrivate` to check availability; `git lfs version`.
- Recursive file counts and size inventory. The raw dataset and paired annotations CSV were the only files above 100 MiB.
- Credential-oriented `rg -l` scan for common GitHub/cloud tokens, private-key headers, credential-bearing URLs, and secret-like assignments; no matching files.

Repository setup and publication:

- `git init -b main`
- `git lfs install --local`
- `git lfs track 'data/raw/wmt25-genmt-humeval.jsonl' 'results/experiment2/paired_annotations.csv'`
- `git add -A` and `git add -f -- data/raw data/processed`
- `git rm --cached --quiet -- results/experiment3/.provider-ledger.json.lock`; added `/results/experiment3/.provider-ledger.json.lock` to local `.git/info/exclude` so the transient marker stayed untouched and untracked.
- `gh api user --jq .id` and `gh api user --jq .login`; set repository-local author name to `christian-hoang-04` and email to the account's GitHub no-reply address.
- `git diff --cached --name-only`, `git diff --cached --shortstat`, `git lfs status`, `git lfs ls-files`, and `git check-attr filter` for the staging/LFS review.
- `git commit -m "Publish WMT25 ESA score prediction project"`
- `gh repo create christian-hoang-04/wmt25-esa-score-prediction --public --source . --remote origin --push`
- Verified with `gh repo view ...`, `git ls-remote origin refs/heads/main`, `git branch -vv`, `git lfs fsck --objects main`, and `git status --porcelain`.

## Files and repository state changed

- `implementation-plan.md`: records scope, decisions, success criteria, and completed status.
- `implementation-process.md`: records this execution and its verification.
- `.gitattributes`: applies Git LFS to the two oversized files.
- Git metadata: initialized `main`, configured local LFS and the account no-reply author identity, added `origin`, and recorded the initial project commit.
- `.git/info/exclude`: locally excludes the 1-byte transient provider lock marker; the marker itself was not modified.
- No existing source, data, model, or experiment output file was edited. `.gitignore` was not changed.

## Verification results

- GitHub reports `isPrivate: false`, URL `https://github.com/christian-hoang-04/wmt25-esa-score-prediction`, and default branch `main`.
- The initial remote `main` SHA matched local `HEAD`: `976a92d8191ff261cf6c5b17102d594c04ddd63e`.
- GitHub accepted both LFS uploads; `git lfs fsck --objects main` returned `Git LFS fsck OK`.
- The staged review counted 233 files, included both raw and processed data files, and found no `.venv`, `__pycache__`, or transient provider lock paths.
- No credential-pattern matches were found.
- `git status --porcelain` returned zero entries; `main` tracked `origin/main`.
- No tests or experiments were run because no project code changed.

## Problems, decisions, and follow-ups

- `.gitignore` omits `data/raw` and `data/processed`; they were explicitly force-added as requested project deliverables.
- `git diff --cached --check` reported pre-existing trailing whitespace in generated result CSVs (2,282 diagnostic lines). The artifact contents were left unchanged.
- Two attempts to interpolate the GitHub no-reply address in one `gh api --jq` expression failed because of PowerShell quoting. Separate `.id` and `.login` queries resolved it; no project files were affected.
- No blocking issues or follow-ups remain.

## New request: repo-wide refactor (planning complete)

- User clarified the scope as the entire repo and local code. No blocking questions remain.
- Assumptions and success criteria are recorded in the active section of `implementation-plan.md`: centralize all authored Python in `wmt25_esa/`, retain old commands as compatibility entrypoints, keep data/models/results in place, refresh generated Kaggle code from its builder, and push to the existing public remote.
- Read-only review commands run: `git status --short --branch`, `git remote -v`, `git log -3 --oneline`, `git ls-files '*.py'`, `rg --files`, and targeted `rg` searches for imports and command/path references.
- Read `README.md`, `pyproject.toml`, `.gitignore`, both existing implementation notes, the Kaggle bundle builder, and core script sections. Inspected top-level Python definitions/imports with a read-only AST inventory.
- Findings: 15 tracked Python files; no dedicated `tests/` directory. Largest core scripts are `experiment2.py` (1,084 lines), `evaluate.py` (891), `train.py` (666), and `evaluate_frontier.py` (621). Kaggle's `main.py` is generated from selected source scripts.
- At planning review, no project source, data, model, report, or run-output file had yet been changed.

### Implementation started

- Chosen structure: canonical implementations under `wmt25_esa/`; keep the documented root and result-folder script paths as thin compatibility wrappers. Leave the Kaggle benchmark adapter at its platform-required path and rebuild the generated Kaggle entrypoint from the canonical source.
- Preserve `data/`, `models/`, `results/`, `execution/run-output/`, and their contents/paths.
- No tests, training, experiments, or API calls will be run.

### Refactor progress

- Moved the eight core pipeline/builder implementations into `wmt25_esa/` and four saved-result analysis implementations into `wmt25_esa/analysis/`.
- Added package initializers and compatibility wrappers at the original root/result-script paths. Updated internal imports to package-relative imports.
- Corrected analysis output directories after relocation so scripts still target their original `results/` folders.
- Updated the Kaggle builder to include package files and create nested directories when materializing its source bundle. Rebuilt `execution/kaggle_kernel/main.py` through the builder; it is 33,908 bytes.
- Added a README code-layout section. No data, models, reports, plots, or run outputs were regenerated or edited.
- Static inspection parsed 28 Python files with `ast.parse`; all 10 declared Kaggle bundle inputs exist, and the generated bundle contains all required package files. No project tests/workloads were run.

Commands run for this refactor so far:

- `git mv` for the eight core scripts and four analysis scripts into `wmt25_esa/`.
- Read-only `rg` searches for stale imports, path constants, and entrypoint references.
- `python build_kaggle_kernel.py` (successful; generated the Kaggle package).
- A Python AST and bundle-content inspection script (syntax and generated-source structure only; no project code executed).

### Final local review before publication

- Compared each relocated canonical source file against its prior `HEAD` version. The source changes are limited to package-relative imports, relocated result-output path constants, and the Kaggle builder's package-aware source/materialization logic.
- Reviewed every compatibility wrapper and its target module. Existing root and result-folder command paths remain present.
- Staged review covered 30 paths. `git diff --cached --name-only -- data models` returned no paths; result/run-output changes were Python entrypoint moves/wrappers only, with no non-code artifacts changed.
- Static syntax inspection passed for 28 Python files. All 10 Kaggle bundle inputs exist, and the generated bundle contains the required package files.
- No project tests, CLI workloads, experiments, training, or API calls were run.
- Commit and push to the existing public `origin/main` remain to be performed after this final local review.
