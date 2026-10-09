# Implementation plan: publish this project

## Understanding

Create a GitHub repository under the confirmed `christian-hoang-04` account and push the current machine translation project, including its tested experiment artifacts. The working folder is `D:\projects\iit\machine_translation`; it had no Git repository or remote at the start of this work.

The project contains Python source and documentation, `pyproject.toml` and `uv.lock`, local data, fitted model files, experiment results, and Kaggle execution packages and outputs. `.gitignore` excludes `.venv`, Python bytecode caches, and `data/raw` plus `data/processed`. The local `.venv` is about 664 MiB and will stay local. The data, results, models, and execution folders total about 354 MiB. Two project files exceed GitHub's normal 100 MB per-file limit: `results/experiment2/paired_annotations.csv` (about 161 MiB) and `data/raw/wmt25-genmt-humeval.jsonl` (about 123 MiB). Git LFS 3.7.1 is installed.

## Success criteria

- A new public GitHub repository exists under `christian-hoang-04`.
- Project source, documentation, data, models, results, and Kaggle execution artifacts are committed and pushed.
- `.venv`, Python bytecode/cache directories, and the transient 1-byte `results/experiment3/.provider-ledger.json.lock` marker are excluded.
- Both oversized files are stored with Git LFS and their LFS objects are uploaded.
- Public commits use the GitHub account's no-reply email rather than the machine's configured personal email.
- The local primary branch tracks the remote branch, and local Git status is clean.
- `implementation-process.md` records commands, changes, verification, problems, decisions, and follow-ups.

## Assumptions and decisions

- The user confirmed the GitHub account and delegated the repository name. Use `wmt25-esa-score-prediction`, derived from the project purpose; a read-only availability check found no repository at that name.
- The user explicitly chose public visibility. Publish all agreed project deliverables in the public repository.
- "Everything tested" includes project source, docs, input/prepared data, saved models, result files/plots/reports, and Kaggle packages and run outputs/logs.
- Exclude the reproducible local `.venv`, Python bytecode/cache directories, and the transient provider lock marker; retain the actual `uv.lock` dependency lockfile.
- Explicitly stage ignored raw and processed data. Use Git LFS for the two files above GitHub's normal size limit.
- Do not edit existing source or experiment artifacts, and do not run project tests or experiments.

## Blocking questions

None. The account, naming discretion, and public visibility are explicit.

## Intended files and commands

Changes are limited to Git metadata, this requested plan, the requested execution log, and `.gitattributes` for LFS rules. Existing source and experiment outputs remain untouched.

1. Initialize Git with `main` as the primary branch and set a repository-local author identity using the account handle and GitHub no-reply email.
2. Configure LFS for the two oversized files. Stage all non-ignored project files and explicitly force-add `data/raw` and `data/processed`.
3. Review staged paths and sizes, confirm local environment/cache exclusions, and perform a credential-oriented filename/content check before committing.
4. Commit the project, create a public GitHub repository under `christian-hoang-04`, add it as `origin`, push `main` and its LFS objects.

## Verification approach

- Before commit, review `git status`, staged names and sizes, presence of data/results/models/execution deliverables, exclusions, and LFS attributes/pointers. Preserve existing result files even if whitespace checks report their stored text formatting.
- After push, verify the remote URL, remote branch and commit, LFS object upload, and clean local status.
- No project tests or experiment runs are planned because no project code is being changed.

## Planning status

Planning is complete. Implementation is in progress; execution details are recorded in `implementation-process.md`.
