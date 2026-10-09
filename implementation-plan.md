# Implementation plan: publish this project

## Understanding

Create a public GitHub repository under the confirmed `christian-hoang-04` account and push this machine translation project, including its tested experiment artifacts. The working folder is `D:\projects\iit\machine_translation`; it had no Git repository or remote at the start.

The project includes Python source and documentation, `pyproject.toml` and `uv.lock`, input and prepared data, fitted models, experiment results, and Kaggle packages and run outputs. `.gitignore` excludes `.venv`, Python bytecode caches, and the raw and processed data folders. The local `.venv` is about 664 MiB and stays local. Two project files exceed GitHub's normal 100 MB per-file limit: `results/experiment2/paired_annotations.csv` (about 161 MiB) and `data/raw/wmt25-genmt-humeval.jsonl` (about 123 MiB); Git LFS is installed.

## Success criteria

- A public GitHub repository exists under `christian-hoang-04` and contains the project deliverables.
- `.venv`, Python bytecode/cache directories, and the transient provider lock marker are excluded.
- The two oversized files use Git LFS and their LFS objects are uploaded.
- `main` tracks the remote branch, and the local checkout is clean.
- The planning and execution notes are present in the working folder and pushed.

## Assumptions and decisions

- Repository name chosen from the project purpose: `wmt25-esa-score-prediction`; the authenticated-account lookup found no existing repository with that name.
- The user explicitly requested public visibility and confirmed the `christian-hoang-04` account.
- “Everything tested” includes source, docs, data, models, results, plots, reports, and Kaggle execution outputs/logs.
- Exclude only the local `.venv`, bytecode/cache directories, and the transient 1-byte `results/experiment3/.provider-ledger.json.lock` marker. Keep `uv.lock` and all other project data and outputs.
- Use Git LFS for the two oversized files. Use the account's GitHub no-reply email for this public commit rather than the machine's global personal email.
- Do not edit existing project source or experiment outputs, and do not run project tests or experiments.

## Blocking questions

None. The account, public visibility, and naming discretion are explicit.

## Intended files and commands

Changes are limited to Git metadata, these requested planning/execution notes, and `.gitattributes` for LFS. Existing source and experiment outputs remain unchanged.

1. Initialize Git on `main`, configure LFS and repository-local commit identity.
2. Stage project files, explicitly force-adding the ignored `data/raw` and `data/processed` deliverables. Exclude `.venv`, caches, and the transient provider lock marker.
3. Review staged names/sizes, credential patterns, exclusions, and LFS attributes.
4. Commit, create a public GitHub repository, push `main` and LFS objects, then verify the remote and working tree.

## Verification approach

- Review staged paths and counts, confirm data is present and environment/cache paths are absent, and confirm LFS filters are set.
- Run a credential-oriented pattern scan before public upload.
- Verify repository visibility, remote URL and branch commit, LFS object integrity, and clean local status.
- No project tests or experiment runs are needed because project code is unchanged.

## Planning status

Implementation complete. Execution details and verification results are in `implementation-process.md`.
