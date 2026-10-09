# Implementation plan

## Active request: repo-wide refactor

### Understanding

Refactor the entire project codebase and local checkout into a clearer structure while preserving its behavior and existing ways to run the project. The repository is public on GitHub, and the user clarified that the request covers the entire repo and local code.

### Review findings

- The checkout is on `main`, tracking `origin/main`; it was clean before the current planning-note updates.
- The project has 15 tracked Python files spanning core experiment scripts, results-specific analysis scripts, a Kaggle benchmark adapter, and a generated Kaggle entrypoint.
- Root scripts contain substantial mixed orchestration/analysis logic (`experiment2.py` 1,084 lines, `evaluate.py` 891, `train.py` 666, `evaluate_frontier.py` 621).
- `build_kaggle_kernel.py` embeds selected source files into `execution/kaggle_kernel/main.py`; the generated bundle must be rebuilt from its generator.
- Existing CLI paths are documented in README and experiment reports. Other modules import `prepare_data`, `train`, and `run_frontier_judge` by their current root names.
- There is no dedicated `tests/` directory. Data, saved models, reports, plots, and run outputs are project deliverables.

### Success criteria

- All general experiment and saved-result analysis code has a clear canonical package location; platform-specific Kaggle entrypoints stay self-contained at their required paths.
- Existing documented commands and module imports remain supported.
- Kaggle bundle generation points at the canonical code and the checked-in generated entrypoint is refreshed from the builder.
- Data, model, result, plot, report, and run-output files remain byte-for-byte unchanged and in their current paths.
- The local refactor is pushed to the existing public `origin` after reviewing the complete diff.
- `implementation-process.md` records the work and actual verification; no project tests are added or run under this request.

### Assumptions

- “Entire repo” means all hand-authored project code and its entrypoints, including local analysis scripts; frozen data and experiment outputs stay in place because their paths are part of the current project interface.
- Preserve root-level commands and paths as compatibility wrappers even after moving canonical implementations into a package.
- Preserve results and models exactly; do not rerun experiments or regenerate scientific outputs.
- The earlier instruction to publish project contents to the public GitHub repository applies to this repo-wide refactor, so push the completed local changes to `origin/main`.
- No tests or experiments are run unless explicitly requested. Verification will use source/entrypoint review and diff/integrity checks that do not execute project workloads.

### Blocking questions

None. The user explicitly requested the whole repo and local code. The assumptions above define a conservative compatibility-preserving scope.

### Intended files and commands

- Create a canonical `wmt25_esa/` Python package containing core scripts and analysis modules. Keep the self-contained Kaggle benchmark adapter and generated Kaggle entrypoint at their required paths.
- Replace existing root script paths with small compatibility entrypoints that delegate to the package; keep Kaggle's required execution path as an adapter.
- Update internal imports, `build_kaggle_kernel.py`, package source selection/materialization, and the README/report command references only where needed. Regenerate `execution/kaggle_kernel/main.py` from the builder; do not hand-edit its embedded payload.
- Keep all data/results/models folders and generated experiment outputs unchanged.
- Use Git moves for code relocation, inspect references, and review `git diff --stat`, staged paths, LFS state, and final Git status before pushing.

### Verification approach

- Compare tracked changes against the approved scope; confirm artifact paths/content are unchanged.
- Review compatibility wrappers, package-relative imports, every documented command, and Kaggle builder inputs/materialization paths.
- Review generated Kaggle output against the generator's canonical inputs.
- Do not run model training, API calls, experiment scripts, or a test suite.
- Confirm local and remote branches agree after push.

## Previous completed request: initial GitHub publication

- Created public `christian-hoang-04/wmt25-esa-score-prediction` and pushed the initial project.
- Uploaded the two oversized files with Git LFS; the working branch was verified clean and tracking `origin/main`.
- Detailed publication notes remain in `implementation-process.md`.

## Refactor result

- Implemented the repo-wide source organization refactor and pushed it to the existing public repository.
- The compatibility entrypoints and generated Kaggle package are included; data, saved models, reports, plots, and run outputs stayed in place and unchanged.
- Static syntax and bundle-structure review passed. No project tests or experiments were run.
- Refactor commit: `f531e831a63a0ed6633faa47f1dd39aa6662c7ee`.
