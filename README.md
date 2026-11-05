# OIBC personal research

Personal solar forecasting experiments. Initial group-direct scripts originate from the #1314 v0.2 team worktree at commit c955267; the individual-plant baseline and competition datasets are separate prerequisites and are not included.

## Data and evaluation

Keep all competition data, predictions, OOFs and derived files outside Git. Do not use locked HOLDOUT data for model selection or tune on leaderboard outcomes. Internal settlement calculations are provisional, not official scores. The archived v1 four-fold results and v2 two-fold reproducibility evidence are different; do not conflate them.

## Scripts

`scripts/oibc_1314/` contains historical feature preparation, DEV evaluation, blending, and forecast export scripts. They require separately supplied approved inputs and a compatible Python environment. Some previous inputs originate from team work; availability and reuse rights must be established before relying on them. Inspect each script and its date filters before running. No model is trained by cloning this repository.

Claude and Codex share the local status board at `C:\dev\icn-solar-oibc\STATUS.md` (local only, never commit).
