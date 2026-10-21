# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An NFL MVP predictor that trains on historical season stats (2000–2024) + AP MVP winners, scores the current season, and publishes "MVP Watch" CSVs that feed Canva graphics via GitHub Pages (`https://jeffwarr4.github.io/nfl-awards-predictor`).

## Commands

There is no test suite, linter, or build step. Every script is run as a module from the repo root (imports and relative paths depend on it):

```bash
python -m src.data_acquisition.fetch_mvp_history   # scrape PFR -> data/raw/mvp_winners.csv (needs lxml)
python -m src.data_acquisition.fetch_player_stats  # nflreadpy -> data/raw/player_season_stats.csv
python -m src.preprocessing.merge_datasets         # -> data/processed/training_data.csv
python -m src.modeling.train_model                 # -> models/*.pkl, outputs/training_metrics.json, feature importance
python -m src.Predictions.predict_current          # -> outputs/*.csv for the current SEASON
python src/assests/download_headshots.py           # newest Canva CSV -> assets/headshots/*.png
```

Dependencies: there is no `requirements.txt`; `requirements.freeze.txt` is a local pip freeze. Core libs are pandas, numpy, scikit-learn, joblib, nflreadpy, polars, pyarrow (plus requests/tqdm for headshots). A local `.venv` exists.

## Pipeline architecture

Linear, file-based pipeline — each stage reads the previous stage's CSV:

1. **Acquisition** (`src/data_acquisition/`): historical season-level stats for QB/RB/WR via `nflreadpy` (polars), joined with team win% computed from schedules. `YEARS` is hard-coded in `fetch_player_stats.py`.
2. **Merge** (`src/preprocessing/merge_datasets.py`): left-joins winners onto players by `(season, normalized name)` to produce the binary `mvp_winner` target.
3. **Train** (`src/modeling/train_model.py`): LogisticRegression and HistGradientBoosting, both `class_weight="balanced"`, evaluated with `GroupKFold` by season and season-level top-1/top-3 hit rates. Writes `outputs/training_metrics.json`, whose `features_used` list is **the contract** for inference.
4. **Predict** (`src/Predictions/predict_current.py`): loads *weekly* current-season stats, aggregates to season totals, computes win% from played games only, rebuilds features, then `reindex`es to `features_used`. Uses **only the logreg model**. Applies presentation filters (games ≥ 8, QB/RB only, RB weight 0.85 → `adjusted_pred`) and writes ranked outputs.

### Logic duplicated across files — keep in sync
- `_clean_name` (name-key normalization) exists in both `merge_datasets.py` and `predict_current.py`.
- `_add_derived_features` exists in both `train_model.py` and `predict_current.py` (`td_int_ratio`, position flags, `win_pct_scaled`, `games_weighted_win`, `efficiency_win`). The predict version additionally computes per-game yards. A feature added to training must be added to prediction too, or it silently becomes 0 via `reindex(fill_value=0)`.

### Outputs consumed downstream
`predict_current.py` writes to `outputs/`: `mvp_predictions_{SEASON}_{all,top10,top5}.csv`, `mvp_watch_canva_{SEASON}.csv`, a dated `mvp_watch_canva_top10_week{ISO week}_{date}.csv`, and single-row wide pivots `mvp_watch_top5_pivot.csv` / `mvp_watch_6to10_pivot.csv` (`rank{K}_<field>` columns used as Canva bulk-create data). Column names in these files are a de-facto external API for the Canva templates — don't rename them casually. All pivots/top-N must come from the single ranked `display` frame so orderings agree.

Team logos are served from `assets/logos/{TEAM}.png` via the GitHub Pages URL in `CDN_BASE`; headshot URLs come from nflreadpy.

## Automation

`.github/workflows/weekly-mvp.yml` runs Tuesdays 15:00 UTC (and on dispatch): retrains, predicts, copies `outputs/*.csv` into `docs/outputs/` (the GitHub Pages copy), and commits as "Weekly MVP Watch auto-update". So `outputs/` and `docs/outputs/` are mirrored, bot-committed artifacts; expect frequent upstream commits touching them.

## Gotchas
- `SEASON = 2025` in `predict_current.py` must be bumped each year; training `YEARS` likewise.
- The headshot script lives in the misspelled `src/assests/` and uses CWD-relative paths.
- `fetch_mvp_history.py` has a mis-indented block (the column `keep_cols`/rename sits inside the `if player_col is None` branch after `raise`, so it never runs) — the rename to `season`/`player_name` won't happen on fresh scrapes.
- The Canva CSV currently contains a duplicated `headshot_url` column (added both explicitly in `display_cols` and via `headshot_cols`); `download_headshots.py` tolerates this.
