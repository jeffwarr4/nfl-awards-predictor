# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An NFL MVP predictor: trains on 2000–2025 season stats + AP MVP vote tables, scores the live season weekly,
and publishes a top-10 board (CSV for a Google Sheet, an AI-written chip/why/caption, and a rendered 1080×1350
social graphic). Mirrors the patterns of the sibling `mlb-awards-predictor` repo; headshots and logos come from
the sibling `stinger-assets` repo. The 2026 rebuild rationale and backtests are in
`docs/nfl-2026-evaluation-and-plan.md`.

## Environment

- Venv lives outside the repo: `C:\DevVenvs2\nfl-awards-predictor-venv\Scripts\python.exe`. The in-repo `.venv`
  points at a Python install that no longer exists.
- `pip install -r requirements.txt`. scikit-learn is pinned because `models/*.joblib` are pickled pipelines.
- `ANTHROPIC_API_KEY` enables chips and captions; without it they are skipped with a warning (not an error).
- `STINGER_ASSETS_DIR` overrides the default `../stinger-assets` path for headshots/logos.

## Commands

No test suite or linter. Every script runs as a module from the repo root (imports rely on it):

```bash
python -m src.data_acquisition.fetch_mvp_history   # PFR scrape -> data/raw/mvp_winners.csv (needs lxml)
python -m src.data_acquisition.fetch_player_stats  # nflreadpy, TRAIN_START..TRAIN_END -> data/raw/player_season_stats.csv
python -m src.preprocessing.merge_datasets         # + winners + data/raw/mvp_votes.csv -> data/processed/training_data.csv
python -m src.modeling.train_model                 # both targets; prints leave-one-season-out winner-rank table
python -m src.modeling.train_model --target winner
python -m src.Predictions.predict_current          # live board + candidates + chips + graphic + caption
python -m src.Predictions.predict_current --season 2025 --through-week 4 --outdir <scratch> --no-publish  # replay
python -m src.publish.render_graphics              # re-render from predictions/{year}/top10_mvp_latest.csv
python -m src.publish.generate_chips               # likewise; generate_captions too
```

`mvp_winners.csv` is hand-maintained for recent seasons (2025 Stafford was added by hand); re-running the PFR
scraper overwrites it.

## Architecture

**`config.py` is the single source of truth** — years (`CURRENT_YEAR`, `TRAIN_START`, `TRAIN_END`), all paths,
the feature list, `MODEL_TARGET`, and the in-season knobs. Bump `CURRENT_YEAR`/`TRAIN_END` each season; nothing
else hardcodes a year.

**Feature engineering lives only in `src/features.py`** (`clean_name`, `add_derived_features`, `feature_matrix`),
used by both merge/training and live scoring. `feature_matrix` reindexes to `config.FEATURES` and fills missing
columns with 0 — a feature added to the list but not computed silently becomes 0 rather than erroring.

**Two targets, one live.** `MODEL_TARGET = "vote_share"`: label = received any MVP vote, vote-getters weighted
`1 + 20*share`, `StandardScaler + LogisticRegression(C=0.5, class_weight="balanced")`. `"winner"` is last year's
winner-only model, kept trainable for comparison. Models and metrics are committed in `models/`
(`nfl_mvp_{target}.joblib`, `metrics_{target}.json` with per-season winner rank, `coefficients_{target}.csv`);
the weekly workflow does **not** retrain. Evaluation is leave-one-season-out on a QB/RB board, reported as
winner #1 / top-3 / top-5 / mean rank — AUC is deliberately not reported (~1 positive per season).

**Live scoring (`predict_current.py`)** — order matters:
1. Weekly REG stats → season-to-date per player; team = most recent team; `passing_cpoe` is an
   attempt-weighted mean across weeks, not a sum.
2. Eligibility: QB/RB with games ≥ 60% of *his team's* played games (survives byes; the old `games >= 8`
   gate emptied the board through Week 7).
3. Counting stats **and games** projected to a 17-game pace (`× 17 / team_games`) before scoring; rates
   aren't scaled. Without this the vote-trained model collapses early in the season.
4. Displayed stats are actual season-to-date values, not the projection. QB-only stats are blank for RBs.
5. Ranked by `model_score`; **no MVP % is displayed** (CSV or graphic). Pace-projected scores all sit at 0.99+,
   so a share-of-top-10 % was ~10% for everyone, and a softmax over logits gave #1 ~90% at Week 6. Don't
   reintroduce one without a calibration that holds up in-season.

**Outputs (`predictions/{year}/`)** — `top10_mvp_latest.csv` is the Google Sheet import; its column order is
`BOARD_COLS` in `predict_current.py` (a contract — don't rename casually). Plus 4 rolling timestamped archives,
`chip_data/`, `graphics/mvp_board_{year}_wk{NN}.png` + `mvp_board_latest.png`, `graphics/caption_mvp.txt`.
`.prev_top10_mvp.csv` is the movement baseline: it is rolled forward only when `through_week` advances, so a
same-week rerun keeps last week's arrows. Deleting it resets every row to NEW.

**stinger-assets contract.** `player_key = firstname_lastname_team` (nflreadpy team code, lowercased) and must
equal `safe_filename(f"{name} {team}")` in `stinger-assets/sync_espn_headshots.py` (apostrophes/periods dropped,
hyphens split). `predictions/nfl_candidates.csv` (top 25 eligible, not just top 10, so headshots exist before a
player climbs) is pulled by stinger-assets' Monday sync. Logos are ESPN-named and lowercase; nflreadpy `LA`/`WAS`
map to ESPN `lar`/`wsh` (`src/publish/stats.py: ESPN_TEAM_CODES`). The runner is case-sensitive; Windows isn't.

**Training-data gotchas.** nflverse labels 2000–2002 player rows with current franchise codes (`LV`/`LA`/`LAC`)
while schedules keep `OAK`/`STL`/`SD`; `fetch_player_stats.RELOCATED_TEAMS` bridges the win% join (previously
611 player-seasons, incl. Faulk 2000 and Gannon 2002, had win% silently zeroed). Same-name collisions in a season
(two Adrian Petersons in 2008) keep vote/winner labels only on the higher-volume player.

**Legacy.** `outputs/` and `docs/outputs/` (GitHub Pages copies of the 2025 Canva CSVs) are no longer written.

## Automation

`.github/workflows/weekly-mvp.yml`: Tuesdays 15:17 UTC (kept off :00, when GitHub delays scheduled runs), Sep–Feb, plus manual dispatch. Checks out
stinger-assets as a sibling (`STINGER_ASSETS_DIR`), runs `predict_current`, fails if the board is empty (the
script also exits non-zero before overwriting the last good board), uploads CSV/PNG/TXT artifacts, and commits
`predictions/` with `[skip ci]`. Needs the `ANTHROPIC_API_KEY` secret for chips/captions.
