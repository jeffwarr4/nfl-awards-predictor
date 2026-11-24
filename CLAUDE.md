# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An NFL MVP predictor: trains on 2000–2025 season stats + AP MVP vote tables, scores the live season weekly,
and publishes a top-10 board (CSV for a Google Sheet, an AI-written chip/why/caption, and a rendered 1080×1350
social graphic), a 30-deep board CSV, and a second "On the Bubble" slide for ranks 11–15. Mirrors the patterns of the sibling `mlb-awards-predictor` repo; headshots and logos come from
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
python -m src.publish.render_reel [--outdir DIR] [--frames-only] [--no-script]   # weekly reel from a board folder
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
the weekly workflow does **not** retrain. Evaluation is leave-one-season-out on a QB/RB/WR board (`BOARD_POSITIONS`), reported as
winner #1 / top-3 / top-5 / mean rank — AUC is deliberately not reported (~1 positive per season).

**Live scoring (`predict_current.py`)** — order matters:
1. Weekly REG stats → season-to-date per player; team = most recent team (logo, label); `passing_cpoe` is an
   attempt-weighted mean across weeks, not a sum. `team_games` and `win_pct` come from `player_records`, by
   stint: a traded player gets his old team's games and results for his weeks there, then the new team's
   (2026 Cousins had been given MIN's 7-2 for nine Raiders games). Players on one team are unaffected.
2. Eligibility: QB/RB/WR with games ≥ 60% of *his team's* played games (survives byes; the old `games >= 8`
   gate emptied the board through Week 7). WRs were added in 2026: receiving yards carry the largest positive
   coefficient, and adding them to the 2000–2025 backtest changed no winner's rank (record-pace WRs land ~8–15).
3. Counting stats **and games** projected to a 17-game pace (`× 17 / team_games`) before scoring; rates
   aren't scaled. Without this the vote-trained model collapses early in the season.
4. Displayed stats are actual season-to-date values, not the projection. QB-only stats are blank for RBs.
5. Ranked by `model_score`; **no MVP % is displayed** (CSV or graphic). Pace-projected scores all sit at 0.99+,
   so a share-of-top-10 % was ~10% for everyone, and a softmax over logits gave #1 ~90% at Week 6. Don't
   reintroduce one without a calibration that holds up in-season.

**Outputs (`predictions/{year}/`)** — `top10_mvp_latest.csv` is the Google Sheet import; its column order is
`BOARD_COLS` in `predict_current.py` (a contract — don't rename casually). `board_mvp_latest.csv` is the same
columns `BOARD_DEPTH` (30) deep — a 2010–2025 replay showed every eventual winner top-15 by Week 8 and top-30
covering 98–100% of top-5 vote finishers from Week 4. Each has 4 rolling timestamped archives. After chips run,
`board_mvp_latest.csv` gets trailing `chips` (top 3, pipe-separated) and `why_blurb` (top-3 and bubble Whys) columns;
the stingercollectibles.com NFL page and home Trending read that file straight from GitHub (no Google Sheet).
`top10_mvp_latest.csv` keeps `BOARD_COLS` only: the Sheet still imports it for the old Squarespace page until cutover. Plus
`chip_data/`, `graphics/mvp_board_{year}_wk{NN}.png` + `mvp_board_latest.png`,
`graphics/bubble_board_{year}_wk{NN}.png` + `bubble_board_latest.png`, `graphics/caption_mvp.txt` (covers
the two-slide carousel). `.prev_board_mvp.csv` (30 rows) is the movement baseline: it is rolled forward only
when `through_week` advances, so a same-week rerun keeps last week's arrows. Deleting it resets every row to NEW.
Until it first exists, the old 10-row `.prev_top10_mvp.csv` is read instead (and deleted on the first roll).

**On the Bubble slide.** `render_bubble_graphic` pastes circular headshots for `BUBBLE_RANKS` into
`MVP Bubble NFL.png`, composites `MVP Bubble NFL overlay.png` (rings, gloss) over them, then draws badges and
labels. All geometry comes from `assets/bubble_slots.json`. `src/publish/make_bubble_bg.py` regenerates the
background, overlay and JSON and renders a reference preview. Bubble players get a Why (≤75 chars, enforced
in code; wrapped to two lines on the slide) but no chips, in the same `chip_data_latest.csv`.

**stinger-assets contract.** `player_key = firstname_lastname_team` (nflreadpy team code, lowercased) and must
equal `safe_filename(f"{name} {team}")` in `stinger-assets/sync_espn_headshots.py` (apostrophes/periods dropped,
hyphens split). `predictions/nfl_candidates.csv` (top 30 eligible — everyone on the saved board — so headshots
exist before a player climbs) is pulled by stinger-assets' daily sync (11:37 UTC, before the Tuesday run). Logos are ESPN-named and lowercase; nflreadpy `LA`/`WAS`
map to ESPN `lar`/`wsh` (`src/publish/stats.py: ESPN_TEAM_CODES`). The runner is case-sensitive; Windows isn't.

**Training-data gotchas.** nflverse labels 2000–2002 player rows with current franchise codes (`LV`/`LA`/`LAC`)
while schedules keep `OAK`/`STL`/`SD`; `fetch_player_stats.RELOCATED_TEAMS` bridges the win% join (previously
611 player-seasons, incl. Faulk 2000 and Gannon 2002, had win% silently zeroed). Same-name collisions in a season
(two Adrian Petersons in 2008) keep vote/winner labels only on the higher-volume player.

**Legacy.** `outputs/` and `docs/outputs/` (GitHub Pages copies of the 2025 Canva CSVs) are no longer written.

**Weekly reel (`src/publish/render_reel.py`).** One story per week as a silent 1080×1920 MP4 (20 s + 1 s hold)
plus a narration script Jeff voices over in Instagram (he adds the Captions sticker there; nothing is burned in);
posted first, the afternoon the board completes, with the static board the next day (order reversed
2026-11-24). Reworked Nov 2026 for retention, matching nba-awards-predictor: the first reels lost
viewers in the hook (NHL: ~6 of 27 s watched), so there are four scenes, `REEL_SCENES` hook 0–3 / proof 3–13 /
context 13–16 / ask 16–20. Frame 0 already shows the hook text, headshot, name and last week's rank (no fade-ins),
and the rank flips to this week's by 0.8 s. The reel ends on a comment question, not a spoken URL (the URL is small
on screen and ends the Instagram caption).
- `reel_story.pick_story` (pure; `tests/test_reel_story.py`, run `python -m pytest tests/`) takes the first match:
  new No. 1, biggest climber into the top 5, 11+ into the top 10, biggest drop by a former top-5 player, else a
  profile of No. 1 (always on the first board, when everyone is NEW). Players are matched by `player_id`. A No. 1
  who held the spot on an earlier weekly board (`board_mvp_wkNN.csv`) is "back on top", not new; the script also gets
  the featured player's rank on each earlier weekly board (`load_history`, `rank_history`). Weeks 6–9 of 2026 were
  backfilled from git (each week's last board; Week 6 is the 10-row top 10, from before the 30-deep board). Hooks are
  short and lead with the number; for a climb, `season_lows` makes the hook his path when it's bigger than last
  week's move and at least 2 weeks back ("No. 30 → No. 3 in 3 weeks"; `hook_line_img` draws the "→").
- `reel_script.py` writes four timed beats in Jeff's first-person voice on `REEL_MODEL` (Opus 5.5, with the API's
  refusal fallback; chips stay on Haiku), plus the on-screen `question`, the Instagram caption and 5 hashtags.
  Records are W-L by stint (`generate_chips._record`). The script explains the move rather than reciting stats: it
  gets a then-vs-now comparison (`comparison_row`: his row on the path hook's week, else the earliest weekly board
  within `REEL_LOOKBACK_WEEKS`; `then_now_lines`: rank, team record and the record since then (left out for a
  player traded since), his rates (`stats.RATE_COLS`: rating, TD to INT ratio, pass YPG for QBs; yards per game for
  RBs/WRs) and the touchdowns and yards since then) plus `MODEL_NOTE`, a plain-words summary of what the model
  weighs (`models/coefficients_vote_share.csv`). It says "TD to INT ratio", never "TD:INT". `problems()` checks
  `REEL_WORDS`, the hook (≤ `REEL_HOOK_MAX_WORDS`, no "our model"), the proof (≤ `REEL_PROOF_MAX_WORDS`), that the
  last beat (≤ `REEL_ASK_MAX_WORDS`) ends with "?", no filler or spoken link, no odds/betting language or percent
  (a completion % is allowed, win % never: `probability_language`), and that the caption ends with "Full board at
  stingercollectibles.com".
  One rewrite request; then `finish()` swaps in `fallback_question()` (last name, suffix dropped: "Is Mahomes your
  MVP right now?"). Tests: `tests/test_reel_script.py`.
- The Proof scene: with a comparison, a "WEEK 8 → WEEK 11" label and 3 then → now cards (`then_now_cards`: the
  team record unless he was traded, then the position's rates that moved most, by relative change; the NFL model's
  passing terms overlap, so cards aren't ranked by model weight as the NBA reel's are). Without one (no earlier
  board), 3 position-aware stat cards (`stat_cols_for` + `select_top_stats`, win % left out), the team record and
  the board's `why_blurb`. Labels come from `stats.REEL_LABELS`. Scene timings, safe
  zones and thresholds are `REEL_*` in `config.py`. Frames are Pillow, encoded by imageio-ffmpeg's bundled ffmpeg
  (no system ffmpeg). Needs anthropic 2.x (`fallbacks`).
- Outputs in `predictions/{year}/reel/`: `reel_script.txt` and `reel_meta.json` are committed (the meta keeps the
  keys stinger-website reads and adds `question`); `reel_latest.mp4` is gitignored and goes to the repo's
  `reel-latest` release as `nfl_reel_{year}_wk{NN}.mp4` (`release_asset` in the meta). stinger-website's
  social-drafts job emails it from there. `--frames-only` saves frame 0 and the end of each scene, with the platform
  UI zones shaded, for layout checks. A replay folder needs the previous week replayed into it first, or everyone is
  NEW and the fallback is picked.
- One-off reels (a correction, a feature): `render_reel --story predictions/{year}/reel/story_<name>.json` names the
  board player and the hook, and can set before/after labels, the Proof cards, a note for the script, and a
  hand-written script (four beats, `question`, caption, hashtags; checked like a generated one). It writes
  `reel_<name>.*` beside the weekly files without touching them; upload the MP4 to the release by hand (the next
  weekly upload removes it). The 2026 story files (`story_cousins_correction.json`, `story_purdy_back.json`) use the
  old five-beat script and are history; rewrite one before rendering it again.

**Weekly and final boards (`scripts/snapshot_board.py`, the same file in all four awards repos).** The
workflow's "Keep this week's board" step, just before the commit, copies `board_mvp_latest.csv` to `board_mvp_wkNN.csv` (NN = `through_week`), which is never
pruned (`ARCHIVES_TO_KEEP` only prunes the timestamped run files). A same-week rerun overwrites that copy. After the
season's last regular-season board, run `python scripts/snapshot_board.py --final` once and commit: it writes
`board_mvp_final.csv`, which the website's season archive pages read. Don't edit or delete the weekly or final files; they're the
season history.

## Automation

`.github/workflows/weekly-mvp.yml`: started Tuesdays 15:17 UTC, Sep–Feb, by the stinger-scheduler Cloudflare Worker (`stinger-website/scheduler`); its own cron (Tuesdays 17:17 UTC) is a late backup. Plus manual dispatch. Checks out
stinger-assets as a sibling (`STINGER_ASSETS_DIR`), runs `predict_current`, fails if the board is empty (the
script also exits non-zero before overwriting the last good board), uploads CSV/PNG/TXT artifacts, and commits
`predictions/` (plus `git add -f` of `.prev_board_mvp.csv`) with `[skip ci]`. Needs the `ANTHROPIC_API_KEY` secret for chips/captions.

A `guard` job runs first: a scheduled start (the cron, or a dispatch with `scheduled=true`, which is how the stinger-scheduler Cloudflare Worker starts it) is skipped when a scheduled run already succeeded in the last 20 hours, so a late cron after an on-time run is a no-op. Manual runs from the Actions tab (`scheduled` left false) always run. Run names end in `(scheduled)` or `(manual)`; the guard relies on that suffix.

After the board is written, `stinger-assets/sync_espn_headshots.py fetch --strict` downloads any missing headshot for
ranks 1–15 (a player new to the slides isn't in stinger-assets until its daily sync reads the committed candidates
file, i.e. after this run), and the slides are re-rendered if it downloaded any. If one still can't be found the run
fails before committing, so a slide with a blank circle never reaches the site or the social draft: add the PNG to
stinger-assets by hand (or fix the team alias there) and rerun.

The prediction step runs `predict_current --no-reel`; the reel is its own step after the headshots (so a newly
fetched headshot is in it), then uploaded to the `reel-latest` release (older weeks' assets deleted). Both are
`continue-on-error`, so a reel failure never blocks the board commit. Locally, `predict_current` renders the reel
after the caption unless `--no-reel` or `--no-publish`.
