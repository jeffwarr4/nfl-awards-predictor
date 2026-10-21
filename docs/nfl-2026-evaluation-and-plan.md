# NFL MVP Model — 2025 Evaluation & 2026 Build Plan
_Written 2026-10-21 (2026 season through Week 6)_

## How 2025 went
Final model board (12/30 → 1/6): Maye #1, Allen #2, Stafford #3.
Actual AP vote: Stafford 366 pts (24 first-place), Maye 361 (23), Allen 91, McCaffrey 71, Lawrence 49.
Model had the right top 3 but ranked the winner 3rd. Lawrence (5th in voting) was #22 in the model.

Backtest (leave-one-season-out, 2000–2025, full-season stats, QB/RB board):
| model | top-1 | top-3 | top-5 |
|---|---|---|---|
| Last year's features (season totals + win% interactions) | 54% | 81% | 85% |
| Rate/percentile variant (tested, rejected) | 38% | 77% | 92% |

The rate-based variant was my hypothesis for fixing partial-season scoring; it lost. Keep the existing feature set.

In-season rank of eventual winner (last year's model, trained without that season):
- 2024 Allen: #2–#3 from Week 4 on, #1 at Week 8.
- 2025 Stafford: #8–#9 early, #1 at Week 12, finished #3.
- 2023 Lamar: never better than #4 — stats-only ceiling for narrative/rushing seasons.

## Problems found
1. **Empty output this season until Week 8.** `games >= 8` gate → zero rows at Week 6 (verified by running it). Also drops a player mid-season after a bye (Stafford vanished at Week 8 in the 2025 replay).
2. **Probabilities saturate** (Maye 100%, Allen 99%) — class_weight='balanced' logreg; display % is not meaningful.
3. **Google Sheet `web_current_top10` shows 2025 final stats stamped "As of 10/21/2026".**
4. `SEASON`/`YEARS` hardcoded in two files; 2025 winner not in `mvp_winners.csv`; `fetch_mvp_history.py` rename block is mis-indented (never runs).
5. No movement tracking, no player_key, no chips/why automation, Canva bulk-create manual step, headshots/logos on Squarespace by hand.
6. stinger-assets NFL gaps: no NFL candidates source in `merge_candidates.py` (commented URL points at a non-existent `nfl-predictor` repo); NFL logos mixed case (9 uppercase, 24 lowercase — breaks on the case-sensitive runner); nflreadpy codes `LA`/`WAS` vs ESPN `lar`/`wsh` need aliases; old NFL headshots are `First_Last.png`, not `player_key`, and none are in `espn_headshots_map.csv`.

## 2026 build plan (port of mlb-awards-predictor patterns)
1. `config.py` single source of truth (CURRENT_YEAR=2026, TRAIN 2000–2025), `requirements.txt`, add 2025 Stafford, fix scraper indent.
2. Eligibility: played ≥ 60% of team games (works from Week 1, survives byes) instead of `games >= 8`.
3. Model: keep current features, retrain through 2025. Experiment: MLB-style top-5 vote-finish target from PFR voting tables (5× the positive labels). Display probability as share of the top-10 total, not raw logreg output.
4. Outputs like MLB: `top10_latest.csv` + 4 timestamped archives, `.prev_top10.csv` → rank_delta/movement, `player_key` = `first_last_team`, `predictions/nfl_candidates.csv`.
5. stinger-assets: add NFL candidates URL, NFL team alias map, lowercase logo filenames, re-sync NFL headshots under player_key.
6. Port `generate_chips.py` + `generate_captions.py` (NFL stat cols, #StingerNFL, team hashtags).
7. Port `render_graphics.py` to a single NFL board on Jeff's background (1080×1350): top-3 cards + ranks 4–10 rows.
8. Workflow: Tuesday after MNF, sibling checkout of stinger-assets, ANTHROPIC_API_KEY secret, fail loudly on empty board.

## Top-5 vote-finish experiment (run 2026-10-21)
Vote tables for 2000–2025 pulled from Pro-Football-Reference (`nfl_mvp_votes_2000_2025.csv`, 113 offensive vote-getters; defensive players dropped since they're not in the training data). Caveat: before 2022 the AP used one vote per voter, so older seasons have only 1–6 vote-getters; 2022+ has true 5-deep ballots.

Same 25 features as last year's model; only the training target changes. Leave-one-season-out 2000–2025:
| target | winner #1 | top 3 | top 5 | mean rank | top-5 overlap 2022–25 |
|---|---|---|---|---|---|
| A winner (last year) | 54% | 81% | 85% | 3.3 | 3.8 / 5 |
| B top-5 finish | 50% | 81% | 88% | 2.5 | 3.8 |
| C any votes | 46% | 85% | 96% | 2.2 | 3.8 |
| D vote-share regression | 46% | 85% | 85% | 11.8 | 3.2 |
| **E any votes, weighted by vote share** | **58%** | **85%** | **96%** | **2.0** | 3.8 |
| F blend A+B | 50% | 77% | 85% | 2.7 | 3.8 |

In-season (2023–25, Weeks 4–14): the top-5 target collapses on raw partial-season totals (Stafford #103 at Week 4) — a scale artifact. Projecting stats to 17-game pace fixes it (#15). With pace projection, E is roughly even with last year's model in-season (mean winner rank 6.4 vs 5.2, top-3 50% vs 44%); 18 data points, dominated by Lamar 2023, so not separable.

Decision: use E (share-weighted any-votes target) + 17-game pace projection for weekly scoring. Gain is modest on the winner call and clearer on the whole board (winner in top 5 in 25/26 seasons vs 22/26).
