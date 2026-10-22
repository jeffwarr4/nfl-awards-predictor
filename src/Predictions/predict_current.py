"""
Score the live NFL season and write the weekly MVP board.

Pipeline
1. Weekly regular-season stats (nflreadpy) → season-to-date totals per player.
   passing_cpoe is an attempt-weighted mean across weeks, not a sum.
2. Team record from played regular-season games only.
3. Eligibility: QB/RB who played ≥ MIN_TEAM_GAME_SHARE of his team's games
   (works from Week 1 and survives byes, unlike a fixed games floor).
4. Counting stats (incl. games) projected to a SEASON_GAMES pace before scoring —
   the model was trained on full seasons; raw partial totals collapse it.
5. Score with config.MODEL_TARGET and rank. No probability is displayed: with
   pace-projected stats the top-10 scores all sit at 0.99+, so any % shown
   (share of total ≈ 10% each) carried no information. model_score is the raw output.

Outputs (predictions/{year}/)
  top10_mvp_latest.csv       ← Google Sheet import
  top10_mvp_{ts}.csv         ← archive (ARCHIVES_TO_KEEP most recent kept)
  .prev_top10_mvp.csv        ← previous week's board; drives rank_delta / movement
  chip_data/, graphics/      ← chips, rendered board PNG, caption (see src/publish/)
predictions/nfl_candidates.csv ← headshot candidates fetched by stinger-assets

Run:
    python -m src.Predictions.predict_current
    python -m src.Predictions.predict_current --season 2025 --through-week 4   # replay
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
import nflreadpy as nfl

from config import (
    CURRENT_YEAR, FEATURES, MODEL_TARGET, model_path, PREDICTIONS, CANDIDATES_CSV,
    SEASON_GAMES, MIN_TEAM_GAME_SHARE, BOARD_POSITIONS, TOP_N, CANDIDATE_POOL,
    ARCHIVES_TO_KEEP,
)
from src.features import add_derived_features, feature_matrix


# Season totals that scale with games played. Rates (passing_cpoe, win_pct) are not projected.
COUNTING_COLS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
    "passing_epa", "rushing_yards", "rushing_tds", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_epa",
]

# Columns written to top10_mvp_latest.csv, in order (the Sheet import contract)
BOARD_COLS = [
    "player_key", "rank", "rank_delta", "movement", "player_name", "team", "position",
    "model_score",
    "games", "team_games", "win_pct", "passer_rating", "total_tds", "td_int",
    "total_yards", "pass_ypg", "rush_yards",
    "season", "through_week", "updated_utc", "player_id",
]


# ------------------------
# Keys
# ------------------------
def _headshot_key(name: str, team: str) -> str:
    """firstname_lastname_team slug for stinger-assets headshots.
    Must match safe_filename(f"{name} {team}") in sync_espn_headshots.py:
    accents stripped, apostrophes and periods dropped, hyphens split.
    e.g. 'C.J. Stroud', 'HOU' → 'cj_stroud_hou'; "Ja'Marr Chase", 'CIN' → 'jamarr_chase_cin'
    """
    n = unicodedata.normalize("NFKD", f"{name} {team}")
    n = "".join(ch for ch in n if not unicodedata.combining(ch)).lower()
    n = n.replace("'", "").replace(".", "").replace("-", " ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", n)).strip("_")


# ------------------------
# Data
# ------------------------
def load_weekly(season: int, through_week: int | None) -> pd.DataFrame:
    w = nfl.load_player_stats([season]).to_pandas()
    w = w[w["season_type"] == "REG"]
    if through_week is not None:
        w = w[w["week"] <= through_week]
    return w


def team_records(season: int, through_week: int | None) -> pd.DataFrame:
    sched = nfl.load_schedules([season]).to_pandas()
    played = sched[(sched["game_type"] == "REG")
                   & sched["home_score"].notna() & sched["away_score"].notna()]
    if through_week is not None:
        played = played[played["week"] <= through_week]
    games = pd.concat([
        pd.DataFrame({"team": played["home_team"],
                      "win": (played["home_score"] > played["away_score"]).astype(int)}),
        pd.DataFrame({"team": played["away_team"],
                      "win": (played["away_score"] > played["home_score"]).astype(int)}),
    ])
    rec = games.groupby("team").agg(team_games=("win", "size"), wins=("win", "sum")).reset_index()
    rec["win_pct"] = rec["wins"] / rec["team_games"]
    return rec


def season_to_date(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per player: summed counting stats, games, attempt-weighted CPOE,
    and the team he most recently played for."""
    w = weekly.sort_values("week")
    w = w.assign(_cpoe_x_att=w["passing_cpoe"] * w["attempts"],
                 _cpoe_att=w["attempts"].where(w["passing_cpoe"].notna(), 0))
    g = w.groupby("player_id")
    out = g[COUNTING_COLS + ["_cpoe_x_att", "_cpoe_att"]].sum(min_count=1)
    out["games"] = g["game_id"].nunique()
    last = g[["player_display_name", "position", "team", "headshot_url"]].last()
    out = last.join(out).reset_index()
    out["passing_cpoe"] = out["_cpoe_x_att"] / out["_cpoe_att"].replace(0, np.nan)
    return out.drop(columns=["_cpoe_x_att", "_cpoe_att"])


def project_to_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Scale counting stats and games to a SEASON_GAMES pace by team games played."""
    df = df.copy()
    factor = SEASON_GAMES / df["team_games"]
    for c in COUNTING_COLS + ["games"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") * factor
    return df


def passer_rating(df: pd.DataFrame) -> pd.Series:
    att = pd.to_numeric(df["attempts"], errors="coerce").replace(0, np.nan)
    comp, yds, tds, ints = (pd.to_numeric(df[c], errors="coerce").fillna(0) for c in
                            ("completions", "passing_yards", "passing_tds", "passing_interceptions"))
    parts = [((comp / att) - 0.3) * 5, ((yds / att) - 3.0) * 0.25,
             (tds / att) * 20.0, 2.375 - ((ints / att) * 25.0)]
    return (sum(p.clip(0.0, 2.375) for p in parts) / 6.0 * 100.0).round(1)


# ------------------------
# Board
# ------------------------
def build_board(season: int, through_week: int | None = None) -> tuple[pd.DataFrame, int]:
    """All eligible QB/RB, scored and ranked. Returns (board, through_week)."""
    weekly = load_weekly(season, through_week)
    if weekly.empty:
        return pd.DataFrame(), 0
    week = int(weekly["week"].max())

    players = season_to_date(weekly).merge(team_records(season, week), on="team", how="left")
    players = players[players["position"].isin(BOARD_POSITIONS) & players["team_games"].gt(0)]
    players = players[players["games"] / players["team_games"] >= MIN_TEAM_GAME_SHARE]
    if players.empty:
        return pd.DataFrame(), week

    scored = add_derived_features(project_to_pace(players))
    model = load(model_path(MODEL_TARGET))
    players = players.assign(model_score=model.predict_proba(feature_matrix(scored, FEATURES))[:, 1])

    # Display stats are actual season-to-date values, not the pace projection.
    actual = add_derived_features(players)
    is_qb = actual["position"] == "QB"
    players["passer_rating"] = passer_rating(actual).where(is_qb)
    players["td_int"] = actual["td_int_ratio"].round(2).where(is_qb)
    players["pass_ypg"] = actual["pass_ypg"].round(1).where(is_qb)
    players["rush_yards"] = actual["rushing_yards"].fillna(0).astype(int)
    players["total_yards"] = actual[["passing_yards", "rushing_yards", "receiving_yards"]].fillna(0).sum(axis=1).astype(int)
    players["total_tds"] = actual[["passing_tds", "rushing_tds", "receiving_tds"]].fillna(0).sum(axis=1).astype(int)
    players["win_pct"] = players["win_pct"].round(3)
    players["games"] = players["games"].astype(int)
    players["team_games"] = players["team_games"].astype(int)

    board = players.rename(columns={"player_display_name": "player_name"})
    board = board.sort_values("model_score", ascending=False).reset_index(drop=True)
    board["rank"] = board.index + 1
    board["player_key"] = [_headshot_key(n, t) for n, t in zip(board["player_name"], board["team"])]
    board["season"] = season
    board["through_week"] = week
    return board, week


def add_movement(top: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
    """rank_delta: +up / -down vs last week ("" if NEW); movement: ↑3 / ↓2 / – / NEW."""
    top = top.copy()
    prev = {}
    if prev_path.exists():
        p = pd.read_csv(prev_path)
        prev = dict(zip(p["player_id"].astype(str), p["rank"].astype(int)))
    deltas, moves = [], []
    for pid, rank in zip(top["player_id"].astype(str), top["rank"]):
        if pid not in prev:
            deltas.append(""); moves.append("NEW")
        else:
            d = prev[pid] - int(rank)
            deltas.append(d)
            moves.append(f"↑{d}" if d > 0 else (f"↓{abs(d)}" if d < 0 else "–"))
    top["rank_delta"] = deltas
    top["movement"] = moves
    return top


def roll_prev_snapshot(latest_path: Path, prev_path: Path, week: int) -> None:
    """The movement baseline is the last board from an EARLIER week, so a
    same-week rerun (manual dispatch, a fix) doesn't reset movement to '–'."""
    if not latest_path.exists():
        return
    latest = pd.read_csv(latest_path)
    if "through_week" in latest.columns and len(latest) and int(latest["through_week"].iloc[0]) < week:
        latest[["player_id", "player_key", "rank", "through_week"]].to_csv(prev_path, index=False)


def write_candidates(board: pd.DataFrame, path: Path) -> None:
    pool = board.head(CANDIDATE_POOL)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sport": "NFL", "player_name": pool["player_name"],
                  "team": pool["team"], "player_key": pool["player_key"]}) \
      .drop_duplicates("player_key").to_csv(path, index=False)
    print(f"Headshot candidates -> {path} ({len(pool)} players)")


# ------------------------
# Main
# ------------------------
def run(season: int = CURRENT_YEAR, through_week: int | None = None,
        outdir: Path | None = None, publish: bool = True) -> Path:
    default_out = PREDICTIONS / str(season)
    outdir = Path(outdir or default_out)
    outdir.mkdir(parents=True, exist_ok=True)
    # A replay/dry run into another folder must not overwrite the live candidates feed.
    candidates = CANDIDATES_CSV if outdir.resolve() == default_out.resolve() else outdir / CANDIDATES_CSV.name

    print(f"Scoring {season} (model: {MODEL_TARGET})...")
    board, week = build_board(season, through_week)
    if board.empty:
        # Fail loudly — never overwrite the last good board with an empty one.
        raise SystemExit(f"ERROR: empty MVP board for {season} through week {week} — "
                         f"no eligible {'/'.join(BOARD_POSITIONS)} found.")

    top = board.head(TOP_N).copy()
    top["model_score"] = top["model_score"].round(6)
    top["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    latest = outdir / "top10_mvp_latest.csv"
    prev = outdir / ".prev_top10_mvp.csv"
    roll_prev_snapshot(latest, prev, week)
    top = add_movement(top, prev)[BOARD_COLS]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    top.to_csv(latest, index=False)
    top.to_csv(outdir / f"top10_mvp_{ts}.csv", index=False)
    for old in sorted(outdir.glob("top10_mvp_2*.csv"))[:-ARCHIVES_TO_KEEP]:
        old.unlink()

    print(f"\n=== {season} NFL MVP — through Week {week} ===")
    show = ["rank", "movement", "player_name", "team", "position",
            "games", "team_games", "win_pct", "passer_rating", "total_tds", "total_yards"]
    print(top[show].to_string(index=False))
    print(f"\nSaved -> {latest}")

    if publish:
        write_candidates(board, candidates)
        publish_outputs(top, outdir, season, week)
    return latest


def publish_outputs(top: pd.DataFrame, outdir: Path, season: int, week: int) -> None:
    """Chips (needs ANTHROPIC_API_KEY) → board graphic → captions (needs key)."""
    from src.publish.generate_chips import generate_chips
    from src.publish.render_graphics import render_board
    from src.publish.generate_captions import generate_caption_file

    chip_df = generate_chips(top, outdir)
    render_board(top, chip_df, outdir / "graphics", season, week)
    generate_caption_file(top, chip_df, outdir / "graphics", season, week)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # ↑/↓ arrows on a cp1252 console
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=CURRENT_YEAR)
    parser.add_argument("--through-week", type=int, default=None,
                        help="replay the board as of this week (default: all played weeks)")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-publish", action="store_true",
                        help="skip candidates, chips, graphic and captions")
    args = parser.parse_args()
    run(args.season, args.through_week, args.outdir, publish=not args.no_publish)


if __name__ == "__main__":
    main()
