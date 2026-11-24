"""Board stat columns, display labels and formatting shared by chips, captions and the graphic."""

from __future__ import annotations

import pandas as pd

# Columns of top10_mvp_latest.csv that describe performance (games is context, not a standout).
STAT_COLS = ["passer_rating", "total_tds", "td_int", "total_yards", "pass_ypg", "rush_yards", "win_pct"]

STAT_LABELS = {
    "passer_rating": "Rating",  "total_tds": "Total TD", "td_int": "TD:INT",
    "total_yards":   "Total Yds", "pass_ypg": "Pass YPG", "rush_yards": "Rush Yds",
    "win_pct":       "Team W%",  "games": "Games",
}

# The reel spells some labels out: they're read aloud and sit on a big card ("TD to INT ratio", not "TD:INT").
# total_ypg / rush_ypg are per-game values derived by per_game() for the reel's then-vs-now comparison.
REEL_LABELS = {**STAT_LABELS, "passer_rating": "Passer rating", "td_int": "TD to INT ratio",
               "total_ypg": "Yds / game", "rush_ypg": "Rush yds / game"}

# The reel's then-vs-now compares rates, not season totals (totals only ever go up): by position.
RATE_COLS = {"QB": ["passer_rating", "td_int", "pass_ypg"], "RB": ["total_ypg", "rush_ypg"], "WR": ["total_ypg"]}


def per_game(row: pd.Series) -> pd.Series:
    """row plus total_ypg and rush_ypg (season totals / games), for comparing two weekly boards."""
    row = row.copy()
    try:
        g = float(row["games"])
        row["total_ypg"] = round(float(row["total_yards"]) / g, 1) if g else None
        row["rush_ypg"] = round(float(row["rush_yards"]) / g, 1) if g else None
    except (KeyError, TypeError, ValueError):
        pass
    return row


# Stats a player's stat cells may show, by position. The cells pick the stats where a player ranks highest within
# the top 10, which is mostly QBs, so without this a WR's 29 rush yards could outrank his 1,250 total yards.
# For a WR, total yards and total TD are effectively his receiving numbers.
POSITION_STAT_COLS = {
    "QB": STAT_COLS,
    "RB": ["rush_yards", "total_yards", "total_tds", "win_pct"],
    "WR": ["total_yards", "total_tds", "win_pct"],
}


def stat_cols_for(player: pd.Series) -> list:
    return POSITION_STAT_COLS.get(str(player.get("position", "")).upper(), STAT_COLS)


# nflreadpy team code → ESPN code, where they differ (stinger-assets logos are ESPN-named)
ESPN_TEAM_CODES = {"LA": "LAR", "WAS": "WSH"}


def format_stat(col: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if col == "win_pct":
        return f"{v:.3f}".lstrip("0") if v < 1 else "1.000"
    if col in ("total_tds", "total_yards", "rush_yards", "games"):
        return f"{int(round(v)):,}"
    if col == "td_int":
        return f"{v:.1f}"
    return f"{v:.1f}"


def stat_line(player: pd.Series, cols=STAT_COLS) -> str:
    """'Rating: 118.1, Total TD: 15, ...' — only stats present for this player."""
    return ", ".join(f"{STAT_LABELS.get(c, c)}: {format_stat(c, player[c])}"
                     for c in cols if c in player and pd.notna(player[c]))
