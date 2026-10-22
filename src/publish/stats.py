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
