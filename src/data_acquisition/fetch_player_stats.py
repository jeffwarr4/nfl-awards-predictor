"""
Fetch season-level player stats (QB/RB/WR) using nflreadpy and attach team win%.
Saves to data/raw/player_season_stats.csv.

Run:
    python -m src.data_acquisition.fetch_player_stats
"""

from typing import Iterable
import polars as pl
import pandas as pd
import nflreadpy as nfl

YEARS: Iterable[int] = range(2000, 2025)

def load_player_season_stats(years: Iterable[int]) -> pl.DataFrame:
    # summary_level='reg' gives regular-season totals by player+season
    df = nfl.load_player_stats(list(years), summary_level="reg")
    # keep only skill positions of interest
    return df.filter(pl.col("position").is_in(["QB", "RB", "WR"]))

def load_team_win_pct(years: Iterable[int]) -> pl.DataFrame:
    sch = nfl.load_schedules(list(years))
    # compute wins per game from home/away scores
    home = sch.select(
        pl.col("season"),
        team=pl.col("home_team"),
        win=(pl.col("home_score") > pl.col("away_score")).cast(pl.Int8)
    )
    away = sch.select(
        pl.col("season"),
        team=pl.col("away_team"),
        win=(pl.col("away_score") > pl.col("home_score")).cast(pl.Int8)
    )
    team_games = pl.concat([home, away])
    team_records = (
        team_games.group_by(["season", "team"])
        .agg(games=pl.len(), wins=pl.col("win").sum())
        .with_columns((pl.col("wins") / pl.col("games")).alias("win_pct"))
    )
    return team_records

def run(years: Iterable[int] = YEARS, out_path: str = "data/raw/player_season_stats.csv") -> str:
    players = load_player_season_stats(years)
    players = players.rename({"recent_team": "team"})
    teams = load_team_win_pct(years)
    out = (
        players.join(teams.select(["season", "team", "win_pct"]), on=["season", "team"], how="left")
        # add simple per-game rates from totals & games
        .with_columns([
            (pl.col("passing_yards") / pl.col("games")).alias("pass_ypg"),
            (pl.col("rushing_yards") / pl.col("games")).alias("rush_ypg"),
            (pl.col("receiving_yards") / pl.col("games")).alias("recv_ypg"),
        ])
    )
    # save as CSV via pandas for broad compatibility
    out.to_pandas().to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    path = run()
    print(f"Wrote {path}")
