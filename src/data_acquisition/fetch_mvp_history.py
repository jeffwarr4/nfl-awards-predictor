"""
Fetch AP NFL MVP winners by season from Pro-Football-Reference and save to data/raw/mvp_winners.csv.

Run:
    python -m src.data_acquisition.fetch_mvp_history
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

PFR_URL = "https://www.pro-football-reference.com/awards/ap-nfl-mvp-award.htm"
POSSIBLE_PLAYER_COLS = ["AP MVP", "Winner", "Player"]  # handle layout changes


def fetch_winners() -> pd.DataFrame:
    tables = pd.read_html(PFR_URL, flavor="lxml")
    if not tables:
        raise RuntimeError("No tables found on PFR page.")

    df: Optional[pd.DataFrame] = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if "Year" in cols and any(col in cols for col in POSSIBLE_PLAYER_COLS):
            df = t.copy()
            break

    if df is None:
        raise ValueError(
            "Could not find a winners table with expected columns. "
            f"Tables present: {[' | '.join(map(str, t.columns)) for t in tables]}"
        )

    # Determine which column holds player names
    player_col: Optional[str] = None
    for candidate in POSSIBLE_PLAYER_COLS:
        if candidate in df.columns:
            player_col = candidate
            break
    if player_col is None:
        raise ValueError(f"Unable to locate player column in PFR table; columns={df.columns.tolist()}")

    # Keep a minimal, stable set of columns (only those present)
        keep_cols = [c for c in ["Year", player_col, "Pos", "Tm"] if c in df.columns]
        df = df[keep_cols].rename(
            columns={
                "Year": "season",
                player_col: "player_name",
                "Pos": "position",
                "Tm": "team",
            }
        )

    # Clean fields
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["player_name"] = (
        df["player_name"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.dropna(subset=["season"]).astype({"season": "int64"})

    df["mvp_winner"] = 1  # winners-only table
    return df


def run(out_path: Optional[Union[str, Path]] = None) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    # Normalize to a Path early so .parent is always valid
    path_obj = Path(out_path) if out_path is not None else (repo_root / "data" / "raw" / "mvp_winners.csv")
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    winners = fetch_winners()
    winners.to_csv(path_obj, index=False)
    print(f"Wrote {path_obj} | Rows={len(winners)} | Seasons={winners['season'].nunique()}")
    return str(path_obj)


if __name__ == "__main__":
    run()
