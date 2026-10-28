"""
Generate a ready-to-paste Instagram/Facebook caption with hashtags for the
weekly NFL MVP carousel: slide 1 the top-10 board, slide 2 "On the Bubble" (ranks 11–15).

Port of mlb-awards-predictor/src/generate_captions.py (one board instead of
four award/league graphics).

Output:
  predictions/{year}/graphics/caption_mvp.txt

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest 30-deep board, else the top 10, + chips):
    python -m src.publish.generate_captions
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import BUBBLE_RANKS, CURRENT_PRED_DIR
from src.publish.generate_chips import CLAUDE_MODEL
from src.publish.stats import stat_line

TEAM_HASHTAGS = {
    "ARI": "#AZCardinals",  "ATL": "#RiseUp",         "BAL": "#RavensFlock",  "BUF": "#BillsMafia",
    "CAR": "#KeepPounding", "CHI": "#DaBears",        "CIN": "#RuleTheJungle", "CLE": "#DawgPound",
    "DAL": "#DallasCowboys", "DEN": "#BroncosCountry", "DET": "#OnePride",    "GB":  "#GoPackGo",
    "HOU": "#Texans",       "IND": "#ForTheShoe",     "JAX": "#DUUUVAL",      "KC":  "#ChiefsKingdom",
    "LA":  "#RamsHouse",    "LAC": "#BoltUp",         "LV":  "#RaiderNation", "MIA": "#FinsUp",
    "MIN": "#Skol",         "NE":  "#Patriots",       "NO":  "#Saints",       "NYG": "#NYGiants",
    "NYJ": "#TakeFlight",   "PHI": "#FlyEaglesFly",   "PIT": "#HereWeGo",     "SEA": "#GoHawks",
    "SF":  "#FTTB",         "TB":  "#GoBucs",         "TEN": "#Titans",       "WAS": "#HTTC",
}

SYSTEM_PROMPT = """You are a sports social media manager for STINGER, an NFL awards prediction tracker.

Write the weekly update post for our NFL MVP carousel: slide 1 is the top-10 board, slide 2 is "On the Bubble" (ranks 11-15, the players knocking on the door). Produce:
1. A punchy caption body — 3 sentences max, under 320 characters total. The first 1-2 sentences cover the top of the board (slide 1). The last sentence names the top bubble player and why they're close to the top 10 (slide 2). Data-driven and slightly hype. Reference actual stats from the input. You may use the history note for context, but don't force it. No filler ("Let's dive in!", "Check this out!"). 1-2 emoji max. Do NOT include hashtags in the body.
2. 15 hashtags for Instagram. Choose ONLY from the approved base tags below plus the player-name and team tags given in the user message. Do NOT invent tags that aren't in the approved list or given in the message.
3. 5 hashtags for Facebook — the 5 most relevant from your Instagram list (favour generic NFL + award tags over player/team specifics).

Approved base hashtags (pick the most relevant, always include #StingerNFL):
#NFL #NFLTwitter #FootballTwitter #NFLFootball #NFL{year} #NFLSeason
#NFLMVP #MVPRace #MVPWatch #MVPAward #NFLStats #NFLWeek{week}
#StingerNFL #StingerSports

Output ONLY valid JSON in exactly this shape, no markdown fences, no preamble:
{{"caption": "...", "hashtags_instagram": ["#tag1", ...], "hashtags_facebook": ["#tag1", ...]}}"""


# Context for the bubble sentence (from the 2010–2025 week-by-week replay).
BUBBLE_HISTORY = ("2012 Adrian Peterson was #15 at Week 8 and won MVP; "
                  "2025 Trevor Lawrence was #25 at Week 8 and finished 5th.")


def _build_prompt(board: pd.DataFrame, chip_df: pd.DataFrame, week: int) -> str:
    whys = dict(zip(chip_df["player_key"], chip_df["Why"])) if not chip_df.empty else {}
    board = board.sort_values("rank")

    def line(p: pd.Series) -> str:
        out = f"#{int(p['rank'])} {p['player_name']} ({p['team']} {p['position']}): {stat_line(p)}"
        if isinstance(whys.get(p["player_key"]), str) and whys[p["player_key"]]:
            out += f"  | {whys[p['player_key']]}"
        return out

    lines = [f"NFL MVP board — through Week {week}", "", "Slide 1 — top 3 contenders:"]
    lines += [line(p) for _, p in board.head(3).iterrows()]
    bubble = board[board["rank"].between(*BUBBLE_RANKS)]
    if not bubble.empty:
        lines += ["", f"Slide 2 — On the Bubble (ranks {BUBBLE_RANKS[0]}-{BUBBLE_RANKS[1]}):"]
        lines += [line(p) for _, p in bubble.iterrows()]
        lines += ["", f"History note: {BUBBLE_HISTORY}"]
    top3 = board.head(3)
    name_tags = " ".join("#" + re.sub(r"[^A-Za-z0-9]", "", n) for n in top3["player_name"])
    team_tags = " ".join(TEAM_HASHTAGS[t] for t in top3["team"] if t in TEAM_HASHTAGS)
    lines += ["", f"Player tags: {name_tags}", f"Team tags: {team_tags}"]
    return "\n".join(lines)


def generate_caption(client, board: pd.DataFrame, chip_df: pd.DataFrame,
                     season: int, week: int, retries: int = 2) -> Optional[dict]:
    prompt = _build_prompt(board, chip_df, week)
    system = SYSTEM_PROMPT.format(year=season, week=week)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            block = next((b for b in resp.content if hasattr(b, "text")), None)
            if block is None:
                raise ValueError("No text block in response")
            text = re.sub(r"^```json|```$", "", block.text.strip(), flags=re.MULTILINE).strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries:
                print(f"  ! Caption generation failed: {e}")
                return None
            time.sleep(1.5)


def generate_caption_file(board: pd.DataFrame, chip_df: pd.DataFrame, gfx_dir: Path,
                          season: int, week: int) -> Optional[Path]:
    """board: the 30-deep board (a top-10-only board still works; slide 2 is then omitted)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — caption generation skipped")
        return None

    from anthropic import Anthropic
    print("  Generating caption: NFL MVP")
    data = generate_caption(Anthropic(), board, chip_df, season, week)
    if data is None:
        return None

    gfx_dir.mkdir(parents=True, exist_ok=True)
    path = gfx_dir / "caption_mvp.txt"
    path.write_text(
        f"=== NFL MVP TRACKER — WEEK {week}, {season} ===\n\n"
        f"CAPTION:\n{data['caption']}\n\n"
        f"INSTAGRAM HASHTAGS:\n{' '.join(data['hashtags_instagram'])}\n\n"
        f"FACEBOOK HASHTAGS:\n{' '.join(data['hashtags_facebook'])}\n",
        encoding="utf-8",
    )
    print(f"Caption -> {path}")
    return path


if __name__ == "__main__":
    deep = CURRENT_PRED_DIR / "board_mvp_latest.csv"
    top = pd.read_csv(deep if deep.exists() else CURRENT_PRED_DIR / "top10_mvp_latest.csv")
    chip_path = CURRENT_PRED_DIR / "chip_data" / "chip_data_latest.csv"
    chips = pd.read_csv(chip_path) if chip_path.exists() else pd.DataFrame(columns=["player_key", "Why"])
    generate_caption_file(top, chips, CURRENT_PRED_DIR / "graphics",
                          int(top["season"].iloc[0]), int(top["through_week"].iloc[0]))
