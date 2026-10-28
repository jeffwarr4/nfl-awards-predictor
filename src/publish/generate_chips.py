"""
Generate the "Chip" (3 pipe-separated tags) and "Why" (one-line rationale) for
the top-3 players on the NFL MVP board, plus a short Why (no chip) for the
"On the Bubble" ranks (config.BUBBLE_RANKS), using the Claude API.

Port of mlb-awards-predictor/src/generate_chips.py.

Output:
  predictions/{year}/chip_data/chip_data_latest.csv   (player_key, Chip, Why; Chip blank for bubble rows)
  predictions/{year}/chip_data/chip_data_{ts}.csv     (+ rank, for reference)

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest 30-deep board, else the top 10):
    python -m src.publish.generate_chips
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import BUBBLE_RANKS, CURRENT_PRED_DIR, TOP_N
from src.publish.stats import stat_line

# Haiku: fast, cheap, no thinking blocks, reliable at strict JSON for 3-word tags.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

CHIP_SYSTEM_PROMPT = """You generate two short labels for NFL awards-prediction graphics: a "Chip" and a "Why".

Rules:
- "chip": exactly 3 short tags (1-3 words each), separated by " | ". Each tag names a distinct statistical strength (e.g. "Efficient passer", "Ball security", "Dual threat", "Workhorse back", "Winning team").
- "why": one sentence, under 100 characters, giving the stat-driven reason for the player's rank. No fluff, no adjectives that aren't backed by a stat in the input.
- Base every word ONLY on the stats given in the user message. Never invent, infer, or assume a stat that wasn't provided.
- Tags describe skill/performance, not identity — don't put the player's name, team, or rank number in the chip itself.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{"chip": "Tag1 | Tag2 | Tag3", "why": "One sentence explanation."}"""


BUBBLE_WHY_MAX = 75   # two wrapped lines on the narrowest bubble label plate (330 px)

BUBBLE_SYSTEM_PROMPT = f"""You write a one-line "Why" for players ranked just outside the top 10 of an NFL MVP prediction board ("On the Bubble").

Rules:
- At most {BUBBLE_WHY_MAX} characters, including spaces. Shorter is better. No ending period.
- Stats only: use ONLY the numbers given in the user message. Never invent, infer, or assume a stat.
- When the stats show it, say what is keeping the player out of the top 10 (e.g. team record, TD total). You may use the comparison player (the lowest-ranked top-10 player at the same position), but only when it makes the point clearer; if the player's own numbers tell the story better (e.g. a strong record or a standout stat), use those instead.
- Never use this player's own name, first or last; the graphic already shows it. Never mention a rank number (no "#10"). Refer to the comparison player by last name only.
- No hype adjectives.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{{"why": "Short line"}}"""


def _record(player: pd.Series) -> str:
    """W-L from win_pct × team_games (ties count as non-wins, as in team_records)."""
    wins = int(round(float(player["win_pct"]) * int(player["team_games"])))
    return f"{wins}-{int(player['team_games']) - wins}"


def build_user_message(player: pd.Series) -> str:
    return (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
            f"Rank: #{int(player['rank'])} in our model's NFL MVP prediction\n"
            f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
            f"Stats: {stat_line(player)}")


def _parse_json(text: str) -> dict:
    """First JSON object in the reply; ignores code fences and any trailing commentary."""
    text = re.sub(r"^```json|```$", "", text.strip()).strip()
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def generate_chip_for_player(client, player: pd.Series, retries: int = 2) -> dict:
    message = build_user_message(player)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=CHIP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
            block = next((b for b in resp.content if hasattr(b, "text")), None)
            if block is None:
                raise ValueError("No text block in response")
            data = _parse_json(block.text)
            return {"chip": data["chip"], "why": data["why"]}
        except Exception as e:
            if attempt == retries:
                print(f"  ! Chip generation failed for {player.get('player_name', '?')}: {e}")
                return {"chip": "", "why": ""}
            time.sleep(1.5)


def build_bubble_message(player: pd.Series, comp: pd.Series | None) -> str:
    msg = (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
           f"Rank: #{int(player['rank'])}, just outside the top 10\n"
           f"Team record: {_record(player)}\n"
           f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
           f"Stats: {stat_line(player)}")
    if comp is not None:
        msg += (f"\n\nFor comparison, the lowest-ranked {comp['position']} inside the top 10: "
                f"{comp['player_name']} (team record {_record(comp)}): {stat_line(comp)}")
    return msg


def _ask_why(client, message: str) -> str:
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=100,
        system=BUBBLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    block = next((b for b in resp.content if hasattr(b, "text")), None)
    if block is None:
        raise ValueError("No text block in response")
    return str(_parse_json(block.text)["why"]).strip()


def generate_bubble_why(client, player: pd.Series, comp: pd.Series | None, retries: int = 2) -> str:
    """≤ BUBBLE_WHY_MAX chars. Errors retry like chips; an over-length answer gets one
    rewrite request, then is hard-truncated (a failed rewrite keeps the first answer)."""
    message = build_bubble_message(player, comp)
    for attempt in range(retries + 1):
        try:
            why = _ask_why(client, message)
            break
        except Exception as e:
            if attempt == retries:
                print(f"  ! Bubble Why failed for {player.get('player_name', '?')}: {e}")
                return ""
            time.sleep(1.5)
    if len(why) > BUBBLE_WHY_MAX:
        try:
            why = _ask_why(client, f"{message}\n\nYour last answer was {len(why)} characters: "
                                   f'"{why}". Rewrite it in {BUBBLE_WHY_MAX} characters or fewer.')
        except Exception as e:
            print(f"  ! Bubble Why rewrite failed for {player.get('player_name', '?')}: {e}")
    if len(why) > BUBBLE_WHY_MAX:
        why = why[:BUBBLE_WHY_MAX - 1].rstrip(" ,;·-") + "…"
    return why


def comparison_player(board: pd.DataFrame, player: pd.Series) -> pd.Series | None:
    """Lowest-ranked top-10 player at the same position (a QB is measured against a QB);
    falls back to #TOP_N when no top-10 player shares the position."""
    top = board[board["rank"] <= TOP_N].sort_values("rank")
    same = top[top["position"] == player["position"]]
    pick = same if not same.empty else top
    return pick.iloc[-1] if not pick.empty else None


def generate_chips(board: pd.DataFrame, outdir: Path, top_n: int = 3) -> pd.DataFrame:
    """Chips + Why for the top_n players, Why only for ranks BUBBLE_RANKS.
    Returns (player_key, Chip, Why); empty if no API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — chip generation skipped")
        return pd.DataFrame(columns=["player_key", "Chip", "Why"])

    from anthropic import Anthropic
    client = Anthropic()

    board = board.sort_values("rank")
    rows = []
    for _, player in board.head(top_n).iterrows():
        print(f"  Generating chip: {player['player_name']}")
        result = generate_chip_for_player(client, player)
        rows.append({"player_key": player["player_key"], "Chip": result["chip"],
                     "Why": result["why"], "rank": int(player["rank"])})

    for _, player in board[board["rank"].between(*BUBBLE_RANKS)].iterrows():
        print(f"  Generating bubble Why: {player['player_name']}")
        rows.append({"player_key": player["player_key"], "Chip": "",
                     "Why": generate_bubble_why(client, player, comparison_player(board, player)), "rank": int(player["rank"])})
    chip_df = pd.DataFrame(rows)

    chip_dir = Path(outdir) / "chip_data"
    chip_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chip_df[["player_key", "Chip", "Why"]].to_csv(chip_dir / "chip_data_latest.csv", index=False)
    chip_df.to_csv(chip_dir / f"chip_data_{ts}.csv", index=False)
    print(f"Chip data -> {chip_dir / 'chip_data_latest.csv'}")
    return chip_df[["player_key", "Chip", "Why"]]


if __name__ == "__main__":
    deep = CURRENT_PRED_DIR / "board_mvp_latest.csv"
    generate_chips(pd.read_csv(deep if deep.exists() else CURRENT_PRED_DIR / "top10_mvp_latest.csv"),
                   CURRENT_PRED_DIR)
