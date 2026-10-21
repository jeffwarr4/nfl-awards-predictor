"""
Render the weekly NFL MVP board graphic (1080×1350) by compositing headshots,
team logos, chips and stat blocks onto the background in the repo root.

Port of mlb-awards-predictor/src/render_graphics.py: same fonts, card anatomy and
percentile-based select_top_stats, but one board (top-3 cards + compact rows for
ranks 4–10) instead of one graphic per award/league.

Assets come from stinger-assets (STINGER_ASSETS_DIR):
  nfl/headshots/{player_key}.png    nfl/logos/{espn_code}.png   (all lowercase)

Outputs: predictions/{year}/graphics/mvp_board_latest.png
         predictions/{year}/graphics/mvp_board_{year}_wk{NN}.png

Run standalone (from the latest board + chips):
    python -m src.publish.render_graphics
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import BOARD_BACKGROUND, CURRENT_PRED_DIR, FONTS_DIR, STINGER_ASSETS_DIR
from src.publish.stats import ESPN_TEAM_CODES, STAT_COLS, STAT_LABELS, format_stat

HEADSHOTS_DIR = STINGER_ASSETS_DIR / "nfl" / "headshots"
LOGOS_DIR     = STINGER_ASSETS_DIR / "nfl" / "logos"

# ── Fonts (OFL, committed in assets/fonts; Windows originals as fallback) ──
_F_WIN = Path("C:/Windows/Fonts")


def _find_font(*candidates: Path) -> str:
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[-1])  # not found → _font() falls back to PIL default


FONT_PATH = {
    "agency_bold": _find_font(FONTS_DIR / "Oswald" / "static" / "Oswald-Bold.ttf", _F_WIN / "AGENCYB.TTF"),
    "fg_med_cond": _find_font(FONTS_DIR / "Barlow_Condensed" / "BarlowCondensed-Medium.ttf", _F_WIN / "FRAMDCN.TTF"),
    "fg_demi":     _find_font(FONTS_DIR / "Barlow_Semi_Condensed" / "BarlowSemiCondensed-SemiBold.ttf", _F_WIN / "FRADM.TTF"),
    "fg_demi_cond": _find_font(FONTS_DIR / "Barlow_Condensed" / "BarlowCondensed-SemiBold.ttf", _F_WIN / "FRADMCN.TTF"),
}

# ── Brand colors ──────────────────────────────────────────────────
GOLD  = (255, 210, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
UP    = (60, 200, 90)
DOWN  = (230, 70, 60)
GREY  = (170, 170, 170)

# ── Canvas ────────────────────────────────────────────────────────
IMG_W, IMG_H = 1080, 1350
CONTENT_TOP  = 220          # below the STINGER header band of the background

LOWER_IS_BETTER: set = set()
_HS_ASPECT = 600 / 436      # ESPN headshot ratio (NFL files are 3400×2450 ≈ same)


# ─────────────────────────────────────────────────────────────────
# STAT SELECTOR — same as MLB, reusable across sports
# ─────────────────────────────────────────────────────────────────
def select_top_stats(player_row: pd.Series, pool_df: pd.DataFrame, stat_cols: List[str],
                     n: int = 3, lower_is_better: Optional[set] = None) -> List[str]:
    """Top-n stat columns for this player, ranked by percentile within pool_df.
    Stats missing for the player (e.g. passer rating for an RB) are skipped."""
    lower_is_better = lower_is_better or set()
    pcts: Dict[str, float] = {}
    for col in stat_cols:
        if col not in pool_df.columns:
            continue
        pval = pd.to_numeric(player_row.get(col), errors="coerce")
        if pd.isna(pval):
            continue
        vals = pd.to_numeric(pool_df[col], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        pct = float((vals < pval).sum()) / len(vals) * 100.0
        pcts[col] = (100.0 - pct) if col in lower_is_better else pct
    return [c for c, _ in sorted(pcts.items(), key=lambda x: x[1], reverse=True)[:n]]


# ─────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────
def _font(key: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH[key], size)
    except Exception:
        return ImageFont.load_default()


def _paste(canvas: Image.Image, layer: Image.Image, xy: Tuple[int, int]):
    canvas.paste(layer, xy, layer) if layer.mode == "RGBA" else canvas.paste(layer, xy)


def load_headshot(player_key: str, target_w: int) -> Optional[Image.Image]:
    path = HEADSHOTS_DIR / f"{player_key}.png"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
        target_h = max(1, int(round(target_w * img.height / img.width)))
        return img.resize((target_w, target_h), Image.LANCZOS)
    except Exception:
        return None


def load_headshot_placeholder(target_w: int) -> Image.Image:
    target_h = max(1, int(round(target_w / _HS_ASPECT)))
    ph = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    ImageDraw.Draw(ph).rectangle((0, 0, target_w - 1, target_h - 1), fill=(70, 70, 70, 180))
    return ph


def load_logo(team_abbr: str, size: int) -> Image.Image:
    code = ESPN_TEAM_CODES.get(team_abbr.upper(), team_abbr).lower()
    p = LOGOS_DIR / f"{code}.png"
    if p.exists():
        img = Image.open(p).convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)
        return img
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def overlay_rect(canvas: Image.Image, box: Tuple, radius: int, fill_rgba: Tuple,
                 outline_rgba: Optional[Tuple] = None, outline_w: int = 2):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius=radius, fill=fill_rgba,
                                            outline=outline_rgba, width=outline_w)
    canvas.alpha_composite(layer)


def draw_text_centered(draw, cx, y, text, font, fill, stroke_fill=(0, 0, 0), stroke_w=2):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((cx - tw // 2, y), text, font=font, fill=stroke_fill,
              stroke_width=stroke_w, stroke_fill=stroke_fill)
    draw.text((cx - tw // 2, y), text, font=font, fill=fill)
    return th


def text_w(draw, text, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def fit_text(draw, text, font, max_w) -> str:
    if text_w(draw, text, font) <= max_w:
        return text
    while len(text) > 4:
        text = text[:-1]
        if text_w(draw, text + "…", font) <= max_w:
            return text + "…"
    return text


def draw_rank_badge(canvas: Image.Image, cx: int, cy: int, r: int, rank: int, size: int):
    bl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(bl).ellipse((cx - r, cy - r, cx + r, cy + r),
                               fill=(*GOLD, 255), outline=(*BLACK, 255), width=2)
    canvas.alpha_composite(bl)
    draw = ImageDraw.Draw(canvas)
    font = _font("agency_bold", size)
    bb = draw.textbbox((0, 0), f"#{rank}", font=font)
    draw.text((cx - (bb[2] - bb[0]) // 2 - bb[0], cy - (bb[3] - bb[1]) // 2 - bb[1]),
              f"#{rank}", font=font, fill=BLACK)


def draw_movement(draw, x: int, y: int, movement: str, size: int) -> int:
    """Draw ▲3 / ▼2 / – / NEW at (x, y) top-left. Returns width drawn.
    Triangles are drawn as polygons — the condensed fonts have no arrow glyphs."""
    font = _font("fg_demi", size)
    m = str(movement)
    if m in ("NEW", "", "nan"):
        draw.text((x, y), "NEW", font=font, fill=GOLD)
        return text_w(draw, "NEW", font)
    if m[0] in "↑↓":
        up = m[0] == "↑"
        t = int(size * 0.55)
        top = y + int(size * 0.3)
        pts = ([(x, top + t), (x + t, top + t), (x + t // 2, top)] if up
               else [(x, top), (x + t, top), (x + t // 2, top + t)])
        draw.polygon(pts, fill=UP if up else DOWN)
        draw.text((x + t + 4, y), m[1:], font=font, fill=UP if up else DOWN)
        return t + 4 + text_w(draw, m[1:], font)
    draw.text((x, y), "–", font=font, fill=GREY)
    return text_w(draw, "–", font)


def pct_text(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


# ─────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────
#   y=0–220    header band (background)
#   y=222      week badge
#   y=262–850  top-3 cards (360 px columns)
#   y=884–1332 compact rows: 2 columns × 4 rows (ranks 4–7 left, 8–10 right + footer)

CARD_TOP   = 262
CARD_COL_W = IMG_W // 3
CARD_CX    = [180, 540, 900]
_HS_W      = 300
_LOGO_SZ   = 190
_BADGE_R   = 24
_NAME_H    = 46
_META_H    = 26
_CHIP_H    = 30
_CELL_H    = 56
_CELL_COLS = 2

_ROW_TOP   = 884
_ROW_H     = 112
_ROW_PAD   = 16
_ROW_GAP   = 16
_ROW_W     = (IMG_W - 2 * _ROW_PAD - _ROW_GAP) // 2
_ROW_LEFTS = [_ROW_PAD, _ROW_PAD + _ROW_W + _ROW_GAP]
_ROW_HS_W  = 128


def draw_card(canvas: Image.Image, player: pd.Series, pool_df: pd.DataFrame,
              chip_row: Optional[dict], rank: int):
    cx = CARD_CX[rank - 1]
    left, right = cx - CARD_COL_W // 2, cx + CARD_COL_W // 2

    # 1. Logo first (behind) so it crowns the head, then the headshot
    headshot = load_headshot(str(player.get("player_key", "")), _HS_W) or load_headshot_placeholder(_HS_W)
    hs_y = CARD_TOP + 92
    hs_bot = hs_y + headshot.height
    logo = load_logo(str(player.get("team", "")), _LOGO_SZ)
    _paste(canvas, logo, (cx - logo.width // 2, CARD_TOP + 4))
    _paste(canvas, headshot, (cx - headshot.width // 2, hs_y))

    # 2. Gold rank badge — bottom-left of image zone
    draw_rank_badge(canvas, left + _BADGE_R + 10, hs_bot - _BADGE_R, _BADGE_R, rank, 26)

    # 3. Name banner
    name_y = hs_bot + 5
    overlay_rect(canvas, (left, name_y, right, name_y + _NAME_H), radius=0, fill_rgba=(0, 0, 0, 235))
    draw = ImageDraw.Draw(canvas)
    name_font = _font("agency_bold", 27)
    name = fit_text(draw, str(player.get("player_name", "")).upper(), name_font, CARD_COL_W - 20)
    draw_text_centered(draw, cx, name_y + (_NAME_H - 27) // 2 - 2, name, name_font, WHITE, stroke_w=0)

    # 4. Meta line: team · pos · MVP % · movement
    y = name_y + _NAME_H + 4
    meta_font = _font("fg_demi", 20)
    meta = f"{player.get('team', '')} · {player.get('position', '')} · MVP {pct_text(player.get('mvp_prob'))}  "
    mw = text_w(draw, meta, meta_font)
    probe = Image.new("RGBA", (10, 10))
    move_w = draw_movement(ImageDraw.Draw(probe), 0, 0, player.get("movement", ""), 20)
    mx = cx - (mw + move_w) // 2
    draw.text((mx, y), meta, font=meta_font, fill=WHITE, stroke_width=1, stroke_fill=BLACK)
    draw_movement(draw, mx + mw, y, player.get("movement", ""), 20)
    y += _META_H + 4

    # 5. Chip pills + Why (only when chip data is available)
    chip_str = chip_row.get("Chip") if chip_row else None
    if isinstance(chip_str, str) and "|" in chip_str:
        chips = [c.strip() for c in chip_str.split("|")]
        for chip_sz in range(17, 10, -1):
            chip_font = _font("fg_med_cond", chip_sz)
            pill_pad = max(8, chip_sz - 4)
            pill_ws = [text_w(draw, c, chip_font) + pill_pad * 2 for c in chips]
            total = sum(pill_ws) + 6 * (len(chips) - 1)
            if total <= CARD_COL_W - 16:
                break
        px = cx - total // 2
        cl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cl)
        for chip, pw in zip(chips, pill_ws):
            cd.rounded_rectangle((px, y, px + pw, y + _CHIP_H), radius=14, fill=(*GOLD, 235))
            bb = cd.textbbox((0, 0), chip, font=chip_font)
            cd.text((px + pill_pad, y + (_CHIP_H - (bb[3] - bb[1])) // 2 - bb[1]), chip, font=chip_font, fill=BLACK)
            px += pw + 6
        canvas.alpha_composite(cl)
        draw = ImageDraw.Draw(canvas)
        y += _CHIP_H + 6

        why = chip_row.get("Why")
        if isinstance(why, str) and why:
            wf = _font("fg_med_cond", 18)
            lines = textwrap.wrap(why, width=max(1, (CARD_COL_W - 16) // 7))[:2]
            line_h = max(draw.textbbox((0, 0), l, font=wf)[3] - draw.textbbox((0, 0), l, font=wf)[1] for l in lines)
            box_h = len(lines) * (line_h + 4) + 8
            overlay_rect(canvas, (left + 4, y, right - 4, y + box_h), radius=6, fill_rgba=(20, 20, 20, 175))
            draw = ImageDraw.Draw(canvas)
            ty = y + 4
            for line in lines:
                draw_text_centered(draw, cx, ty, line, wf, (220, 220, 220), stroke_w=0)
                ty += line_h + 4
            y += box_h + 4

    # 6. 2×2 stat grid — the player's 4 standout stats within the top 10
    top_stats = select_top_stats(player, pool_df, STAT_COLS, n=4, lower_is_better=LOWER_IS_BETTER)
    cell_w = CARD_COL_W // _CELL_COLS
    lbl_font, val_font = _font("fg_med_cond", 14), _font("fg_demi", 26)
    for i, col in enumerate(top_stats):
        row_i, col_i = divmod(i, _CELL_COLS)
        cx_cell = left + col_i * cell_w + cell_w // 2
        cy_top = y + row_i * _CELL_H
        overlay_rect(canvas, (left + col_i * cell_w + 3, cy_top + 3,
                              left + (col_i + 1) * cell_w - 3, cy_top + _CELL_H - 3),
                     radius=6, fill_rgba=(15, 15, 15, 200))
        draw = ImageDraw.Draw(canvas)
        draw_text_centered(draw, cx_cell, cy_top + 6, STAT_LABELS.get(col, col), lbl_font, GOLD, stroke_w=1)
        draw_text_centered(draw, cx_cell, cy_top + 22, format_stat(col, player.get(col)), val_font, WHITE)


def draw_compact_row(canvas: Image.Image, player: pd.Series, pool_df: pd.DataFrame,
                     rank: int, row_top: int, row_left: int):
    row_right, row_bot = row_left + _ROW_W, row_top + _ROW_H
    row_cy = (row_top + row_bot) // 2
    overlay_rect(canvas, (row_left + 4, row_top + 4, row_right - 4, row_bot - 4),
                 radius=14, fill_rgba=(10, 10, 10, 192))

    headshot = load_headshot(str(player.get("player_key", "")), _ROW_HS_W) or load_headshot_placeholder(_ROW_HS_W)
    hs_x, hs_y = row_left + 14, row_cy - headshot.height // 2
    _paste(canvas, headshot, (hs_x, hs_y))
    logo = load_logo(str(player.get("team", "")), 36)
    _paste(canvas, logo, (hs_x + headshot.width - logo.width + 6, hs_y - 4))
    draw_rank_badge(canvas, hs_x + 12, hs_y + headshot.height - 12, 16, rank, 18)

    draw = ImageDraw.Draw(canvas)
    text_x = hs_x + headshot.width + 14
    max_w = row_right - text_x - 14

    name_font = _font("agency_bold", 26)
    name = fit_text(draw, str(player.get("player_name", "")).upper(), name_font, max_w - 70)
    draw.text((text_x, row_top + 12), name, font=name_font, fill=WHITE, stroke_width=1, stroke_fill=BLACK)
    pf = _font("fg_demi", 22)
    pct = pct_text(player.get("mvp_prob"))
    draw.text((row_right - 18 - text_w(draw, pct, pf), row_top + 16), pct, font=pf, fill=GOLD)

    meta_font = _font("fg_med_cond", 17)
    meta = f"{player.get('team', '')} · {player.get('position', '')}   "
    draw.text((text_x, row_top + 50), meta, font=meta_font, fill=GREY)
    draw_movement(draw, text_x + text_w(draw, meta, meta_font), row_top + 48, player.get("movement", ""), 17)

    lbl_font, val_font = _font("fg_med_cond", 15), _font("fg_demi", 20)
    sx, sy = text_x, row_top + 76
    top_stats = select_top_stats(player, pool_df, STAT_COLS, n=2, lower_is_better=LOWER_IS_BETTER)
    for i, col in enumerate(top_stats):
        lbl, val = STAT_LABELS.get(col, col) + ":", format_stat(col, player.get(col))
        draw.text((sx, sy + 2), lbl, font=lbl_font, fill=GOLD)
        sx += text_w(draw, lbl, lbl_font) + 4
        draw.text((sx, sy - 1), val, font=val_font, fill=WHITE, stroke_width=1, stroke_fill=BLACK)
        sx += text_w(draw, val, val_font) + 10
        if i < len(top_stats) - 1:
            draw.text((sx, sy - 1), "·", font=val_font, fill=GOLD)
            sx += text_w(draw, "·", val_font) + 10


def draw_week_badge(canvas: Image.Image, season: int, week: int):
    font = _font("agency_bold", 30)
    label = f"{season} · WEEK {week}"
    draw = ImageDraw.Draw(canvas)
    bb = draw.textbbox((0, 0), label, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bw, bh = tw + 40, 40
    bx, by = IMG_W // 2 - bw // 2, CONTENT_TOP + 2
    overlay_rect(canvas, (bx, by, bx + bw, by + bh), radius=10, fill_rgba=(*GOLD, 245),
                 outline_rgba=(*BLACK, 255), outline_w=2)
    draw = ImageDraw.Draw(canvas)
    draw.text((bx + (bw - tw) // 2 - bb[0], by + (bh - th) // 2 - bb[1]), label, font=font, fill=BLACK)


def draw_footer(canvas: Image.Image, row_top: int, row_left: int, week: int):
    draw = ImageDraw.Draw(canvas)
    f1, f2 = _font("agency_bold", 24), _font("fg_med_cond", 16)
    cx = row_left + _ROW_W // 2
    draw_text_centered(draw, cx, row_top + 22, f"STATS THROUGH WEEK {week}", f1, GOLD, stroke_w=1)
    draw_text_centered(draw, cx, row_top + 58, "Model: nflverse stats projected to a 17-game pace",
                       f2, GREY, stroke_w=0)


def placeholder_background() -> Image.Image:
    """Plain dark board with a header band — used until MVP Awards NFL.png is provided."""
    bg = Image.new("RGBA", (IMG_W, IMG_H), (12, 14, 20, 255))
    d = ImageDraw.Draw(bg)
    for y in range(IMG_H):                      # subtle vertical gradient
        c = int(12 + 14 * y / IMG_H)
        d.line([(0, y), (IMG_W, y)], fill=(c, c + 2, c + 10, 255))
    draw_text_centered(d, IMG_W // 2, 40, "STINGER", _font("agency_bold", 40), GOLD, stroke_w=0)
    draw_text_centered(d, IMG_W // 2, 92, "NFL MVP WATCH", _font("agency_bold", 78), WHITE, stroke_w=0)
    d.line([(60, 205), (IMG_W - 60, 205)], fill=(*GOLD, 255), width=3)
    return bg


def load_background() -> Image.Image:
    if BOARD_BACKGROUND.exists():
        bg = Image.open(BOARD_BACKGROUND).convert("RGBA")
        return bg if bg.size == (IMG_W, IMG_H) else bg.resize((IMG_W, IMG_H), Image.LANCZOS)
    print(f"  [WARN] {BOARD_BACKGROUND.name} not found — using placeholder background")
    return placeholder_background()


# ─────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────
def render_board(top10: pd.DataFrame, chip_df: pd.DataFrame, gfx_dir: Path,
                 season: int, week: int) -> Optional[Path]:
    df = top10.sort_values("rank").reset_index(drop=True)
    if df.empty:
        print("  No board data — skipping graphic")
        return None

    canvas = load_background()
    chip_map: Dict[str, dict] = {}
    if chip_df is not None and not chip_df.empty:
        chip_map = chip_df.set_index("player_key").to_dict(orient="index")

    missing = [k for k in df["player_key"] if not (HEADSHOTS_DIR / f"{k}.png").exists()]
    if missing:
        print(f"  [WARN] {len(missing)} headshot(s) not in {HEADSHOTS_DIR}: {', '.join(missing)}")

    draw_week_badge(canvas, season, week)
    for _, p in df[df["rank"] <= 3].iterrows():
        draw_card(canvas, p, df, chip_map.get(p["player_key"]), int(p["rank"]))

    slots = [(_ROW_LEFTS[0], _ROW_TOP + i * _ROW_H) for i in range(4)] + \
            [(_ROW_LEFTS[1], _ROW_TOP + i * _ROW_H) for i in range(4)]
    for (left, top), (_, p) in zip(slots, df[df["rank"].between(4, 10)].iterrows()):
        draw_compact_row(canvas, p, df, int(p["rank"]), top, left)
    draw_footer(canvas, slots[-1][1], slots[-1][0], week)

    final = Image.new("RGB", (IMG_W, IMG_H), BLACK)
    final.paste(canvas, mask=canvas.split()[3])
    gfx_dir.mkdir(parents=True, exist_ok=True)
    out = gfx_dir / f"mvp_board_{season}_wk{week:02d}.png"
    final.save(out, "PNG")
    final.save(gfx_dir / "mvp_board_latest.png", "PNG")
    print(f"Graphic -> {out}")
    return out


if __name__ == "__main__":
    top = pd.read_csv(CURRENT_PRED_DIR / "top10_mvp_latest.csv")
    chip_path = CURRENT_PRED_DIR / "chip_data" / "chip_data_latest.csv"
    chips = pd.read_csv(chip_path) if chip_path.exists() else pd.DataFrame()
    render_board(top, chips, CURRENT_PRED_DIR / "graphics",
                 int(top["season"].iloc[0]), int(top["through_week"].iloc[0]))
