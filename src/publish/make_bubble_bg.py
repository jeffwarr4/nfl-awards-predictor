"""Build the 'On the Bubble' background for ranks 11-15.

Outputs (argv[2] = out dir):
  MVP Bubble NFL.png          background: header + 'ON THE BUBBLE' pill + 5 empty bubbles
  MVP Bubble NFL overlay.png  transparent layer: gold rings + gloss, composited OVER headshots
  bubble_slots.json           centers, diameters, label boxes, rank-badge anchors
  bubble_preview_wk7.png      filled with the real Week 7 ranks 11-15 (validation only)
"""
import json, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO, OUT, ASSETS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1080, 1350
GOLD, BLACK, WHITE, GREY = (255, 210, 0), (0, 0, 0), (255, 255, 255), (170, 170, 170)
UP, DOWN = (60, 200, 90), (230, 70, 60)
FONTS = REPO / "assets" / "fonts"
def font(name, size):
    p = {"oswald": FONTS / "Oswald/static/Oswald-Bold.ttf",
         "bc_semi": FONTS / "Barlow_Condensed/BarlowCondensed-SemiBold.ttf",
         "bc_med": FONTS / "Barlow_Condensed/BarlowCondensed-Medium.ttf"}[name]
    return ImageFont.truetype(str(p), size)

# ── Slots: #11 biggest and highest ("closest to breaking in"), shrinking and
#    sinking toward #15. Zig-zag so labels never sit under another bubble.
SLOTS = [
    {"rank": 11, "cx": 300, "cy": 470,  "d": 330},
    {"rank": 12, "cx": 790, "cy": 525,  "d": 290},
    {"rank": 13, "cx": 545, "cy": 918,  "d": 255},
    {"rank": 14, "cx": 215, "cy": 1075, "d": 225},
    {"rank": 15, "cx": 870, "cy": 1085, "d": 200},
]
LABEL_H = 112  # name + meta + up to two Why lines
for s in SLOTS:
    r = s["d"] // 2
    lw = max(s["d"] + 40, 330)
    s["label_box"] = [s["cx"] - lw // 2, s["cy"] + r + 8, s["cx"] + lw // 2, s["cy"] + r + 8 + LABEL_H]
    a = math.radians(135)  # rank badge at ~7:30 on the ring
    s["rank_badge"] = [int(s["cx"] + r * math.cos(a)), int(s["cy"] + r * math.sin(a))]
    a = math.radians(-45)  # team logo badge at ~1:30
    s["logo_badge"] = [int(s["cx"] + r * math.cos(a)), int(s["cy"] + r * math.sin(a))]
FOOTER = {"cx": 540, "cy": 1255}

# sanity: no bubble/label overlaps, everything on canvas
def boxes(s):
    r = s["d"] // 2
    return [(s["cx"] - r, s["cy"] - r, s["cx"] + r, s["cy"] + r), tuple(s["label_box"])]
allb = [(s["rank"], b) for s in SLOTS for b in boxes(s)]
for i, (ra, a) in enumerate(allb):
    assert a[0] >= 0 and a[1] >= 280 and a[2] <= W and a[3] <= H, (ra, a)
    for rb, b in allb[i + 1:]:
        if ra != rb and not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
            raise SystemExit(f"overlap #{ra} {a} vs #{rb} {b}")

# ── Background ────────────────────────────────────────────────────
bg = Image.open(REPO / "MVP Awards NFL.png").convert("RGBA")

def pill(img, text, cy, size=30):
    d = ImageDraw.Draw(img); f = font("oswald", size)
    bb = d.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bw, bh = tw + 44, 42; bx, by = W // 2 - bw // 2, cy - bh // 2
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=10, fill=(*GOLD, 245), outline=BLACK, width=2)
    d.text((bx + (bw - tw) // 2 - bb[0], by + (bh - th) // 2 - bb[1]), text, font=f, fill=BLACK)
pill(bg, "ON THE BUBBLE", 243)
d = ImageDraw.Draw(bg)
sub = "RANKS 11–15 · KNOCKING ON THE DOOR"
f = font("bc_semi", 24); bb = d.textbbox((0, 0), sub, font=f)
d.text((W // 2 - (bb[2] - bb[0]) // 2, 274), sub, font=f, fill=WHITE, stroke_width=2, stroke_fill=BLACK)

# decorative micro-bubbles, kept clear of slots and labels
rng = np.random.default_rng(7)
deco = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(deco)
placed = 0
while placed < 22:
    r = int(rng.integers(8, 26)); x = int(rng.integers(30, W - 30)); y = int(rng.integers(320, H - 30))
    clear = all(math.hypot(x - s["cx"], y - s["cy"]) > s["d"] / 2 + r + 18 for s in SLOTS)
    clear &= all(not (lb[0] - r < x < lb[2] + r and lb[1] - r < y < lb[3] + r) for lb in (s["label_box"] for s in SLOTS))
    clear &= math.hypot(x - FOOTER["cx"], y - FOOTER["cy"]) > 170
    if clear:
        dd.ellipse((x - r, y - r, x + r, y + r), outline=(*GOLD, 90), width=2, fill=(255, 255, 255, 18))
        dd.arc((x - r * .6, y - r * .6, x + r * .2, y + r * .2), 200, 260, fill=(255, 255, 255, 120), width=2)
        placed += 1
bg = Image.alpha_composite(bg, deco)

# empty bubble wells: soft shadow + dark glassy interior (headshots paste here)
for s in SLOTS:
    r = s["d"] // 2; cx, cy = s["cx"], s["cy"]
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((cx - r + 6, cy - r + 14, cx + r + 6, cy + r + 14), fill=(0, 0, 0, 120))
    bg = Image.alpha_composite(bg, sh.filter(ImageFilter.GaussianBlur(14)))
    well = Image.new("RGBA", (W, H), (0, 0, 0, 0)); wd = ImageDraw.Draw(well)
    for i in range(r, 0, -2):  # radial: lighter centre, darker edge
        t = i / r
        wd.ellipse((cx - i, cy - i, cx + i, cy + i), fill=(int(20 + 25 * (1 - t)), int(40 + 30 * (1 - t)), 22, int(150 + 60 * t)))
    bg = Image.alpha_composite(bg, well)
    # faint label plate so name/stats read on the turf
    lb = s["label_box"]; plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle(lb, radius=14, fill=(10, 18, 10, 150))
    bg = Image.alpha_composite(bg, plate)

# ── Overlay (goes on top of headshots) ────────────────────────────
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for s in SLOTS:
    r = s["d"] // 2; cx, cy = s["cx"], s["cy"]
    ring = max(6, s["d"] // 45)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6), outline=(*GOLD, 140), width=ring + 8)
    ov = Image.alpha_composite(ov, glow.filter(ImageFilter.GaussianBlur(6)))
    od = ImageDraw.Draw(ov)
    od.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*GOLD, 255), width=ring)
    od.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), outline=(0, 0, 0, 160), width=2)
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
    gr = int(r * 0.78)
    gd.arc((cx - gr, cy - gr, cx + gr, cy + gr), 195, 255, fill=(255, 255, 255, 170), width=max(5, r // 14))
    hx, hy, hr = cx - int(r * .52), cy - int(r * .52), max(5, r // 16)
    gd.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(255, 255, 255, 200))
    ov = Image.alpha_composite(ov, gl.filter(ImageFilter.GaussianBlur(1.5)))

bg.convert("RGB").save(OUT / "MVP Bubble NFL.png")
ov.save(OUT / "MVP Bubble NFL overlay.png")
json.dump({"canvas": [W, H], "slots": SLOTS, "footer": FOOTER,
           "notes": "Sizes fixed by rank (#11 largest). Paste circular headshot into each slot, then composite 'MVP Bubble NFL overlay.png' on top."},
          open(OUT / "bubble_slots.json", "w"), indent=2)

# ── Preview with real Week 7 ranks 11-15 ──────────────────────────
board = pd.read_csv(sys.argv[4]).head(15).iloc[10:15]
whys = {11: "8 TDs, 1,697 yds on a 5-2 Browns team", 12: "855 scrimmage yds; Steelers 5-2",
        13: "1,932 yds, 13 TDs; DAL's 3-4 record holds him back", 14: "704 rush yds; ATL's 3-4 record drags him",
        15: "Volume back with a winning MIN team"}
canvas = bg.copy()
def circle_headshot(path, d):
    im = Image.open(path).convert("RGBA")
    w, h = im.size; side = int(h * 0.92)                  # face sits upper-centre of ESPN 600x436
    x0 = (w - side) // 2; y0 = 0
    im = im.crop((x0, y0, x0 + side, y0 + side)).resize((d, d), Image.LANCZOS)
    m = Image.new("L", (d, d), 0); ImageDraw.Draw(m).ellipse((0, 0, d - 1, d - 1), fill=255)
    out = Image.new("RGBA", (d, d), (0, 0, 0, 0)); out.paste(im, (0, 0), Image.composite(im.getchannel("A"), m, m))
    return out
for s, (_, p) in zip(SLOTS, board.iterrows()):
    r = s["d"] // 2
    hs = ASSETS / "nfl" / "headshots" / f"{p.player_key}.png"
    if hs.exists():
        canvas.alpha_composite(circle_headshot(hs, s["d"] - 8), (s["cx"] - r + 4, s["cy"] - r + 4))
canvas = Image.alpha_composite(canvas, ov)
d = ImageDraw.Draw(canvas)
for s, (_, p) in zip(SLOTS, board.iterrows()):
    # logo badge
    lp = next((q for q in [ASSETS / "nfl/logos" / f"{p.team.lower()}.png", ASSETS / "nfl/logos" / f"{p.team}.png",
                          ASSETS / "nfl/logos" / f"{ {'LA':'lar','WAS':'wsh'}.get(p.team, p.team).lower()}.png"] if q.exists()), None)
    bx, by = s["logo_badge"]; br = max(30, s["d"] // 7)
    d.ellipse((bx - br, by - br, bx + br, by + br), fill=(15, 25, 15), outline=GOLD, width=3)
    if lp:
        lg = Image.open(lp).convert("RGBA"); lg.thumbnail((int(br * 1.5), int(br * 1.5)))
        canvas.alpha_composite(lg, (bx - lg.width // 2, by - lg.height // 2))
    # rank badge
    rx, ry = s["rank_badge"]; rr = max(26, s["d"] // 9)
    d.ellipse((rx - rr, ry - rr, rx + rr, ry + rr), fill=GOLD, outline=BLACK, width=3)
    rf = font("oswald", int(rr * 0.95)); t = f"#{int(p['rank'])}"; bb = d.textbbox((0, 0), t, font=rf)
    d.text((rx - (bb[2] - bb[0]) // 2 - bb[0], ry - (bb[3] - bb[1]) // 2 - bb[1]), t, font=rf, fill=BLACK)
    # label
    lb = s["label_box"]; cx = (lb[0] + lb[2]) // 2
    nf = font("oswald", 30 if s["d"] > 240 else 26)
    name = str(p.player_name).upper(); bb = d.textbbox((0, 0), name, font=nf)
    d.text((cx - (bb[2] - bb[0]) // 2, lb[1] + 6), name, font=nf, fill=WHITE, stroke_width=1, stroke_fill=BLACK)
    mf = font("bc_semi", 22)
    mv = str(p.movement) if isinstance(p.movement, str) else "NEW"
    meta = f"{p.team} · {p.position}  "; bb1 = d.textbbox((0, 0), meta, font=mf); bb2 = d.textbbox((0, 0), mv, font=mf)
    x = cx - (bb1[2] + bb2[2]) // 2
    d.text((x, lb[1] + 44), meta, font=mf, fill=GREY)
    d.text((x + bb1[2], lb[1] + 44), mv, font=mf, fill=UP if mv.startswith("↑") else DOWN if mv.startswith("↓") else GOLD)
    # Why: greedy word-wrap to two lines; ellipsis only if it still overflows
    wf = font("bc_med", 20); maxw = lb[2] - lb[0] - 16
    tw = lambda t: d.textbbox((0, 0), t, font=wf)[2]
    lines, cur = [], ""
    for word in whys[int(p['rank'])].split():
        if cur and tw(cur + " " + word) > maxw:
            lines.append(cur); cur = word
        else:
            cur = f"{cur} {word}".strip()
    lines.append(cur)
    if len(lines) > 2:
        last = " ".join(lines[1:])
        while tw(last + "…") > maxw:
            last = last[:-1].rstrip()
        lines = [lines[0], last + "…"]
    for i, wt in enumerate(lines):
        d.text((cx - tw(wt) // 2, lb[1] + 66 + i * 21), wt, font=wf, fill=GOLD)
f1, f2 = font("oswald", 24), font("bc_med", 16)
for t, fnt, col, y in (("STATS THROUGH WEEK 7", f1, GOLD, FOOTER["cy"] - 14), ("Bubble size = closeness to the top 10", f2, GREY, FOOTER["cy"] + 20)):
    bb = d.textbbox((0, 0), t, font=fnt); d.text((FOOTER["cx"] - (bb[2] - bb[0]) // 2, y), t, font=fnt, fill=col, stroke_width=1 if fnt is f1 else 0, stroke_fill=BLACK)
canvas.convert("RGB").save(OUT / "bubble_preview_wk7.png")
print("ok", [ (s["rank"], s["d"]) for s in SLOTS])
