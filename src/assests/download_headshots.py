# src/assets/download_headshots.py   <-- fix the folder name to "assets", not "assests"
import os, re, glob, unicodedata
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

OUTPUTS_DIR = Path("outputs")
SAVE_DIR = Path("assets/headshots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def newest_csv():
    # match both dated and undated files
    patterns = [
        "mvp_watch_canva*.csv",
        "mvp_watch_canva_*_week*.csv",
        "mvp_watch_canva_top*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(OUTPUTS_DIR.glob(p))
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No Canva CSVs found in {OUTPUTS_DIR.resolve()}")
    return files[0]

def normalize_cols(cols):
    return [re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in cols]

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s\-\.]", "", s, flags=re.UNICODE)
    s = s.strip().replace(" ", "_")
    return re.sub(r"_+", "_", s)

def first_nonempty(series: pd.Series) -> pd.Series:
    """For duplicate URL columns, pick the first non-empty value across columns."""
    for s in series:
        s2 = s.astype(str).str.strip()
        if s2.where(~s2.eq("")).notna().any():
            return s2
    return series[0].astype(str).str.strip()

def iter_headshot_urls(df: pd.DataFrame):
    norm = df.copy()
    norm.columns = normalize_cols(norm.columns)

    # Accept any of these name columns
    name_col = next((c for c in ["player_display_name","player_name","display_name","name"] if c in norm.columns), None)

    # Collect any headshot columns (headshot_url, headshot_url.1, etc.)
    headshot_cols = [c for c in norm.columns if c.startswith("headshot_url")]

    # Case A: Long/top-10 format with headshot_url
    if headshot_cols and name_col:
        # collapse duplicates (e.g., headshot_url + headshot_url_1 from pandas)
        urls = first_nonempty([norm[c] for c in headshot_cols])
        for name, url in zip(norm[name_col].astype(str), urls):
            u = str(url).strip()
            if u.lower().startswith(("http://","https://")):
                yield name, u
        return

    # Case B: Wide/top-3 format: rank1_headshot, rank2_headshot...
    rank_cols = [c for c in norm.columns if c.endswith("_headshot")]
    if rank_cols:
        row = norm.iloc[0]
        for rc in sorted(rank_cols):  # rank1_headshot, rank2_headshot, rank3_headshot
            idx = rc.split("_")[0]    # rank1
            nc = f"{idx}_name"
            url = str(row.get(rc, "")).strip()
            name = str(row.get(nc, idx)).strip()
            if url.lower().startswith(("http://","https://")):
                yield name, url
        return

    raise KeyError(
        "Could not find headshot columns.\n"
        f"Available columns: {list(norm.columns)}\n"
        "Expected either 'headshot_url' (+ a name column) or 'rank#_headshot' format."
    )

def main():
    csv_path = newest_csv()
    print(f"Using CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Columns found:", list(df.columns))

    session = requests.Session()
    session.headers.update({"User-Agent": "MVP-Headshot-Downloader/1.0"})

    failures = 0
    pairs = list(iter_headshot_urls(df))
    for name, url in tqdm(pairs, desc="Downloading headshots"):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            ext = os.path.splitext(url.split("?")[0])[1].lower()
            if ext not in (".png",".jpg",".jpeg",".webp"):
                ext = ".png"
            filename = SAVE_DIR / f"{slugify(name)}{ext}"
            with open(filename, "wb") as f:
                f.write(r.content)
        except Exception as e:
            failures += 1
            print(f"❌ {name}: {e}")

    print(f"\n✅ Done. Saved to {SAVE_DIR.resolve()}. Downloaded: {len(pairs) - failures} | Failed: {failures}")

if __name__ == "__main__":
    main()
