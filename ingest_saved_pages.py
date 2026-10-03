#!/usr/bin/env python3
"""Import listing pages you saved from your own browser into the project cache.

Save a page in Chrome with Cmd+S, format "Webpage, HTML Only", into data/manual/ (any file name).
This reads each saved file, works out which house it is from the page itself, caches it as
data/raw/<redfin_id>.html.gz, and adds its description and photo URLs to data/descriptions.jsonl.
Processed files move to data/manual/done/. No network access.

Usage: python3 ingest_saved_pages.py [--in data/manual]
"""
import argparse
import csv
import gzip
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fetch_descriptions import current_photos, load_jsonl, parse, write_jsonl


def page_id(text):
    m = re.search(r'rel="canonical"[^>]*href="[^"]*/home/(\d+)"', text) or \
        re.search(r'property="og:url"[^>]*content="[^"]*/home/(\d+)"', text) or \
        re.search(r'redfin\.com/[A-Z]{2}/[^"\\\s]+/home/(\d+)', text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="data/manual")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/descriptions.jsonl")
    ap.add_argument("--houses", default="data/houses_5yr.csv")
    args = ap.parse_args()

    indir, raw_dir, out = Path(args.indir), Path(args.raw_dir), Path(args.out)
    done = indir / "done"
    files = sorted(p for p in indir.glob("*") if p.is_file() and p.suffix.lower() in (".html", ".htm"))
    if not files:
        print(f"no .html files in {indir}")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    done.mkdir(exist_ok=True)
    houses = {h["redfin_id"]: h for h in csv.DictReader(open(args.houses, newline="", encoding="utf-8"))}
    records = load_jsonl(out)

    ok = skipped = 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        rid = page_id(text)
        if not rid:
            print(f"SKIP {p.name}: can't tell which listing this is (not a Redfin home page?)")
            skipped += 1
            continue
        fallback = None
        if "fullScreenPhotoUrl" not in text:
            # Some saved pages never hydrated the photo lightbox, so only mid-size photoUrl entries exist.
            # Accept those if the listing's own remarks and photos are present.
            fb = list(dict.fromkeys(u.replace("\\u002F", "/") for u in re.findall(
                r'(?<![A-Za-z])photoUrl\\+":\\+"(https:(?:[^"\\]|\\u002F)+)', text)))
            if fb and "remarksDisplayLevel" in text:
                fallback = fb
        if "fullScreenPhotoUrl" not in text and fallback is None:
            print(f"SKIP {p.name} (home {rid}): no gallery data, so it was probably saved before the page finished loading, "
                  "or Redfin showed a limited page. Try saving it again after it fully loads.")
            skipped += 1
            continue
        h = houses.get(rid)
        if h is None:
            print(f"note: {p.name} is home {rid}, which isn't in {args.houses}; keeping it anyway")
        with gzip.open(raw_dir / f"{rid}.html.gz", "wt", encoding="utf-8") as f:
            f.write(text)
        desc, status, photos = parse(text)
        if fallback is not None:
            photos = fallback
        records[rid] = {
            "redfin_id": rid,
            "mls_id": h["mls_id"] if h else "",
            "url": h["redfin_url"] if h else "",
            "fetched_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "source": "manual_save",
            "description_status": status,
            "description": desc,
            "n_images": len(photos),
            "image_urls": photos,
            "image_urls_current": current_photos(text, photos),
        }
        shutil.move(str(p), done / p.name)
        print(f"OK   {h['address'] if h else rid}: {status}, {len(photos)} photos")
        ok += 1
    write_jsonl(out, records)
    print(f"\nimported {ok}, skipped {skipped}; {len(records)} houses now in {out}")


if __name__ == "__main__":
    main()
