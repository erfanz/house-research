#!/bin/bash
# Wait for the first photo download to finish, check it completed cleanly (no block), then download the second list.
# Stops without retrying if the first batch looks incomplete (a block) or the second batch stops on a block.
cd "$(dirname "$0")"
while pgrep -f "fetch_images.py --ids-file data/photo_shortlist.txt" > /dev/null; do sleep 30; done
python3 - <<'PY' || { echo "first batch incomplete (possible block): NOT starting the second batch" | tee data/images/chain.log; exit 3; }
import json, sys
from pathlib import Path
import fetch_images as f
recs = {}
for l in Path("data/descriptions.jsonl").read_text().split("\n"):
    if l.strip():
        r = json.loads(l); recs[r["redfin_id"]] = r
bad = []
for rid in Path("data/photo_shortlist.txt").read_text().split():
    urls = recs[rid].get("image_urls_current") or []
    if len(urls) < 8 and len(recs[rid]["image_urls"]) > len(urls):
        urls = recs[rid]["image_urls"]
    want = f.spread(urls, 15, 8)
    have = [i for i in want if (Path("data/images") / rid / f"{i:03d}.jpg").exists()]
    if len(have) < len(want) - 1:   # allow one skipped photo (network hiccup)
        bad.append((rid, len(have), len(want)))
print("incomplete homes:", bad)
sys.exit(1 if bad else 0)
PY
python3 -u fetch_images.py --ids-file data/photo_shortlist_2.txt --per-home 15 --first 8 > data/images/fetch2.log 2>&1
echo "second batch exit code $?" >> data/images/chain.log
