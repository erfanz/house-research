#!/usr/bin/env python3
"""Pull structured MLS fields from the cached listing pages (offline, no network).

For every data/raw/<redfin_id>.html.gz writes one line to data/structured_facts.jsonl:
  private_pool (1 if the MLS "Pool Information" section lists a Private Pool, else 0), pool_detail, spa, view_field (agent-entered View: list),
  sewer, water, roof, parking, lot_features, fireplace, ac.
Missing fields are null, never guessed.

Usage: python3 extract_structured.py [--raw-dir data/raw] [--out data/structured_facts.jsonl]
"""
import argparse
import gzip
import html
import json
import re
from pathlib import Path


def li(text, label):
    m = re.search(r">%s: ([^<]*)<" % re.escape(label), text)
    return m.group(1).strip() if m else None


def facts(page, rid):
    text = html.unescape(page)
    # Redfin's hasPrivatePool flag is false even on homes whose MLS details list a private pool, so don't use it.
    sec = re.search(r"<span>Pool Information</span></li>(.*?)</ul>", text)
    items = re.findall(r'entryItem ">([^<]*)<', sec.group(1)) if sec else []
    pool_detail = "; ".join(i for i in items if not i.startswith("Spa")) or None
    return {
        "redfin_id": rid,
        "private_pool": 1 if any(i.startswith("Private Pool") for i in items) else 0,
        "pool_detail": pool_detail,
        "spa": li(text, "Spa Features") or li(text, "Spa"),
        "view_field": li(text, "View"),
        "sewer": li(text, "Sewer"),
        "water": li(text, "Water Source"),
        "roof": li(text, "Roof"),
        "parking": li(text, "Parking Features"),
        "lot_features": li(text, "Lot Features"),
        "fireplace": 1 if re.search(r">Has Fireplace<", text) else None,
        "ac": 1 if re.search(r">Has A/C<|Heating & cooling: Has A/C", text) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/structured_facts.jsonl")
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p in sorted(Path(args.raw_dir).glob("*.html.gz")):
            rid = p.name[: -len(".html.gz")]
            page = gzip.open(p, "rt", encoding="utf-8", errors="replace").read()
            f.write(json.dumps(facts(page, rid)) + "\n")
            n += 1
    print(f"{n} pages -> {args.out}")


if __name__ == "__main__":
    main()
