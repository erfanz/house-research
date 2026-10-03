#!/usr/bin/env python3
"""Merge Redfin CSV exports in data/redfin_exports/ into data/houses.csv.

One row per property, keyed by Redfin's own property id (the number at the end
of the listing URL), which is present on every row, unlike MLS#.

Usage: python3 merge_exports.py [--in data/redfin_exports] [--out data/houses.csv]
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

KEEP_TYPES = {"Single Family Residential"}

OUT_COLUMNS = [
    "redfin_id", "mls_id", "address_link", "address", "city", "state", "zip",
    "neighborhood", "area", "sold_price", "sold_date", "sqft", "lot_sqft",
    "beds", "baths", "year_built", "hoa_month", "price_per_sqft",
    "days_on_market", "property_type", "lat", "lng", "redfin_url", "source",
    "flags",
]


def num(s, cast=float):
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        v = cast(s)
    except ValueError:
        return None
    return v


def to_int(s):
    v = num(s, float)
    return None if v is None else int(round(v))


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%B-%d-%Y").date().isoformat()
    except ValueError:
        return None


def load_polygon(path):
    return [tuple(float(v) for v in pt.split()) for pt in Path(path).read_text().strip().split(",")]


def in_polygon(lng, lat, poly):
    """Ray casting; poly is a list of (lng, lat) vertices."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def read_export(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    # Row 2 is Redfin's MLS disclaimer, so keep only rows that match the header width.
    return [dict(zip(header, r)) for r in rows[1:] if len(r) == len(header)]


def normalize(raw, area):
    url = next((v for k, v in raw.items() if k.startswith("URL")), "").strip()
    m = re.search(r"/home/(\d+)", url)
    if not m:
        return None
    flags = []
    sold_date = parse_date(raw["SOLD DATE"])
    if not sold_date:
        flags.append("no_sold_date")
    sqft = to_int(raw["SQUARE FEET"])
    if sqft is None:
        flags.append("no_sqft")
    if not raw["MLS#"].strip():
        flags.append("no_mls")
    beds = to_int(raw["BEDS"])
    if beds is None:
        flags.append("no_beds")
    if not raw["YEAR BUILT"].strip():
        flags.append("no_year_built")
    address = raw["ADDRESS"].strip()
    return {
        "redfin_id": m.group(1),
        "mls_id": raw["MLS#"].strip(),
        "address_link": f'=HYPERLINK("{url}","{address}")',
        "address": address,
        "city": raw["CITY"].strip(),
        "state": raw["STATE OR PROVINCE"].strip(),
        "zip": raw["ZIP OR POSTAL CODE"].strip(),
        "neighborhood": raw["LOCATION"].strip(),
        "area": area,
        "sold_price": to_int(raw["PRICE"]),
        "sold_date": sold_date or "",
        "sqft": sqft,
        "lot_sqft": to_int(raw["LOT SIZE"]),
        "beds": beds,
        "baths": num(raw["BATHS"]),
        # Blank HOA means none listed; treated as 0.
        "hoa_month": to_int(raw["HOA/MONTH"]) or 0,
        "price_per_sqft": to_int(raw["$/SQUARE FEET"]),
        "days_on_market": to_int(raw["DAYS ON MARKET"]),
        "property_type": raw["PROPERTY TYPE"].strip(),
        "year_built": to_int(raw["YEAR BUILT"]),
        "lat": num(raw["LATITUDE"]),
        "lng": num(raw["LONGITUDE"]),
        "redfin_url": url,
        "source": raw["SOURCE"].strip(),
        "flags": ";".join(flags),
    }


def completeness(rec):
    return sum(1 for v in rec.values() if v not in (None, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="data/redfin_exports")
    ap.add_argument("--out", default="data/houses.csv")
    ap.add_argument("--clip-polygon", help="file of 'lng lat,lng lat,...'; rows from files matching --clip-prefix outside it are dropped")
    ap.add_argument("--clip-prefix", default="ws_")
    ap.add_argument("--clip-area", default="west_sedona", help="area label given to rows kept by the clip")
    args = ap.parse_args()
    clip = load_polygon(args.clip_polygon) if args.clip_polygon else None

    files = sorted(Path(args.indir).glob("*.csv"))
    if not files:
        sys.exit(f"no CSV files in {args.indir}")

    merged, dropped_type, dupes, clipped = {}, 0, 0, 0
    for path in files:
        area = path.stem
        clip_this = clip is not None and path.stem.startswith(args.clip_prefix)
        if clip_this:
            area = args.clip_area
        for raw in read_export(path):
            if raw["PROPERTY TYPE"].strip() not in KEEP_TYPES:
                dropped_type += 1
                continue
            if clip_this:
                lng, lat = num(raw["LONGITUDE"]), num(raw["LATITUDE"])
                if lng is None or lat is None or not in_polygon(lng, lat, clip):
                    clipped += 1
                    continue
            rec = normalize(raw, area)
            if rec is None:
                print(f"warning: no property id in {path.name}: {raw['ADDRESS']}", file=sys.stderr)
                continue
            prev = merged.get(rec["redfin_id"])
            if prev is None:
                merged[rec["redfin_id"]] = rec
                continue
            dupes += 1
            # Same property in several exports: keep the fuller record, union the areas.
            areas = sorted(set(prev["area"].split("|")) | {area})
            keep = rec if completeness(rec) > completeness(prev) else prev
            keep["area"] = "|".join(areas)
            merged[rec["redfin_id"]] = keep

    out = sorted(merged.values(), key=lambda r: (r["sold_date"] == "", r["sold_date"]), reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(out)

    no_date = sum(1 for r in out if not r["sold_date"])
    print(f"{len(files)} files -> {len(out)} houses written to {args.out}")
    print(f"dropped {dropped_type} non-single-family rows, {clipped} rows outside the clip polygon, merged {dupes} duplicates")
    print(f"{no_date} rows have no sold_date (public-record sales; can't verify the 2-year window)")


if __name__ == "__main__":
    main()
