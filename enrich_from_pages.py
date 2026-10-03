#!/usr/bin/env python3
"""Extract extra facts from the cached listing pages (offline, no network).

For every data/raw/<redfin_id>.html.gz writes one line to data/page_facts.jsonl:
  apn, sold_date / sold_price (from the page, fills blanks in the CSV),
  beds and MLS# (which the CSV withholds for some listings), list_price / list_date,
  days_on_market and n_price_changes for the sale.

Usage: python3 enrich_from_pages.py [--raw-dir data/raw] [--out data/page_facts.jsonl]
"""
import argparse
import gzip
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_date(s):
    try:
        return datetime.strptime(s.strip(), "%b %d, %Y").date()
    except ValueError:
        return None


def find_apn(text):
    cands = re.findall(r"APN[^A-Za-z0-9]{0,6}([0-9][0-9A-Za-z\-\.]{4,})", text)
    # Meta descriptions are truncated with "...", so ignore those and keep the cleanest, longest.
    clean = [c.rstrip(".") for c in cands if "..." not in c]
    return max(clean, key=len) if clean else ""


def parse_events(page):
    """Property history events, newest first: (description, price, date)."""
    i = page.find('\\"events\\":[')
    if i < 0:
        return []
    chunks = page[i:i + 60000].split('{\\"isEventAdminOnly\\":')[1:]
    out = []
    for c in chunks:
        d = re.search(r'eventDescription\\":\\"([^\\]+)\\"', c)
        p = re.search(r'\\"price\\":(\d+)', c)
        t = re.search(r'eventDateString\\":\\"([^\\]+)\\"', c)
        e = re.search(r'\\"eventDate\\":(\d+)', c)
        # Some events (Listed, Price Changed, Contingent) have only an epoch-ms timestamp.
        # Arizona does not observe DST, so UTC-7 all year.
        dt = parse_date(t.group(1)) if t else None
        if dt is None and e:
            dt = (datetime.fromtimestamp(int(e.group(1)) / 1000, timezone.utc) - timedelta(hours=7)).date()
        if d and dt:
            out.append((html.unescape(d.group(1)), int(p.group(1)) if p else None, dt))
    return out


def facts(page, rid):
    text = html.unescape(page.replace("\\u002F", "/"))
    rec = {"redfin_id": rid, "apn": find_apn(text)}
    m = re.search(r"sold for \$([\d,]+) on ([A-Z][a-z]{2} \d{1,2}, \d{4})", text)
    sold_date = parse_date(m.group(2)) if m else None
    rec["page_sold_price"] = int(m.group(1).replace(",", "")) if m else None
    rec["page_sold_date"] = sold_date.isoformat() if sold_date else ""

    # og:description looks like "(ARMLS) 2 beds, 2 baths, 1520 sq. ft. house located at ... MLS# 538099."
    b = re.search(r"(\d+) beds?,", text)
    rec["page_beds"] = int(b.group(1)) if b else None
    mls = re.search(r"MLS# ?([A-Za-z0-9]+)", text)
    rec["page_mls_id"] = mls.group(1) if mls else ""

    # Events are newest first. The listing for this sale is the latest "Listed" event on or before
    # the sale date (within 2 years); count price changes between the two.
    events = [(d, p, dt) for d, p, dt in parse_events(page) if dt]
    rec.update(list_price=None, list_date="", days_on_market=None, n_price_changes=0)
    # Most recent listing of any kind: descriptions on pages for off-market sales come from this
    # earlier listing, not from the sale itself.
    listed = [dt for d, _, dt in events if d.startswith("Listed")]
    rec["last_listed_date"] = max(listed).isoformat() if listed else ""
    if sold_date:
        for k, (d, p, dt) in enumerate(events):
            # Skip a listing if another sale happened after it: it belongs to that earlier sale.
            intervening = any(d2.startswith("Sold") and dt <= dt2 < sold_date for d2, _, dt2 in events)
            if d.startswith("Listed") and dt <= sold_date and (sold_date - dt).days < 730 and not intervening:
                rec["list_price"], rec["list_date"] = p, dt.isoformat()
                rec["days_on_market"] = (sold_date - dt).days
                rec["n_price_changes"] = sum(
                    1 for d2, _, dt2 in events if d2.startswith("Price Changed") and dt <= dt2 <= sold_date)
                break
    rec["off_market_sale"] = rec["list_price"] is None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/page_facts.jsonl")
    args = ap.parse_args()
    files = sorted(Path(args.raw_dir).glob("*.html.gz"))
    n_apn = n_list = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p in files:
            with gzip.open(p, "rt", encoding="utf-8") as g:
                rec = facts(g.read(), p.name.split(".")[0])
            n_apn += bool(rec["apn"])
            n_list += rec["list_price"] is not None
            f.write(json.dumps(rec) + "\n")
    print(f"{len(files)} pages -> {args.out}; apn found: {n_apn}, list price found: {n_list}")


if __name__ == "__main__":
    main()
