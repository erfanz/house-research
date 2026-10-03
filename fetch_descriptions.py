#!/usr/bin/env python3
"""Fetch each house's Redfin listing page slowly, cache the raw HTML, and extract
the agent description and photo URLs into data/descriptions.jsonl.

  data/raw/<redfin_id>.html.gz   cached page (fetched once, re-parsed offline)
  data/descriptions.jsonl        one JSON object per house

Reruns are incremental: houses already cached are never re-fetched.
Use --parse-only to rebuild the JSONL from the cache without touching the network.

Usage: python3 fetch_descriptions.py [--limit N] [--delay-min 120] [--delay-max 240] [--priority FILE] [--top N] [--parse-only]
"""
import argparse
import csv
import gzip
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIN_PAGE_BYTES = 200_000  # real listing pages are ~1 MB; blocks/captchas are tiny
PLACEHOLDER = "listed by non-mls member"
MAX_CONSECUTIVE_FAILURES = 3


class Blocked(Exception):
    pass


class RateLimited(Blocked):
    """Redfin's CDN sent us to ratelimited.redfin.com. Retrying soon only extends the penalty."""


def fetch(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if "ratelimited" in r.geturl():
                raise RateLimited(f"redirected to {r.geturl()}")
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"HTTP {e.code} (Retry-After: {e.headers.get('Retry-After')})")
        raise
    if len(body) < MIN_PAGE_BYTES or "captcha" in body[:5000].lower():
        raise Blocked(f"suspicious response ({len(body)} bytes), possible bot check")
    # Under throttling Redfin serves a stripped page with no gallery or remarks data
    # before it serves the hard block. Never cache those.
    if "fullScreenPhotoUrl" not in body:
        raise Blocked("degraded page (no gallery data), we are being throttled")
    return body


def js_decode(s):
    """Page JSON is embedded in a JS string, so text is escaped twice; undo both layers."""
    for _ in range(2):
        try:
            s = json.loads(f'"{s}"')
        except ValueError:
            break
    return html.unescape(s).strip()


def parse(page):
    # The first marketingRemarks list after remarksDisplayLevel belongs to the listing
    # itself. Later ones are past sale events or other homes on the page.
    desc, status = "", "no_remarks"
    m = re.search(r'remarksDisplayLevel\\":\d+,\\"marketingRemarks\\":\[(.*?)\]', page)
    if m:
        parts = re.findall(r'\\"marketingRemark\\":\\"(.*?)\\",\\"displayLevel', m.group(1))
        desc = "\n\n".join(js_decode(p) for p in parts).strip()
        if desc.lower().startswith(PLACEHOLDER):
            desc, status = "", "non_mls_placeholder"
        elif desc:
            status = "ok"
    photos = re.findall(r'fullScreenPhotoUrl\\":\\"(https:(?:[^"\\]|\\u002F)+)', page)
    photos = list(dict.fromkeys(u.replace("\\u002F", "/") for u in photos))
    return desc, status, photos


def current_photos(page, photos):
    """Photos from the sale's own MLS listing. Pages also carry photos from older listings of the house."""
    m = re.search(r"MLS# ?([A-Za-z0-9]+)", html.unescape(page))
    if not m:
        return photos
    cur = [u for u in photos if f"/{m.group(1)}_" in u]
    return cur or photos


def load_jsonl(path):
    if not path.exists():
        return {}
    out = {}
    # Not splitlines(): it also splits on U+2028/U+0085 etc., which can occur inside descriptions.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            out[rec["redfin_id"]] = rec
    return out


def write_jsonl(path, records):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", default="data/houses.csv")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/descriptions.jsonl")
    ap.add_argument("--limit", type=int, help="fetch at most N new pages this run")
    ap.add_argument("--delay-min", type=float, default=120)
    ap.add_argument("--delay-max", type=float, default=240)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--priority", help="ranked CSV (e.g. data/deals_2yr.csv); fetch houses in that order first")
    ap.add_argument("--top", type=int, help="with --priority, only consider the first N houses of that ranking")
    args = ap.parse_args()

    raw_dir, out_path = Path(args.raw_dir), Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    houses = list(csv.DictReader(open(args.houses, newline="", encoding="utf-8")))
    records = load_jsonl(out_path)
    if args.priority:
        order = [r["redfin_id"] for r in csv.DictReader(open(args.priority, newline="", encoding="utf-8"))]
        if args.top:
            order = order[: args.top]
        by_id = {h["redfin_id"]: h for h in houses}
        houses = [by_id[i] for i in order if i in by_id]

    fetched = failures = 0
    for i, h in enumerate(houses, 1):
        rid, cache = h["redfin_id"], raw_dir / f"{h['redfin_id']}.html.gz"
        have_raw = cache.exists()
        if have_raw and rid in records:
            continue
        if not have_raw:
            if args.parse_only:
                continue
            if args.limit is not None and fetched >= args.limit:
                break
            if fetched:
                time.sleep(random.uniform(args.delay_min, args.delay_max))
            try:
                page = fetch(h["redfin_url"])
            except RateLimited as e:
                print(f"[{i}/{len(houses)}] RATE LIMITED {h['address']}: {e}", file=sys.stderr)
                write_jsonl(out_path, records)
                print("Redfin rate-limited us. Stop for at least 24 hours before trying again.", file=sys.stderr)
                sys.exit(2)
            except Blocked as e:
                print(f"[{i}/{len(houses)}] BLOCKED {h['address']}: {e}", file=sys.stderr)
                write_jsonl(out_path, records)
                sys.exit("stopping so we don't hammer the site; retry later")
            except Exception as e:
                failures += 1
                print(f"[{i}/{len(houses)}] error {h['address']}: {e}", file=sys.stderr)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    write_jsonl(out_path, records)
                    sys.exit("too many consecutive failures, stopping")
                continue
            failures = 0
            fetched += 1
            with gzip.open(cache, "wt", encoding="utf-8") as f:
                f.write(page)
        else:
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                page = f.read()

        desc, status, photos = parse(page)
        records[rid] = {
            "redfin_id": rid,
            "mls_id": h["mls_id"],
            "url": h["redfin_url"],
            "fetched_at": datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "description_status": status,
            "description": desc,
            "n_images": len(photos),
            "image_urls": photos,
            "image_urls_current": current_photos(page, photos),
        }
        print(f"[{i}/{len(houses)}] {h['address']}: {status}, {len(photos)} photos")
        if len(records) % 10 == 0:
            write_jsonl(out_path, records)

    write_jsonl(out_path, records)
    by_status = {}
    for r in records.values():
        by_status[r["description_status"]] = by_status.get(r["description_status"], 0) + 1
    print(f"done: {len(records)}/{len(houses)} houses in {out_path} ({fetched} fetched this run) {by_status}")


if __name__ == "__main__":
    main()
