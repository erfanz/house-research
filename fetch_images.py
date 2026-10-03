#!/usr/bin/env python3
"""Download a spread of listing photos for chosen homes from the URLs already in data/descriptions.jsonl.

Approved scope (user, 2026-09-30): throttled downloads for a shortlist only, for private research use.
Slow on purpose (2-5 s between photos), resumable (existing files are skipped), and it STOPS on the first sign of
blocking (HTTP 403/429, a redirect to ratelimited.redfin.com, or a non-image response). It never retries around a block.

Photos land in data/images/<redfin_id>/NN.jpg, with NN the photo's position in the gallery. A line per file is
appended to data/images/manifest.jsonl.

Usage:
  python3 fetch_images.py --ids 123836328,123062313 [--per-home 15]
  python3 fetch_images.py --ids-file shortlist.txt     (one redfin_id per line)
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIN_IMAGE_BYTES = 5_000


class Blocked(Exception):
    pass


def spread(urls, n, first):
    """The first `first` photos (agents lead with their best shots, often the view), then evenly spaced picks
    through the rest of the gallery, n in total."""
    if len(urls) <= n:
        return list(range(len(urls)))
    head = list(range(first))
    rest = n - first
    tail = [first + round(i * (len(urls) - 1 - first) / (rest - 1)) for i in range(rest)]
    return sorted(set(head + tail))


def fetch(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "image/avif,image/webp,image/jpeg,image/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.redfin.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if "ratelimited" in r.geturl():
                raise Blocked(f"redirected to {r.geturl()}")
            ctype = r.headers.get("Content-Type", "")
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"HTTP {e.code} (Retry-After: {e.headers.get('Retry-After')})")
        raise
    if not ctype.startswith("image/") or len(body) < MIN_IMAGE_BYTES:
        raise Blocked(f"not a real image (content-type {ctype!r}, {len(body)} bytes)")
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated redfin ids")
    ap.add_argument("--ids-file")
    ap.add_argument("--per-home", type=int, default=15)
    ap.add_argument("--first", type=int, default=8, help="how many leading photos to take in a row")
    ap.add_argument("--desc", default="data/descriptions.jsonl")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--delay-min", type=float, default=2.0)
    ap.add_argument("--delay-max", type=float, default=5.0)
    args = ap.parse_args()

    ids = [i.strip() for i in (args.ids or "").split(",") if i.strip()]
    if args.ids_file:
        ids += [l.strip() for l in Path(args.ids_file).read_text().split("\n") if l.strip()]
    recs = {}
    for line in Path(args.desc).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            recs[r["redfin_id"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = open(out / "manifest.jsonl", "a", encoding="utf-8")

    got = skipped = 0
    for n, rid in enumerate(ids, 1):
        r = recs.get(rid)
        if not r:
            print(f"[{n}/{len(ids)}] {rid}: not in {args.desc}, skipping")
            continue
        # image_urls_current keeps only photos matching the sale's MLS#. For a few homes that filter keeps just 1 photo,
        # so fall back to the full gallery when the filtered list is tiny.
        urls = r.get("image_urls_current") or []
        if len(urls) < 8 and len(r["image_urls"]) > len(urls):
            urls = r["image_urls"]
        d = out / rid
        d.mkdir(exist_ok=True)
        picks = spread(urls, args.per_home, args.first)
        new = 0
        for idx in picks:
            path = d / f"{idx:03d}.jpg"
            if path.exists() and path.stat().st_size >= MIN_IMAGE_BYTES:
                skipped += 1
                continue
            try:
                body = fetch(urls[idx])
            except Blocked as e:
                print(f"STOPPED: {e}. Not retrying; wait and ask before resuming.")
                manifest.close()
                raise SystemExit(2)
            except ConnectionError as e:
                print(f"STOPPED: connection reset/aborted ({e}); treating as possible throttling. Not retrying.")
                manifest.close()
                raise SystemExit(2)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  {rid} photo {idx}: network error {e}; skipping this photo")
                continue
            path.write_bytes(body)
            manifest.write(json.dumps({"redfin_id": rid, "index": idx, "of": len(urls), "bytes": len(body),
                                       "url": urls[idx]}) + "\n")
            manifest.flush()
            got += 1
            new += 1
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        print(f"[{n}/{len(ids)}] {rid}: {new} new, {len(picks)} wanted, gallery {len(urls)}")
    manifest.close()
    print(f"done: {got} downloaded, {skipped} already present")


if __name__ == "__main__":
    main()
