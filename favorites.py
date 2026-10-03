#!/usr/bin/env python3
"""Favorite-houses list: paste a Redfin link, get the facts, photos, Red Rock View rating, a fair price range and a deal verdict.

  python3 favorites.py add <redfin url> [--no-rate] [--no-photos]
  python3 favorites.py list
  python3 favorites.py show <redfin_id or url>
  python3 favorites.py rate <redfin_id or url>      (re-run the photo rating, e.g. after setting ANTHROPIC_API_KEY)
  python3 favorites.py remove <redfin_id or url> [--purge]
  python3 favorites.py serve                        (map on http://localhost:8765 with add/remove buttons)

State: data/favorites.jsonl (one line per house). Page cache: data/raw/<id>.html.gz. Photos: data/favorites/<id>/.

Network rules (same as fetch_descriptions.py / fetch_images.py): one page request per new house, photos 2-5 s apart, and it
STOPS on the first sign of blocking (403/429, rate-limit redirect, degraded page, non-image). It never retries around a block.

Rating: Claude vision (ANTHROPIC_API_KEY from the environment, model FAVORITES_MODEL, default claude-sonnet-5-5), same 0-3
Red Rock scale as red_rock_rating.py, read from the first 7 photos like the 206-home training set. Without a key the house is
saved with the rating "pending" and the price range is provisional (view/red-rock inputs set to the sample average).

Fair price: the two-stage grid formula in data/GRID_MODEL.md, refit on every run from data/*.csv (pure stdlib, ~1 s).
The range is +-13%, the band 50% of held-out sales fell inside (held-out rmse of ln price 0.193). It is a ranking aid, not a valuation.
The formula only knows West Sedona + Chapel; outside its 2 km cells, or without sqft/lot/year, no range is given.
"""
import argparse
import base64
import csv
import gzip
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import enrich_from_pages as ep
import fetch_descriptions as fd
import fetch_images as fi
import grid_model as g
import grid_two_stage as gts

FAV = Path("data/favorites.jsonl")
RAW = Path("data/raw")
PHOTOS = Path("data/favorites")
CELL_KM, LAMBDA = 2.0, 1
BAND = 0.13                 # half-width of the fair range in ln units; 50% of held-out sales landed inside +-13%
MIN_CELL_SALES = 10         # fewer 5-yr sales than this: the neighbourhood level is rough
N_PHOTOS, N_RATED = 12, 7
API = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("FAVORITES_MODEL", "claude-sonnet-5-5")

RUBRIC = """You are rating a Sedona, AZ house for a price model. You get the listing description and the first listing photos.
Return ONLY a JSON object with these keys:

red_rock_rating (0-3):
  0 no red rock visible or claimed
  1 glimpse: distant, small in the frame, or partly hidden by trees
  2 clear red-rock view from the house or deck, but at a distance or wide and muted
  3 large red-rock formations dominate the view from the main living space or deck, close or wide and unobstructed
  Judge from the photos first; text alone is capped at 2. A view that is not photographed may still exist, so rate what is visible.
red_rock_evidence: one sentence saying which photo(s) or text show it.
view_score (0-5, null if neither text nor photos say anything): 0 none, 1 minimal, 2 limited, 3 partial or hedged,
  4 clear scenic view, 5 panoramic or dramatic.
forest_backing (0/1): the lot backs onto or adjoins national forest or permanent open land (description or photos).
pool (0/1): a private pool (not just a hot tub).
renovated (0/1): recently remodeled or fully updated (the description says so, or the photos clearly show new finishes).
confidence: "high" or "low".
Do not guess beyond the evidence; use 0 or null when it is not there."""


# ---------- helpers ----------

def redfin_id(s):
    m = re.search(r"/home/(\d+)", s) or re.fullmatch(r"\s*(\d+)\s*", s)
    if not m:
        raise SystemExit(f"Not a Redfin home link (expected .../home/<id>): {s}")
    return m.group(1)


def load_favs():
    out = {}
    if FAV.exists():
        for line in FAV.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                out[r["redfin_id"]] = r
    return out


def save_favs(favs):
    FAV.parent.mkdir(exist_ok=True)
    tmp = FAV.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in favs.values()), encoding="utf-8")
    tmp.replace(FAV)
    # keep map.html in step with the list
    r = subprocess.run([sys.executable, "build_map.py"], capture_output=True, text=True)
    print("map.html rebuilt" if r.returncode == 0 else f"map rebuild failed: {r.stderr.strip()[-200:]}")


def money(x):
    return f"${x:,.0f}" if x is not None else "-"


# ---------- page -> facts ----------

def get_page(rid, url):
    """Cached page if we have one, else exactly one throttled-polite request."""
    cache = RAW / f"{rid}.html.gz"
    if cache.exists():
        return gzip.open(cache, "rt", encoding="utf-8", errors="replace").read(), True
    page = fd.fetch(url)          # raises fd.Blocked / fd.RateLimited
    RAW.mkdir(exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        f.write(page)
    return page, False


def num(s):
    return int(s.replace(",", "")) if s else None


def extract(page, rid):
    """Facts for the subject house. Anything not found is None, never guessed."""
    text = html.unescape(page.replace("\\u002F", "/"))
    og = re.search(r'og:description" content="([^"]*)"', text)
    og = og.group(1) if og else ""
    f = {"redfin_id": rid}
    m = re.search(r"located at (.+?), ([A-Za-z .]+), ([A-Z]{2}) (\d{5})", og)
    f["address"] = f"{m.group(1)}, {m.group(2)} {m.group(3)} {m.group(4)}" if m else None
    if not f["address"]:        # active listings: "... ∙ 2250 E Mule Deer Rd, Sedona, AZ 86336 ∙ $1,100,000 ∙ ..."
        m = re.search(r"∙ ([^∙]+?, [A-Za-z .]+, [A-Z]{2} \d{5})", og) or re.search(r'og:title" content="([^"]+?, [A-Z]{2} \d{5})', text)
        f["address"] = m.group(1).strip() if m else None
    m = re.search(r"([\d,]+) sq\. ft\.", og)
    f["sqft"] = num(m.group(1)) if m else None
    m = re.search(r"(\d+) beds?,", og)
    f["beds"] = int(m.group(1)) if m else None
    m = re.search(r"([\d.]+) baths?", og)
    f["baths"] = float(m.group(1)) if m else None
    m = re.search(r"sq\. ft\. (\w[\w ]*?) located", og)
    f["property_type"] = m.group(1) if m else None
    m = re.search(r'"latitude\\?":(-?[\d.]+)', text)
    f["lat"] = float(m.group(1)) if m else None
    m = re.search(r'"longitude\\?":(-?[\d.]+)', text)
    f["lng"] = float(m.group(1)) if m else None
    m = re.search(r'yearBuilt\\?":(\d{4})', text)
    f["year_built"] = int(m.group(1)) if m else None
    m = re.search(r'lotSize\\?":(\d+)', text) or re.search(r"Lot Size Area\"[^\]]*?amenityValues\":\[\"([\d,]+)\"", text)
    f["lot_sqft"] = num(m.group(1)) if m else None
    m = re.search(r"MLS# ?([A-Za-z0-9]+)", og)
    f["mls_id"] = m.group(1) if m else None

    # status + prices. Sold pages say "sold for $X on <date>"; active ones are expected to say "listed for $X"
    # (not verified: no active page was cached when this was written), with the property history as a fallback.
    sold = re.search(r"sold for \$([\d,]+) on ([A-Z][a-z]{2} \d{1,2}, \d{4})", og)
    events = [(d, p, dt) for d, p, dt in ep.parse_events(page) if dt]
    f.update(status=None, ask_price=None, sold_price=None, sold_date=None)
    if sold:
        f.update(status="sold", sold_price=num(sold.group(1)),
                 sold_date=datetime.strptime(sold.group(2), "%b %d, %Y").date().isoformat())
    else:
        pre = re.match(r"\([A-Z]+\) (Pending|Contingent|Active|For Sale|Coming Soon|Under Contract)", og, re.I)
        lst = re.search(r"(?:listed|for sale) (?:for|at) \$([\d,]+)", og) or re.search(r"∙ \$([\d,]+) ∙", og)
        price = num(lst.group(1)) if lst else None
        if price is None:       # latest price event of the current listing
            for d, p, dt in events:
                if d.startswith(("Listed", "Price Changed")) and p:
                    price = p
                    break
                if d.startswith(("Sold", "Delisted", "Listing Removed")):
                    break
        f["ask_price"] = price
        f["status"] = (pre.group(1).lower().replace("for sale", "active") if pre else ("active" if price else "off-market/unknown"))
    desc, dstatus, photos = fd.parse(page)
    f["description"], f["description_status"] = desc, dstatus
    f["photo_urls"] = fd.current_photos(page, photos)
    return f


# ---------- photos ----------

def download_photos(f):
    urls = f["photo_urls"]
    if not urls:
        return []
    d = PHOTOS / f["redfin_id"]
    d512 = d / "512"
    d512.mkdir(parents=True, exist_ok=True)
    picks = fi.spread(urls, N_PHOTOS, 6)
    paths = []
    for idx in picks:
        p, small = d / f"{idx:03d}.jpg", d512 / f"{idx:03d}.jpg"
        if not (p.exists() and p.stat().st_size >= fi.MIN_IMAGE_BYTES):
            try:
                p.write_bytes(fi.fetch(urls[idx]))
            except fi.Blocked as e:
                print(f"  photos STOPPED: {e}. Not retrying; wait before running `rate`/`add` again.")
                break
            except ConnectionError as e:
                print(f"  photos STOPPED: connection reset ({e}); treating as throttling. Not retrying.")
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  photo {idx}: network error {e}; skipping")
                continue
            time.sleep(random.uniform(2, 5))
        if not small.exists():
            subprocess.run(["sips", "-Z", "512", "-s", "formatOptions", "60", str(p), "--out", str(small)],
                           capture_output=True)
        if small.exists():
            paths.append(str(small))
    return paths


# ---------- rating (Claude vision) ----------

def parse_rating(txt):
    j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    j["red_rock_rating"] = int(j["red_rock_rating"])
    if not 0 <= j["red_rock_rating"] <= 3:
        raise ValueError(f"rating {j['red_rock_rating']} out of range")
    return j


def rate_cli(f):
    """Rate with the user's own Claude Code login (`claude -p`), no API key. Claude views the photos with its Read tool."""
    imgs = [str(Path(p).resolve()) for p in f.get("photos", []) if Path(p).exists()][:N_RATED]
    if not imgs and not f.get("description"):
        return {"status": "pending", "reason": "no photos or description to rate"}
    prompt = (f"{RUBRIC}\n\nAddress: {f.get('address')}\nUse the Read tool to look at each of these {len(imgs)} photos, in order:\n"
              + "\n".join(imgs) + f"\n\nListing description:\n{f.get('description') or '(none)'}\n\n"
              "Reply with the JSON object only, no other text.")
    cmd = ["claude", "-p", prompt, "--tools", "Read", "--allowedTools", "Read", "--add-dir", str(PHOTOS.resolve())]
    if os.environ.get("FAVORITES_MODEL"):
        cmd += ["--model", MODEL]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode:
            return {"status": "pending", "reason": f"claude exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"}
        j = parse_rating(r.stdout)
    except (subprocess.TimeoutExpired, OSError, AttributeError, KeyError, ValueError) as e:
        return {"status": "pending", "reason": f"rating failed: {type(e).__name__}: {e}"}
    j.update(status="ok", model=f"Claude Code ({os.environ.get('FAVORITES_MODEL') or 'default model'})", photos_used=len(imgs),
             rated_at=datetime.now(timezone.utc).isoformat())
    if not imgs:
        j["red_rock_rating"] = min(j["red_rock_rating"], 2)
        j["confidence"] = "low"
    return j


def rate(f):
    # Default: Claude Code CLI. Set FAVORITES_RATER=api to use ANTHROPIC_API_KEY instead.
    if os.environ.get("FAVORITES_RATER") == "api":
        return rate_api(f)
    return rate_cli(f)


def rate_api(f):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"status": "pending", "reason": "no API key; ask Claude Code to rate it from the photos"}
    imgs = [p for p in f.get("photos", []) if Path(p).exists()][:N_RATED]
    if not imgs and not f.get("description"):
        return {"status": "pending", "reason": "no photos or description to rate"}
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(Path(p).read_bytes()).decode()}} for p in imgs]
    content.append({"type": "text", "text": f"{RUBRIC}\n\nAddress: {f.get('address')}\nPhotos shown: {len(imgs)}\n"
                                            f"Listing description:\n{f.get('description') or '(none)'}"})
    body = json.dumps({"model": MODEL, "max_tokens": 800, "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        txt = "".join(b.get("text", "") for b in out["content"] if b["type"] == "text")
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        rating = int(j["red_rock_rating"])
        if not 0 <= rating <= 3:
            raise ValueError(f"rating {rating} out of range")
    except urllib.error.HTTPError as e:
        return {"status": "pending", "reason": f"API error HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, AttributeError) as e:
        return {"status": "pending", "reason": f"rating failed: {type(e).__name__}: {e}"}
    j.update(status="ok", model=MODEL, photos_used=len(imgs), rated_at=datetime.now(timezone.utc).isoformat())
    if not imgs:
        j["red_rock_rating"] = min(j["red_rock_rating"], 2)    # text-only is capped at 2, as in red_rock_rating.py
        j["confidence"] = "low"
    return j


# ---------- valuation ----------

_MODEL = None


def fit_model():
    """Final two-stage formula (GRID_MODEL.md): cell levels from the 5-yr sales, stage 2 on the 2-yr sales."""
    global _MODEL
    if _MODEL:
        return _MODEL
    rows2, rows5 = g.load("2yr"), g.load("5yr")
    ids = [r["id"] for r in rows2]
    random.Random(11).shuffle(ids)
    eff = {}
    for k in range(5):                                   # cross-fitted cell level for each training home
        grp = set(ids[k::5])
        e = gts.stage1(rows5, grp, CELL_KM, LAMBDA)
        for r in rows2:
            if r["id"] in grp:
                eff[r["id"]] = e.get(g.cell_ids([r], CELL_KM)[0], 0.0)
    means = g.feature_means(rows2, gts.FULL)
    beta, _ = gts.ols(gts.design2(rows2, means, eff), [r["y"] for r in rows2])
    cells = {}
    for r in rows5:
        c = g.cell_ids([r], CELL_KM)[0]
        cells[c] = cells.get(c, 0) + 1
    # grid origin and projection exactly as grid_model.load()
    hall = [(g.fnum(r["lat"]), g.fnum(r["lng"])) for r in csv.DictReader(open("data/houses_5yr.csv")) if g.fnum(r["lat"])]
    origin = (min(a for a, _ in hall), min(b for _, b in hall), math.cos(math.radians(sum(a for a, _ in hall) / len(hall))))
    _MODEL = {"beta": beta, "means": means, "cell_eff": gts.stage1(rows5, set(), CELL_KM, LAMBDA),
              "cell_n": cells, "origin": origin}
    return _MODEL


def valuation(f):
    """Fair range and verdict. Returns a dict; 'range' is None with a 'why_not' when the model cannot be applied."""
    missing = [k for k in ("sqft", "lot_sqft", "year_built", "lat", "lng") if not f.get(k)]
    if missing:
        return {"range": None, "why_not": "missing " + ", ".join(missing) + " on the page"}
    m = fit_model()
    lat0, lng0, cl = m["origin"]
    x_km, y_km = (f["lng"] - lng0) * g.KM_PER_DEG * cl, (f["lat"] - lat0) * g.KM_PER_DEG
    cell = (int(x_km // CELL_KM), int(y_km // CELL_KM))
    if cell not in m["cell_eff"]:
        return {"range": None, "why_not": "outside the modeled area (West Sedona + Chapel); the formula would be extrapolating"}
    notes = []
    if f.get("property_type") and f["property_type"].lower() not in ("house", "single family", "single family residential"):
        notes.append(f"property type is '{f['property_type']}'; the model was fitted on single-family homes")
    n_cell = m["cell_n"].get(cell, 0)
    if n_cell < MIN_CELL_SALES:
        notes.append(f"only {n_cell} five-year sales in this neighbourhood cell; its price level is rough")
    r = f.get("rating") or {}
    ok = r.get("status") == "ok"
    mean = m["means"]
    v = {
        "ln_sqft": math.log(f["sqft"]), "ln_lot": math.log(f["lot_sqft"]),
        "age": max(0, date.today().year - f["year_built"]) / 10, "t": (date.today() - g.T0).days / 365.25,
        "view_score": r.get("view_score") if ok and r.get("view_score") is not None else mean["view_score"],
        "red_rock": r["red_rock_rating"] if ok else mean["red_rock"],
        "forest": r["forest_backing"] if ok else mean["forest"],
        "pool": r["pool"] if ok else mean["pool"],
        "renovated": r["renovated"] if ok else mean["renovated"],
    }
    if not ok:
        notes.append("PROVISIONAL: no photo rating yet, so view/red-rock/forest/pool/renovated are set to the sample average")
    x = [1.0] + [v[k] for k in gts.FULL] + [m["cell_eff"][cell]]
    mid = math.exp(sum(a * b for a, b in zip(x, m["beta"])))
    lo, hi = mid * math.exp(-BAND), mid * math.exp(BAND)
    out = {"range": [round(lo), round(hi)], "mid": round(mid), "cell": list(cell), "cell_sales_5yr": n_cell,
           "notes": notes, "provisional": not ok}
    price = f.get("ask_price")
    if price:
        out["compare"], out["compare_price"] = "asking", price
    elif f.get("sold_price"):
        out["compare"], out["compare_price"] = "sold", f["sold_price"]
    if out.get("compare_price"):
        p = out["compare_price"]
        out["pct_vs_mid"] = round(100 * (p / mid - 1), 1)
        if f.get("status") == "sold":
            out["verdict"] = None                      # already sold: informational only
        elif p < lo:
            out["verdict"] = "GOOD DEAL"
        elif p > hi:
            out["verdict"] = "BAD DEAL (priced above the fair range)"
        else:
            out["verdict"] = "FAIR PRICE (inside the range)"
    return out


# ---------- display ----------

RR = {0: "none", 1: "glimpse", 2: "clear, distant/muted", 3: "dominant, unobstructed"}


def show(f):
    r, v = f.get("rating") or {}, valuation(f)
    print(f"\n{f.get('address') or f['redfin_id']}   [{f['redfin_id']}]")
    print(f"  {f['url']}")
    bb = f"{f.get('beds') or '?'} bd / {f.get('baths') or '?'} ba, {f.get('sqft') or '?'} sqft, lot {f.get('lot_sqft') or '?'} sqft, built {f.get('year_built') or '?'}"
    print(f"  {bb}")
    st = f.get("status") or "?"
    if st == "sold":
        print(f"  Status: sold {money(f['sold_price'])} on {f['sold_date']}")
    else:
        print(f"  Status: {st}, asking {money(f.get('ask_price'))}")
    if r.get("status") == "ok":
        print(f"  Red Rock View: {r['red_rock_rating']}/3 ({RR[r['red_rock_rating']]}), confidence {r.get('confidence')}, "
              f"from {r.get('photos_used')} photos by {r.get('model')}")
        print(f"    {r.get('red_rock_evidence')}")
        print(f"    view {r.get('view_score')}/5, forest backing {r.get('forest_backing')}, pool {r.get('pool')}, renovated {r.get('renovated')}")
    else:
        print(f"  Red Rock View: PENDING ({r.get('reason', 'not rated')}). Run: python3 favorites.py rate {f['redfin_id']}")
    if v["range"] is None:
        print(f"  Fair price: not computed - {v['why_not']}")
    else:
        lo, hi = v["range"]
        print(f"  Fair price range: {money(lo)} - {money(hi)}   (point estimate {money(v['mid'])}; ~50% of held-out sales fell inside this band)")
        if v.get("compare_price"):
            tag = "asking" if v["compare"] == "asking" else "sold at"
            print(f"  {tag.capitalize()} {money(v['compare_price'])} is {v['pct_vs_mid']:+.1f}% vs the point estimate")
        if v.get("verdict"):
            print(f"  Verdict: {v['verdict']}")
        for n in v["notes"]:
            print(f"  note: {n}")
        print("  (Typical model miss is ~12%; condition, interior and off-market factors are not in the data.)")


# ---------- commands ----------

def cmd_add(a):
    rid = redfin_id(a.url)
    favs = load_favs()
    try:
        page, cached = get_page(rid, a.url)
    except fd.Blocked as e:
        raise SystemExit(f"STOPPED: Redfin blocked or throttled the request ({e}). Not retrying; wait and run it again later.")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not fetch the page: {e}")
    f = extract(page, rid)
    f.update(url=a.url.split("?")[0], added=favs.get(rid, {}).get("added") or datetime.now(timezone.utc).isoformat(),
             page_from_cache=cached)
    print(f"{'Using cached page' if cached else 'Fetched page'}: {f['address']}")
    f["photos"] = [] if a.no_photos else download_photos(f)
    if not a.no_photos:
        print(f"  {len(f['photos'])} photos saved in {PHOTOS / rid}")
    f["rating"] = {"status": "pending", "reason": "skipped (--no-rate)"} if a.no_rate else rate(f)
    favs[rid] = f
    save_favs(favs)
    show(f)


def pick(a):
    favs = load_favs()
    rid = redfin_id(a.target)
    if rid not in favs:
        raise SystemExit(f"{rid} is not in the favorites list")
    return favs, favs[rid]


def cmd_show(a):
    show(pick(a)[1])


def cmd_rate(a):
    favs, f = pick(a)
    if not f.get("photos"):
        f["photos"] = download_photos(f)
    f["rating"] = rate(f)
    save_favs(favs)
    show(f)


def cmd_set_rating(a):
    """Store a rating made outside the API, e.g. by Claude Code viewing data/favorites/<id>/512/*.jpg."""
    favs, f = pick(a)
    f["rating"] = {"status": "ok", "model": a.by, "photos_used": a.photos_used, "red_rock_rating": a.red_rock,
                   "red_rock_evidence": a.evidence, "view_score": a.view, "forest_backing": a.forest, "pool": a.pool,
                   "renovated": a.renovated, "confidence": a.confidence, "rated_at": datetime.now(timezone.utc).isoformat()}
    save_favs(favs)
    show(f)


def cmd_remove(a):
    favs, f = pick(a)
    del favs[f["redfin_id"]]
    save_favs(favs)
    print(f"removed {f.get('address')}")
    if a.purge:
        import shutil
        shutil.rmtree(PHOTOS / f["redfin_id"], ignore_errors=True)
        (RAW / f"{f['redfin_id']}.html.gz").unlink(missing_ok=True)
        print("  also deleted its photos and cached page")


def cmd_list(a):
    favs = load_favs()
    if not favs:
        print("No favorites yet. Add one: python3 favorites.py add <redfin link>")
        return
    print(f"{'address':<34}{'price':>12}{'rr':>4}  {'fair range':<25}verdict")
    for f in favs.values():
        v, r = valuation(f), f.get("rating") or {}
        price = f.get("ask_price") if f.get("status") != "sold" else f.get("sold_price")
        rr = f"{r['red_rock_rating']}/3" if r.get("status") == "ok" else "n/a"
        rng = f"{money(v['range'][0])}-{money(v['range'][1])}" if v["range"] else "n/a"
        verdict = v.get("verdict") or ("sold" if f.get("status") == "sold" else v.get("why_not", ""))
        if v.get("provisional") and v["range"]:
            verdict += " (provisional)"
        print(f"{(f.get('address') or f['redfin_id'])[:33]:<34}{money(price):>12}{rr:>4}  {rng:<25}{verdict}")


def cmd_serve(a):
    """Serve map.html on localhost with add/remove buttons that call this same CLI."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    lock = threading.Lock()                       # one add/remove at a time (Redfin throttling, file writes)
    host_ok = {f"localhost:{a.port}", f"127.0.0.1:{a.port}"}

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/map.html"):
                self.send(200, Path("map.html").read_bytes(), "text/html; charset=utf-8")
            else:
                self.send(404, "{}")

        def do_POST(self):
            # DNS-rebinding / cross-site guard: only our own page on localhost may call this
            if self.headers.get("Host") not in host_ok or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(403, json.dumps({"ok": False, "msg": "forbidden"}))
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/add":
                    url = str(body.get("url", "")).strip()
                    if not re.fullmatch(r"https://www\.redfin\.com/[^\s?#]*/home/\d+/?(\?\S*)?", url):
                        return self.send(400, json.dumps({"ok": False, "msg": "That is not a Redfin home link (…/home/<id>)."}))
                    cmd = ["add", url]
                elif self.path == "/api/remove":
                    rid = str(body.get("id", ""))
                    if not rid.isdigit():
                        return self.send(400, json.dumps({"ok": False, "msg": "bad id"}))
                    cmd = ["remove", rid]
                else:
                    return self.send(404, "{}")
            except (ValueError, TypeError):
                return self.send(400, json.dumps({"ok": False, "msg": "bad request"}))
            with lock:
                r = subprocess.run([sys.executable, "favorites.py", *cmd], capture_output=True, text=True, timeout=900)
            msg = (r.stdout if r.returncode == 0 else (r.stderr or r.stdout)).strip()
            self.send(200, json.dumps({"ok": r.returncode == 0, "msg": msg[-600:]}))

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"Map with add/remove: http://localhost:{a.port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("url")
    p.add_argument("--no-rate", action="store_true")
    p.add_argument("--no-photos", action="store_true")
    p.set_defaults(fn=cmd_add)
    for name, fn in (("show", cmd_show), ("rate", cmd_rate), ("remove", cmd_remove)):
        p = sub.add_parser(name)
        p.add_argument("target")
        if name == "remove":
            p.add_argument("--purge", action="store_true", help="also delete its photos and cached page")
        p.set_defaults(fn=fn)
    p = sub.add_parser("set-rating", help="save a rating made by Claude Code (no API key)")
    p.add_argument("target")
    p.add_argument("--red-rock", type=int, choices=range(4), required=True)
    p.add_argument("--view", type=int, choices=range(6), required=True)
    p.add_argument("--forest", type=int, choices=(0, 1), default=0)
    p.add_argument("--pool", type=int, choices=(0, 1), default=0)
    p.add_argument("--renovated", type=int, choices=(0, 1), default=0)
    p.add_argument("--evidence", default="")
    p.add_argument("--confidence", choices=("high", "low"), default="high")
    p.add_argument("--photos-used", type=int, default=7)
    p.add_argument("--by", default="Claude Code (photo review)")
    p.set_defaults(fn=cmd_set_rating)
    p = sub.add_parser("serve", help="serve the map on localhost with add/remove buttons")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
