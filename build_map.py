#!/usr/bin/env python3
"""Build map.html: Leaflet map of sales since 2024-10 (houses_5yr.csv) with pool / red-rock rating (features.jsonl),
the photo-based deal rank (deals_2yr_photo.csv), the grid-model formula price and gap (grid_predictions.csv)
and the 1 km grid cells with their adjusted price level (grid_comps.csv)."""
import base64, csv, json, math, subprocess
from pathlib import Path

import favorites as fav

CELL_KM = 2.0           # must match the size passed to grid_two_stage.py
KM_PER_DEG = 111.2

feats = {}
for l in open("data/features.jsonl"):
    if l.strip():
        r = json.loads(l); feats[r["redfin_id"]] = r
rank = {}
for i, r in enumerate(csv.DictReader(open("data/deals_2yr_photo.csv")), 1):
    rank[r["redfin_id"]] = (i, float(r["discount_pct"]))
grid = {r["redfin_id"]: r for r in csv.DictReader(open("data/grid_predictions.csv"))}

all5 = [x for x in csv.DictReader(open("data/houses_5yr.csv")) if x["lat"]]
lat0 = min(float(x["lat"]) for x in all5)
lng0 = min(float(x["lng"]) for x in all5)
cl = math.cos(math.radians(sum(float(x["lat"]) for x in all5) / len(all5)))

THUMBS = Path("data/map_thumbs")      # 320 px copies of each home's lead photo, made once with sips
THUMBS.mkdir(exist_ok=True)


def thumb(rid):
    out = THUMBS / f"{rid}.jpg"
    if not out.exists():
        for src in (Path(f"data/images/{rid}/000.jpg"), Path(f"data/images_512/{rid}/000.jpg")):
            if src.exists():
                subprocess.run(["sips", "-Z", "320", "-s", "formatOptions", "55", str(src), "--out", str(out)], capture_output=True)
                break
    return "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode() if out.exists() else None


rows = []
for x in all5:
    if not x["sold_date"] or x["sold_date"] < "2024-10-01":
        continue
    f = feats.get(x["redfin_id"], {})
    rk = rank.get(x["redfin_id"])
    g = grid.get(x["redfin_id"])
    rows.append({
        "img": thumb(x["redfin_id"]),
        "a": x["address"], "p": int(float(x["sold_price"])), "d": x["sold_date"],
        "b": int(float(x["beds"])) if x["beds"] else None,
        "ba": float(x["baths"]) if x["baths"] else None,
        "s": int(float(x["sqft"])) if x["sqft"] else None,
        "l": int(float(x["lot_sqft"])) if x["lot_sqft"] else None,
        "lat": float(x["lat"]), "lng": float(x["lng"]), "u": x["redfin_url"],
        "pool": f.get("pool"), "rr": f.get("red_rock_rating"),
        "rk": rk[0] if rk else None,                              # deal rank (photo-based)
        "fp": int(g["predicted_price_out_of_fold"]) if g else None,  # formula price, never saw this sale
        "fg": float(g["pct_below_prediction"]) if g else None,       # % the sale was below the formula price
    })

cells = []
for c in csv.DictReader(open("data/grid_comps.csv")):
    cx, cy = int(c["cell_x"]), int(c["cell_y"])
    s0 = lat0 + cy * CELL_KM / KM_PER_DEG
    s1 = lat0 + (cy + 1) * CELL_KM / KM_PER_DEG
    w0 = lng0 + cx * CELL_KM / (KM_PER_DEG * cl)
    w1 = lng0 + (cx + 1) * CELL_KM / (KM_PER_DEG * cl)
    cells.append({"b": [[s0, w0], [s1, w1]], "n": int(c["n_sales_5yr"]), "e": float(c["adjusted_cell_effect_pct"]),
                  "pp": int(c["median_price_per_sqft"]), "ex": c["example_addresses"]})

favs = []
for f in fav.load_favs().values():
    if not (f.get("lat") and f.get("lng")):
        continue
    v = fav.valuation(f)
    r = f.get("rating") or {}
    sold = f.get("status") == "sold"
    verdict = v.get("verdict")
    vc = "none" if (sold or not verdict) else ("good" if verdict.startswith("GOOD") else "bad" if verdict.startswith("BAD") else "fair")
    img = next((p for p in f.get("photos", []) if Path(p).exists()), None)
    favs.append({
        "id": f["redfin_id"], "a": f.get("address"), "u": f["url"], "lat": f["lat"], "lng": f["lng"], "status": f.get("status"),
        "price": f.get("sold_price") if sold else f.get("ask_price"), "sd": f.get("sold_date"),
        "b": f.get("beds"), "ba": f.get("baths"), "s": f.get("sqft"), "l": f.get("lot_sqft"), "yb": f.get("year_built"),
        "rr": r.get("red_rock_rating") if r.get("status") == "ok" else None, "conf": r.get("confidence"),
        "rng": v["range"], "mid": v.get("mid"), "why": v.get("why_not"), "pct": v.get("pct_vs_mid"),
        "verdict": verdict, "vc": vc, "prov": bool(v.get("provisional")),
        "img": "data:image/jpeg;base64," + base64.b64encode(Path(img).read_bytes()).decode() if img else None,
    })

tpl = open("map_template.html").read()
out = tpl.replace("__DATA__", json.dumps(rows, separators=(",", ":"))).replace("__FAVS__", json.dumps(favs, separators=(",", ":"))).replace("__CELLS__", json.dumps(cells, separators=(",", ":")))
open("map.html", "w").write(out)
print(sum(1 for r in rows if r["img"]), "with a photo;", len(favs), "favorites,", len(rows), "homes,", sum(1 for r in rows if r["rk"]), "ranked,", sum(1 for r in rows if r["fp"]), "with formula price,", len(cells), "cells")
