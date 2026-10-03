#!/usr/bin/env python3
"""Score every sale: how underpriced was it vs. comps, and how good are its view/land.

  python3 analyze.py --houses data/houses.csv --out data/deals_2yr.csv
  python3 analyze.py --houses data/houses_5yr.csv --out data/deals_5yr.csv

Inputs (optional ones are used when present):
  data/page_facts.jsonl   from enrich_from_pages.py (fills sold date, beds, adds APN, list price)
  data/features.jsonl     view/land features extracted from descriptions (see FEATURE KEYS below)

Model: OLS on ln(sold_price) with ln(sqft), ln(lot), age, area, a time trend, and view features.
Residual = how far the price sits below/above what comps predict. Pure stdlib, no numpy.

FEATURE KEYS in features.jsonl (one JSON object per redfin_id):
  view_score 0-5 (null if the text says nothing), red_rock_strict 0/1 (strict, description-based; the looser
  red_rock_view is kept in the file but not used here), forest_backing 0/1, pool 0/1, renovated 0/1,
  confidence "high"|"low", exclude_reason (non-empty = left out of the model and ranking).
  Display only: guest_unit, needs_work, spa, str_note.
"""
import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path

# Deal score weights: view/land quality is weighted heavily; red-rock views get a modest bonus.
W_UNDERPRICED = 0.45
W_QUALITY = 0.45
W_REDROCK = 0.10
# Quality index = view score (most), land size, backing onto forest.
Q_VIEW, Q_LOT, Q_FOREST = 0.55, 0.30, 0.15

T0 = date(2021, 10, 1)


def load_jsonl(path, key="redfin_id"):
    p = Path(path)
    if not p.exists():
        return {}
    return {r[key]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def solve_ols(X, y, ridge=1e-8):
    """Return (beta, cov_unscaled). Normal equations with Gauss-Jordan inversion."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (ridge if i == j else 0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * yi for r, yi in zip(X, y)) for i in range(k)]
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(k):
            if r != c:
                f = M[r][c]
                M[r] = [v - f * w for v, w in zip(M[r], M[c])]
    inv = [row[k:] for row in M]
    beta = [sum(inv[i][j] * b[j] for j in range(k)) for i in range(k)]
    return beta, inv


def pct_rank(values):
    """Percentile rank 0-1 for each value (ties share the average rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / max(len(values) - 1, 1)
        i = j + 1
    return ranks


def build_rows(houses_path):
    facts = load_jsonl("data/page_facts.jsonl")
    feats = load_jsonl("data/features.jsonl")
    rows = []
    for h in csv.DictReader(open(houses_path, newline="", encoding="utf-8")):
        f, ft = facts.get(h["redfin_id"], {}), feats.get(h["redfin_id"], {})
        sold = h["sold_date"] or f.get("page_sold_date", "")
        r = dict(h)
        r["sold_date"] = sold
        r["sold_date_source"] = "csv" if h["sold_date"] else ("page" if sold else "")
        if not h["beds"] and f.get("page_beds"):
            r["beds"] = str(f["page_beds"])
        r["apn"] = f.get("apn", "")
        r["list_price"] = f.get("list_price") or ""
        r["off_market_sale"] = f.get("off_market_sale", "")
        r["_feat"] = ft
        r["_exclude"] = ft.get("exclude_reason", "")
        rows.append(r)
    return rows, bool(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", default="data/houses.csv")
    ap.add_argument("--out", default="data/deals.csv")
    ap.add_argument("--summary", help="model summary text file (default: next to --out)")
    ap.add_argument("--red-rock-field", default="red_rock_strict",
                    help="feature used for the red-rock term and bonus: red_rock_strict (0/1, default) or red_rock_rating (0-3)")
    args = ap.parse_args()
    RR = args.red_rock_field
    rr_max = 3.0 if RR == "red_rock_rating" else 1.0

    rows, have_feats = build_rows(args.houses)
    excluded = [r for r in rows if r["_exclude"]]
    rows = [r for r in rows if not r["_exclude"]]

    # ---- feature matrix ---------------------------------------------------
    def base_vec(r):
        sqft, lot, yb, price = fnum(r["sqft"]), fnum(r["lot_sqft"]), fnum(r["year_built"]), fnum(r["sold_price"])
        if not (sqft and lot and yb and price) or not r["sold_date"]:
            return None
        sold = datetime.strptime(r["sold_date"], "%Y-%m-%d").date()
        return [1.0, math.log(sqft), math.log(lot), max(0.0, sold.year - yb) / 10, 1.0 if "chapel" in r["area"] else 0.0,
                (sold - T0).days / 365.25]

    names = ["const", "ln_sqft", "ln_lot", "age_decades", "chapel", "years_since_2021-10"]
    view_cols = []
    if have_feats:
        for k in ("view_score", RR, "forest_backing", "pool", "renovated"):
            got = [r["_feat"].get(k) for r in rows if r["_feat"].get(k) is not None]
            if len(got) >= 0.5 * len(rows) and len(set(got)) > 1:  # enough coverage to be worth fitting
                mean = sum(got) / len(got)
                view_cols.append((k, mean))
        names += [k for k, _ in view_cols]

    def vec(r):
        b = base_vec(r)
        if b is None:
            return None
        return b + [float(r["_feat"].get(k) if r["_feat"].get(k) is not None else m) for k, m in view_cols]

    fit = [(r, vec(r)) for r in rows]
    fit = [(r, v) for r, v in fit if v]
    X = [v for _, v in fit]
    y = [math.log(float(r["sold_price"])) for r, _ in fit]
    n, k = len(X), len(names)
    beta, inv = solve_ols(X, y)
    pred = [sum(b * x for b, x in zip(beta, v)) for v in X]
    sse = sum((yi - p) ** 2 for yi, p in zip(y, pred))
    sst = sum((yi - sum(y) / n) ** 2 for yi in y)
    sigma = math.sqrt(sse / (n - k))
    r2 = 1 - sse / sst
    se = [sigma * math.sqrt(inv[i][i]) for i in range(k)]

    # ---- scoring ------------------------------------------------------------
    scored = []
    for (r, v), yi, p in zip(fit, y, pred):
        ft = r["_feat"]
        resid = yi - p
        scored.append({
            **{c: r[c] for c in r if not c.startswith("_")},
            "predicted_price": round(math.exp(p)),
            "discount_pct": round(100 * (1 - math.exp(resid)), 1),  # + = sold below comps
            "resid_z": round(resid / sigma, 2),
            "view_score": ft.get("view_score", ""),
            "red_rock_strict": ft.get("red_rock_strict", ""),
            "red_rock_rating": ft.get("red_rock_rating", ""),
            "red_rock_rating_source": ft.get("red_rock_rating_source", ""),
            "forest_backing": ft.get("forest_backing", ""),
            "pool": ft.get("pool", ""),
            "guest_unit": ft.get("guest_unit", ""),
            "renovated": ft.get("renovated", ""),
            "needs_work": ft.get("needs_work", ""),
            "features_confidence": ft.get("confidence", ""),
            "str_note": ft.get("str_note", ""),
            "_resid": resid, "_lot": v[2], "_ft": ft,
        })
    under = pct_rank([-s["_resid"] for s in scored])          # higher = cheaper than comps
    lot_pct = pct_rank([s["_lot"] for s in scored])
    for s, u, lp in zip(scored, under, lot_pct):
        ft = s["_ft"]
        vs = ft.get("view_score")
        view = (vs / 5) if vs is not None else None
        quality = (Q_VIEW * (view if view is not None else 0.5) + Q_LOT * lp + Q_FOREST * float(ft.get("forest_backing") or 0))
        s["_quality"] = quality
        s["_under"] = u
        s["_rr"] = float(ft.get(RR) or 0) / rr_max
        s["view_known"] = view is not None
    qpct = pct_rank([s["_quality"] for s in scored])
    for s, q in zip(scored, qpct):
        s["quality_pct"] = round(100 * q)
        s["deal_score"] = round(100 * (W_UNDERPRICED * s["_under"] + W_QUALITY * q + W_REDROCK * s["_rr"]), 1)
    scored.sort(key=lambda s: -s["deal_score"])

    out_cols = ["deal_score", "address_link", "address", "area", "sold_price", "predicted_price", "discount_pct",
                "resid_z", "quality_pct", "view_score", "red_rock_strict", "red_rock_rating", "red_rock_rating_source", "forest_backing", "pool", "guest_unit", "renovated",
                "needs_work", "features_confidence", "str_note", "view_known", "sold_date",
                "sold_date_source", "sqft", "lot_sqft", "beds", "baths", "year_built", "hoa_month", "list_price",
                "off_market_sale", "apn", "mls_id", "redfin_id", "redfin_url", "flags"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    unscored = [r for r in rows if vec(r) is None]
    if excluded:
        with open(str(Path(args.out).with_suffix("")) + "_excluded.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["address", "sold_price", "sold_date", "redfin_url", "reason"])
            for r in excluded:
                w.writerow([r["address"], r["sold_price"], r["sold_date"], r["redfin_url"], r["_exclude"]])
    lines = [f"model: ln(sold_price) ~ {' + '.join(names[1:])}", f"houses: {len(rows)}  fitted/scored: {n}  unscored: {len(unscored)}  excluded: {len(excluded)}",
             f"R^2 = {r2:.3f}   residual sd = {sigma:.3f} (about {100 * (math.exp(sigma) - 1):.0f}% price noise)", "",
             f"{'term':22s}{'coef':>9s}{'se':>8s}{'t':>7s}"]
    lines += [f"{nm:22s}{b:9.3f}{s_:8.3f}{b / s_:7.1f}" for nm, b, s_ in zip(names, beta, se)]
    why = {}
    for r in unscored:
        miss = [c for c in ("sold_date", "sqft", "lot_sqft", "year_built", "sold_price") if not r[c]]
        why[",".join(miss) or "other"] = why.get(",".join(miss) or "other", 0) + 1
    lines += ["", f"unscored reasons (missing fields): {why}",
              f"view features used: {[k for k, _ in view_cols] or 'none (no features.jsonl or too sparse)'}"]
    text = "\n".join(lines)
    Path(args.summary or str(Path(args.out).with_suffix(".model.txt"))).write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out} ({len(scored)} rows)")


if __name__ == "__main__":
    main()
