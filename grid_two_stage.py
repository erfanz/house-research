#!/usr/bin/env python3
"""Two-stage grid model: neighbourhood price level from the 5-year sales, features from the 2-year sales.

Why: the 2-year set has ~4 sales per cell, too few for a trustworthy cell level. The 5-year set has 502 sales with sqft, lot,
age and sale date, so a cell's level can be estimated from about 2.4x as many sales (adjusted for size, lot, age and date).

Stage 1 (5-yr sales):   ln(price) = const + ln_sqft + ln_lot + age + t + cell effect (ridge-shrunk)  -> cell effect e_c
Stage 2 (2-yr sales):   ln(price) = const + [full features] + b * e_c

No leakage: a home's own sale (and, for held-out folds, the whole fold) is excluded from the stage-1 fit that supplies its e_c.
 - training homes get cross-fitted e_c (stage 1 refit without each of 5 random groups of them)
 - held-out homes get e_c from a stage 1 fit that never saw them
Everything is scored on held-out folds. Cell size and shrinkage are fixed per run (try several; the table shows all of them).

Usage: python3 grid_two_stage.py
"""
import math
import random

import sys

import grid_model as g

FORCE_SIZE = float(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else None

S1_FEATS = g.FEATURE_SETS["structure"]
FULL = g.FEATURE_SETS["full"]


def stage1(rows5, exclude, size, lam):
    """Cell effects from the 5-yr sales not in `exclude`. Returns {cell: effect}."""
    tr = [r for r in rows5 if r["id"] not in exclude]
    m = g.feature_means(tr, S1_FEATS)
    cells = g.cell_ids(tr, size)
    idx = {c: i for i, c in enumerate(sorted(set(cells)))}
    X = g.design(tr, S1_FEATS, m)
    beta, _, _ = g.fit_ridge(X, cells, [r["y"] for r in tr], lam, idx)
    p = len(X[0])
    return {c: beta[p + i] for c, i in idx.items()}


def ols(X, y, ridge=0.0):
    k = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (ridge if (i == j and i > 0) else 0.0) for j in range(k)] for i in range(k)]
    b = [sum(x[i] * yi for x, yi in zip(X, y)) for i in range(k)]
    inv = g.solve_inv(A)
    return [sum(inv[i][j] * b[j] for j in range(k)) for i in range(k)], inv


def design2(rows, means, eff_by_id, with_cell=True):
    return [[1.0] + [(r[f] if r[f] is not None else means[f]) for f in FULL] + ([eff_by_id[r["id"]]] if with_cell else [])
            for r in rows]


def run(rows2, rows5, size, lam, folds=10, seed=7, with_cell=True):
    n = len(rows2)
    rnd = random.Random(seed)
    order = list(range(n))
    rnd.shuffle(order)
    parts = [order[i::folds] for i in range(folds)]
    errs = {}
    for f in parts:
        test = [rows2[i] for i in f]
        test_ids = {r["id"] for r in test}
        train = [r for r in rows2 if r["id"] not in test_ids]
        means = g.feature_means(train, FULL)
        eff = {}
        if with_cell:
            # test homes: stage-1 never saw them (nor any other test home)
            e_test = stage1(rows5, test_ids, size, lam)
            for r in test:
                eff[r["id"]] = e_test.get(g.cell_ids([r], size)[0], 0.0)
            # train homes: cross-fit in 5 groups
            tids = [r["id"] for r in train]
            rnd2 = random.Random(seed + 1)
            rnd2.shuffle(tids)
            for k in range(5):
                grp = set(tids[k::5])
                e = stage1(rows5, test_ids | grp, size, lam)
                for r in train:
                    if r["id"] in grp:
                        eff[r["id"]] = e.get(g.cell_ids([r], size)[0], 0.0)
        else:
            for r in rows2:
                eff[r["id"]] = 0.0
        Xtr = design2(train, means, eff, with_cell)
        beta, _ = ols(Xtr, [r["y"] for r in train])
        for r, x in zip(test, design2(test, means, eff, with_cell)):
            errs[r["id"]] = r["y"] - sum(a * b for a, b in zip(x, beta))
    return errs


def main():
    rows2, rows5 = g.load("2yr"), g.load("5yr")
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say(f"2-yr homes {len(rows2)}, 5-yr homes {len(rows5)} (the 2-yr set is a subset)")
    base = run(rows2, rows5, 1.0, 3, with_cell=False)
    say(g.fmt("full features, no location (10-fold)", g.stats(list(base.values()))))
    say("")
    say("Two-stage: full features + cell level estimated from the 5-yr sales (10-fold, no leakage)")
    best = None
    for size in [0.5, 0.75, 1.0, 1.5, 2.0]:
        for lam in [1, 3, 10]:
            e = run(rows2, rows5, size, lam)
            s = g.stats(list(e.values()))
            say(g.fmt(f"  cell {size} km, lambda {lam}", s))
            if best is None or s["rmse_ln"] < best[0]:
                best = (s["rmse_ln"], size, lam)
    say(f"\nbest of the 15 configurations (picked on the same held-out folds, so a little optimistic): "
        f"{best[1]} km, lambda {best[2]}, rmse {best[0]:.4f} vs {g.stats(list(base.values()))['rmse_ln']:.4f} without location")

    # final fit: cell levels from all 5-yr sales, cross-fitted for the 2-yr homes
    size, lam = best[1], best[2]
    if FORCE_SIZE:   # user-chosen cell size: take the best shrinkage for that size
        cand = [(g.stats(list(run(rows2, rows5, FORCE_SIZE, l).values()))["rmse_ln"], l) for l in (1, 3, 10)]
        size, lam = FORCE_SIZE, min(cand)[1]
        say(f"\nCell size fixed at {size} km by choice; best shrinkage for it: lambda {lam}")
    eff = {}
    ids = [r["id"] for r in rows2]
    rnd = random.Random(11)
    rnd.shuffle(ids)
    for k in range(5):
        grp = set(ids[k::5])
        e = stage1(rows5, grp, size, lam)
        for r in rows2:
            if r["id"] in grp:
                eff[r["id"]] = e.get(g.cell_ids([r], size)[0], 0.0)
    means = g.feature_means(rows2, FULL)
    X = design2(rows2, means, eff)
    beta, inv = ols(X, [r["y"] for r in rows2])
    resid = [r["y"] - sum(a * b for a, b in zip(x, beta)) for r, x in zip(rows2, X)]
    k = len(beta)
    sigma = math.sqrt(sum(e * e for e in resid) / (len(rows2) - k))
    say(f"\nFinal stage-2 formula on all 206 homes (cell {size} km, lambda {lam}); resid sd {sigma:.3f}")
    for nme, b, i in zip(["const"] + FULL + ["cell_level (from 5-yr)"], beta, range(k)):
        say(f"  {nme:<24} {b:+.4f}  (t {b / (sigma * math.sqrt(inv[i][i])):.1f})")
    # per-home out-of-fold predictions at the chosen configuration (each home predicted by a model that never saw it)
    oof = run(rows2, rows5, size, lam)
    import csv
    with open("data/grid_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["redfin_id", "address", "sold_date", "sold_price", "cell_x", "cell_y", "cell_n_5yr_sales",
                    "predicted_price_out_of_fold", "pct_below_prediction", "abs_error_pct"])
        cell_n = {}
        for r in rows5:
            c = g.cell_ids([r], size)[0]
            cell_n[c] = cell_n.get(c, 0) + 1
        for r in sorted(rows2, key=lambda r: oof[r["id"]]):
            pred = math.exp(r["y"] - oof[r["id"]])
            c = g.cell_ids([r], size)[0]
            w.writerow([r["id"], r["address"], r["sold"], round(r["sold_price"]), c[0], c[1], cell_n.get(c, 0), round(pred),
                        f"{100 * (1 - r['sold_price'] / pred):+.1f}", f"{100 * abs(r['sold_price'] / pred - 1):.1f}"])
    # per-cell comps from all 5-yr sales
    m5 = g.feature_means(rows5, S1_FEATS)
    e_all = stage1(rows5, set(), size, lam)
    by = {}
    for r in rows5:
        by.setdefault(g.cell_ids([r], size)[0], []).append(r)
    with open("data/grid_comps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_x", "cell_y", "centre_lat", "centre_lng", "n_sales_5yr", "n_sales_2yr", "median_price",
                    "median_price_per_sqft", "median_sqft", "median_lot_sqft", "adjusted_cell_effect_pct", "example_addresses"])
        ids2 = {r["id"] for r in rows2}
        for c, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            pr = sorted(r["sold_price"] for r in rs)
            pp = sorted(r["sold_price"] / r["sqft"] for r in rs)
            sq = sorted(r["sqft"] for r in rs)
            lt = sorted(r["lot"] for r in rs)
            n = len(rs)
            w.writerow([c[0], c[1], f"{sum(r['lat'] for r in rs) / n:.5f}", f"{sum(r['lng'] for r in rs) / n:.5f}", n,
                        sum(r["id"] in ids2 for r in rs), round(pr[n // 2]), round(pp[n // 2]), round(sq[n // 2]),
                        round(lt[n // 2]), f"{100 * (math.exp(e_all[c]) - 1):+.1f}",
                        "; ".join(r["address"] for r in rs[:3])])
    open("data/grid_two_stage.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
