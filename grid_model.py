#!/usr/bin/env python3
"""Grid-cell comps + price formula (pure stdlib; no numpy installed, do not pip install).

1. Project lat/lng to km and cut the study area into square cells of a chosen size.
2. Fit  ln(price) = covariates + cell effect  with the cell effects ridge-shrunk toward 0, so a cell with few sales borrows
   from the global level instead of trusting 2-3 sales.
3. Pick the cell size and shrinkage by leave-one-out error (hat-matrix shortcut) and report the error of that choice on
   held-out folds (nested: the choice is made inside each training fold), against simpler baselines.

Outputs:  data/grid_comps.csv (per cell), data/grid_predictions.csv (per home), data/grid_cv.txt (full CV table).
Usage:    python3 grid_model.py [--set 2yr|5yr]
"""
import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime, date

T0 = date(2021, 10, 1)
SIZES = [0.5, 0.75, 1.0, 1.5, 2.0]          # km
LAMBDAS = [0.3, 1, 3, 10, 30]               # ridge penalty on cell effects (in units of "number of sales")
KM_PER_DEG = 111.2


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def solve_inv(A):
    """Invert a small symmetric positive-definite matrix by Gauss-Jordan."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def load(set_name):
    deals = "data/deals_2yr_photo.csv" if set_name == "2yr" else "data/deals_5yr.csv"
    houses = "data/houses.csv" if set_name == "2yr" else "data/houses_5yr.csv"
    ll = {r["redfin_id"]: (fnum(r["lat"]), fnum(r["lng"])) for r in csv.DictReader(open(houses))}
    rows = []
    for r in csv.DictReader(open(deals)):
        lat, lng = ll.get(r["redfin_id"], (None, None))
        sqft, lot, yb, price = fnum(r["sqft"]), fnum(r["lot_sqft"]), fnum(r["year_built"]), fnum(r["sold_price"])
        if None in (lat, lng, sqft, lot, yb, price) or not r["sold_date"]:
            continue
        sold = datetime.strptime(r["sold_date"], "%Y-%m-%d").date()
        rows.append({
            "id": r["redfin_id"], "address": r["address"], "area": r["area"], "lat": lat, "lng": lng,
            "sold_price": price, "sqft": sqft, "lot": lot, "sold": r["sold_date"],
            "ln_sqft": math.log(sqft), "ln_lot": math.log(lot), "age": max(0.0, sold.year - yb) / 10,
            "t": (sold - T0).days / 365.25,
            "view_score": fnum(r.get("view_score")), "red_rock": fnum(r.get("red_rock_rating")),
            "forest": fnum(r.get("forest_backing")), "pool": fnum(r.get("pool")), "renovated": fnum(r.get("renovated")),
            "y": math.log(price),
        })
    # One fixed grid origin for every set (the 5-yr set's corner), so a cell id means the same place in both sets.
    hall = [(fnum(r["lat"]), fnum(r["lng"])) for r in csv.DictReader(open("data/houses_5yr.csv")) if fnum(r["lat"])]
    lat0 = min(a for a, _ in hall)
    lng0 = min(b for _, b in hall)
    cl = math.cos(math.radians(sum(a for a, _ in hall) / len(hall)))
    for r in rows:
        r["x_km"] = (r["lng"] - lng0) * KM_PER_DEG * cl
        r["y_km"] = (r["lat"] - lat0) * KM_PER_DEG
    return rows


FEATURE_SETS = {
    # what you listed: size, lot, view, pool
    "core": ["ln_sqft", "ln_lot", "view_score", "red_rock", "pool"],
    # plus the other things the market clearly pays for
    "full": ["ln_sqft", "ln_lot", "age", "t", "view_score", "red_rock", "forest", "pool", "renovated"],
    "structure": ["ln_sqft", "ln_lot", "age", "t"],   # for the 5-yr set, which has no view/pool labels
}


def design(rows, feats, means):
    X = []
    for r in rows:
        X.append([1.0] + [(r[f] if r[f] is not None else means[f]) for f in feats])
    return X


def cell_ids(rows, size, off=(0.0, 0.0)):
    return [(int((r["x_km"] + off[0]) // size), int((r["y_km"] + off[1]) // size)) for r in rows]


def fit_ridge(X, cells, y, lam, cell_index):
    """Columns: covariates (unpenalised) then one dummy per cell (ridge lam). Returns beta, inverse, hat diagonal."""
    p = len(X[0])
    k = len(cell_index)
    D = p + k
    rowsX = []
    for xi, c in zip(X, cells):
        v = xi[:] + [0.0] * k
        v[p + cell_index[c]] = 1.0
        rowsX.append(v)
    A = [[0.0] * D for _ in range(D)]
    b = [0.0] * D
    for v, yi in zip(rowsX, y):
        nz = [i for i, a in enumerate(v) if a != 0.0]
        for i in nz:
            b[i] += v[i] * yi
            Ai = A[i]
            for j in nz:
                Ai[j] += v[i] * v[j]
    for j in range(p, D):
        A[j][j] += lam
    inv = solve_inv(A)
    beta = [sum(inv[i][j] * b[j] for j in range(D)) for i in range(D)]
    hat = []
    for v in rowsX:
        nz = [i for i, a in enumerate(v) if a != 0.0]
        hat.append(sum(v[i] * inv[i][j] * v[j] for i in nz for j in nz))
    return beta, rowsX, hat


def loo_errors(rows, feats, size, lam, means, off=(0.0, 0.0)):
    cells = cell_ids(rows, size, off)
    idx = {c: i for i, c in enumerate(sorted(set(cells)))}
    X = design(rows, feats, means)
    y = [r["y"] for r in rows]
    beta, rowsX, hat = fit_ridge(X, cells, y, lam, idx)
    out = []
    for v, yi, h in zip(rowsX, y, hat):
        pred = sum(a * b for a, b in zip(v, beta))
        out.append((yi - pred) / (1 - h))
    return out


def predict_new(train, test, feats, size, lam, means, off=(0.0, 0.0)):
    cells = cell_ids(train, size, off)
    idx = {c: i for i, c in enumerate(sorted(set(cells)))}
    X = design(train, feats, means)
    beta, _, _ = fit_ridge(X, cells, [r["y"] for r in train], lam, idx)
    p = len(X[0])
    tc = cell_ids(test, size, off)
    Xt = design(test, feats, means)
    preds = []
    for xi, c in zip(Xt, tc):
        s = sum(a * b for a, b in zip(xi, beta[:p]))
        if c in idx:
            s += beta[p + idx[c]]
        preds.append(s)
    return preds


def feature_means(rows, feats):
    m = {}
    for f in feats:
        v = [r[f] for r in rows if r[f] is not None]
        m[f] = sum(v) / len(v) if v else 0.0
    return m


def stats(errs):
    n = len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    ape = sorted(abs(math.exp(e) - 1) if e < 0 else abs(1 - math.exp(-e)) for e in errs)  # relative miss vs actual price
    med = ape[n // 2]
    return {"n": n, "rmse_ln": rmse, "median_abs_pct": 100 * med, "mean_abs_pct": 100 * sum(ape) / n,
            "within10": 100 * sum(a <= 0.10 for a in ape) / n, "within20": 100 * sum(a <= 0.20 for a in ape) / n}


def fmt(name, s):
    return (f"{name:<44} rmse(ln)={s['rmse_ln']:.3f}  median miss={s['median_abs_pct']:.1f}%  mean miss={s['mean_abs_pct']:.1f}%  "
            f"within10%={s['within10']:.0f}%  within20%={s['within20']:.0f}%")


def best_config(rows, feats, means):
    best = None
    table = []
    for size in SIZES:
        for lam in LAMBDAS:
            e = loo_errors(rows, feats, size, lam, means)
            sse = sum(v * v for v in e)
            table.append((size, lam, math.sqrt(sse / len(e))))
            if best is None or sse < best[0]:
                best = (sse, size, lam)
    return best[1], best[2], table


def naive_comp_errors(rows, size):
    """Cell median $/sqft x sqft, leave-one-out within cell (global median if the cell has no other sale)."""
    cells = cell_ids(rows, size)
    by = defaultdict(list)
    for i, (r, c) in enumerate(zip(rows, cells)):
        by[c].append((i, r["sold_price"] / r["sqft"]))
    allp = sorted(r["sold_price"] / r["sqft"] for r in rows)
    out = []
    for i, (r, c) in enumerate(zip(rows, cells)):
        others = sorted(p for j, p in by[c] if j != i)
        if len(others) >= 3:
            m = others[len(others) // 2]
        else:
            m = allp[len(allp) // 2]
        out.append(r["y"] - math.log(m * r["sqft"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="2yr", choices=["2yr", "5yr"])
    ap.add_argument("--features", default=None, help="core | full | structure (default: full for 2yr, structure for 5yr)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    fs = args.features or ("full" if args.set == "2yr" else "structure")
    feats = FEATURE_SETS[fs]
    tag = args.tag or f"{args.set}_{fs}"
    rows = load(args.set)
    n = len(rows)
    means = feature_means(rows, feats)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say(f"set={args.set} features={fs} {feats}  homes={n}")
    say(f"extent: {max(r['x_km'] for r in rows):.1f} km east-west x {max(r['y_km'] for r in rows):.1f} km north-south")

    # --- 1. choose size / shrinkage on all data by LOO -------------------------------------------------------
    size, lam, table = best_config(rows, feats, means)
    say("\nLeave-one-out rmse(ln price) by cell size (km, rows) and shrinkage lambda (columns):")
    say("size   " + "  ".join(f"lam={l:<5}" for l in LAMBDAS))
    for s in SIZES:
        say(f"{s:<6} " + "  ".join(f"{[t for t in table if t[0]==s and t[1]==l][0][2]:.4f}   " for l in LAMBDAS))
    say(f"chosen: cell size {size} km, lambda {lam}")

    # no-location reference with the same covariates
    base_err = loo_errors(rows, feats, 1000.0, 1e9, means)   # one giant cell, effectively no location
    # --- 2. nested CV: pick config on each training fold, score on the held-out fold ---------------------------
    rnd = random.Random(7)
    order = list(range(n))
    rnd.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    nested, global_only, chapel_only = [], [], []
    for f in folds:
        fs_set = set(f)
        train = [rows[i] for i in range(n) if i not in fs_set]
        test = [rows[i] for i in f]
        m = feature_means(train, feats)
        s_, l_, _ = best_config(train, feats, m)
        pr = predict_new(train, test, feats, s_, l_, m)
        nested += [t["y"] - p for t, p in zip(test, pr)]
        pg = predict_new(train, test, feats, 1000.0, 1e9, m)
        global_only += [t["y"] - p for t, p in zip(test, pg)]
    say("\nHeld-out error (10-fold; cell size and shrinkage re-chosen inside each training fold):")
    say(fmt("covariates only (no location)", stats(global_only)))
    say(fmt("covariates + grid cell effects (chosen)", stats(nested)))
    say(fmt("LOO, covariates only (all data)", stats(base_err)))
    say(fmt(f"LOO, covariates + grid {size} km, lam {lam}", stats(loo_errors(rows, feats, size, lam, means))))
    say(fmt(f"naive comps: cell median $/sqft x sqft ({size} km)", stats(naive_comp_errors(rows, size))))
    allp = sorted(r["sold_price"] / r["sqft"] for r in rows)
    mp = allp[len(allp) // 2]
    say(fmt("naive: one citywide median $/sqft x sqft", stats([r["y"] - math.log(mp * r["sqft"]) for r in rows])))

    # --- 3. sensitivity of the chosen size to where the grid starts -----------------------------------------
    say("\nSensitivity to grid origin (LOO rmse, chosen size and lambda):")
    for off in [(0, 0), (size / 2, 0), (0, size / 2), (size / 2, size / 2)]:
        e = loo_errors(rows, feats, size, lam, means, off)
        say(f"  shift {off[0]:.2f},{off[1]:.2f} km: rmse {math.sqrt(sum(v*v for v in e)/n):.4f}")

    # --- 4. final fit on all data ---------------------------------------------------------------------------
    cells = cell_ids(rows, size)
    idx = {c: i for i, c in enumerate(sorted(set(cells)))}
    X = design(rows, feats, means)
    beta, rowsX, hat = fit_ridge(X, cells, [r["y"] for r in rows], lam, idx)
    p = len(X[0])
    names = ["const"] + feats
    say("\nFinal formula  ln(price) = const + sum(coef x feature) + cell_effect")
    for nme, b in zip(names, beta[:p]):
        say(f"  {nme:<10} {b:+.4f}")
    def pct(f):
        return 100 * (math.exp(beta[1 + feats.index(f)]) - 1) if f in feats else float("nan")
    say("  Reading: +1%% sqft -> %+.2f%% price; +1%% lot -> %+.2f%%; pool -> %+.0f%%; each red-rock point -> %+.1f%%" % (
        beta[1 + feats.index("ln_sqft")], beta[1 + feats.index("ln_lot")], pct("pool"), pct("red_rock")))
    if "view_score" in feats:
        say("  each view-score point -> %+.1f%%" % (100 * (math.exp(beta[1 + feats.index("view_score")]) - 1)))

    # --- 5. per-cell comps + per-home predictions -------------------------------------------------------------
    by = defaultdict(list)
    for i, (r, c) in enumerate(zip(rows, cells)):
        by[c].append(i)
    loo = loo_errors(rows, feats, size, lam, means)
    with open(f"data/grid_comps_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_x", "cell_y", "centre_lat", "centre_lng", "n_sales", "median_price", "median_price_per_sqft",
                    "median_lot_sqft", "cell_effect_pct", "mean_resid_pct", "streets_sample"])
        for c in sorted(by, key=lambda c: -len(by[c])):
            ii = by[c]
            pr = sorted(rows[i]["sold_price"] for i in ii)
            pp = sorted(rows[i]["sold_price"] / rows[i]["sqft"] for i in ii)
            lots = sorted(rows[i]["lot"] for i in ii)
            lat = sum(rows[i]["lat"] for i in ii) / len(ii)
            lng = sum(rows[i]["lng"] for i in ii) / len(ii)
            eff = beta[p + idx[c]]
            res = sum(rows[i]["y"] - sum(a * b for a, b in zip(rowsX[i], beta)) for i in ii) / len(ii)
            w.writerow([c[0], c[1], f"{lat:.5f}", f"{lng:.5f}", len(ii), round(pr[len(ii) // 2]), round(pp[len(ii) // 2]),
                        round(lots[len(ii) // 2]), f"{100*(math.exp(eff)-1):+.1f}", f"{100*(math.exp(res)-1):+.1f}",
                        "; ".join(rows[i]["address"] for i in ii[:3])])
    with open(f"data/grid_predictions_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["redfin_id", "address", "sold_date", "sold_price", "cell_x", "cell_y", "n_cell_sales",
                    "predicted_price_in_sample", "predicted_price_leave_one_out", "pct_below_loo_prediction"])
        for i, r in enumerate(rows):
            c = cells[i]
            pred = math.exp(sum(a * b for a, b in zip(rowsX[i], beta)))
            loo_pred = math.exp(r["y"] - loo[i])
            w.writerow([r["id"], r["address"], r["sold"], round(r["sold_price"]), c[0], c[1], len(by[c]), round(pred),
                        round(loo_pred), f"{100*(1-r['sold_price']/loo_pred):+.1f}"])
    say(f"\ncells with sales: {len(by)}; sales per cell: min {min(len(v) for v in by.values())}, "
        f"median {sorted(len(v) for v in by.values())[len(by)//2]}, max {max(len(v) for v in by.values())}")
    open(f"data/grid_cv_{tag}.txt", "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
