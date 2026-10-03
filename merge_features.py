#!/usr/bin/env python3
"""Merge the hand-read description features with the structured MLS fields into data/features.jsonl.

Inputs:
  data/features_reading.jsonl   one object per house, read from the listing description by a person/model, each with the
                                supporting quotes in `q`. view_score null means the text and MLS View field say nothing.
  data/structured_facts.jsonl   from extract_structured.py (MLS fields read directly from the saved pages)
Output: data/features.jsonl with the keys analyze.py expects (view_score, red_rock_view, forest_backing, pool,
renovated, confidence) plus guest_unit, needs_work, spa, sewer, water, STR notes and the quotes.

Pool = the remarks describe the home's own pool, OR the MLS lists Pool Features. The MLS "Private Pool" tick alone is
not trusted: gated-community listings tick it for the HOA's pool.
"""
import json
from pathlib import Path


# Sales left out of the model and the ranking (kept in features.jsonl with the reason).
EXCLUDE = {
    "123623954": "duplicate: same sale (2025-12-05, $2.5M, 4,125 sq ft) as 750 Rhapsody Rd (124349641); this row is the '730 & 750' listing",
    "124033387": "price includes an adjacent lot (15 Bronco Ct): $1.55M residence + $200K lot",
    "122983530": "price includes an adjacent parcel (390 Last Wagon Dr, listed at $259,000)",
    "123585394": "price includes a second 2.5-acre parcel (408-22-105) with no vehicular access",
}


def load(p):
    return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]


def main():
    reading = {r["id"]: r for r in load("data/features_reading.jsonl")}
    struct = {r["redfin_id"]: r for r in load("data/structured_facts.jsonl")}
    strict = {r["redfin_id"]: r for r in load("data/red_rock_strict.jsonl")} if Path("data/red_rock_strict.jsonl").exists() else {}
    rating = {r["redfin_id"]: r for r in load("data/red_rock_rating.jsonl")} if Path("data/red_rock_rating.jsonl").exists() else {}
    out = []
    for rid, r in reading.items():
        s = struct.get(rid, {})
        pool_feat = "Pool Features" in (s.get("pool_detail") or "")
        pool = 1 if (r["pool_desc"] == 1 or pool_feat) else 0
        out.append({
            "redfin_id": rid,
            "view_score": r["view_score"],
            "red_rock_view": r["red_rock_view"],
            "red_rock_strict": strict.get(rid, {}).get("red_rock_strict"),
            "red_rock_strict_reason": strict.get(rid, {}).get("red_rock_strict_reason"),
            "forest_backing": r["forest_backing"],
            "pool": pool,
            "pool_mls_tick_only": 1 if (s.get("private_pool") and not pool) else 0,
            "renovated": r["renovated"],
            "needs_work": r["needs_work"],
            "guest_unit": r["guest_unit"],
            "spa": 1 if s.get("spa") else 0,
            "confidence": r["confidence"],
            "str_note": r["str_note"],
            "exclude_reason": EXCLUDE.get(rid, ""),
            "red_rock_rating": rating.get(rid, {}).get("red_rock_rating"),
            "red_rock_rating_source": rating.get(rid, {}).get("source"),
            "mls_view_field": s.get("view_field"),
            "sewer": s.get("sewer"),
            "water": s.get("water"),
            "quotes": r["q"],
        })
    with open("data/features.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n = len(out)
    print(f"{n} houses -> data/features.jsonl")
    for k in ("red_rock_view", "red_rock_strict", "forest_backing", "pool", "renovated", "needs_work", "guest_unit", "spa"):
        print(f"  {k}: {sum(o[k] or 0 for o in out)}")
    vs = [o["view_score"] for o in out]
    print("  view_score:", {v: vs.count(v) for v in (None, 1, 2, 3, 4, 5)})
    print("  low confidence:", sum(o["confidence"] == "low" for o in out))


if __name__ == "__main__":
    main()
