#!/usr/bin/env python3
"""Strict red-rock-view flag, stored as a NEW field (red_rock_strict); the original red_rock_view is left as is.

Rule (judged from the description, not the MLS View field): red_rock_strict = 1 only if
  1. the description itself describes the view from the home (not just the MLS View: field),
  2. it names a specific formation, or says unobstructed / panoramic / direct / 360 about red rocks,
  3. it is not hedged (glimpses, peek-a-boo, "some") and not only from a rooftop deck,
  4. view_score >= 4.
Everything else gets 0 plus a reason. Run: python3 red_rock_strict.py  (writes data/red_rock_strict.jsonl)
"""
import json

STRICT = """123836328 123043126 123064428 124316745 123011629 124032419 122672379 123178229 122685851 122990691 162365414
124179591 123387189 124326888 123175528 124115902 116211076 116532995 116450787 124340558 116451781 124179558 123962062
116330721 116562789 122985461 122986101 122740215 123910014 116881704 123585394 22043459 122743870 116222759 123388443
202099463 123045356 124023056 167290034 116585355 122593976 123944782 122983530 123971765 116661323 124333352 116701998
116055880 116771510 123630506 116441892 116794054 123259469 123972226 124325512 116430208 122987892 124032194 123414738
123416376 162304333 116863546 123395962 116794215 124333396 171385871 192292286 116233580 116605970 122766966 124083844
123045081 116245262 116763940 162364858 116771457 116862763 124054196 122594033 124349641 124325292 123623954""".split()

# Candidates (score >= 4, description-based) excluded for a specific reason; any other excluded candidate = generic phrase.
REASON = {
    "116558128": "rooftop_only (360 deg view is from the rooftop deck only)",
    "116856674": "rooftop_only (named formations seen from the rooftop deck only)",
    "116084151": "generic_phrase and rooftop deck ('dramatic red mountain views')",
    "124333254": "red rock not stated in description (panoramic/canyon views only; MLS field says red rocks)",
    "116645986": "borderline: 'sweeping vistas of the surrounding formations', no named formation or explicit panoramic red rock",
    "123079874": "panoramic views are of 'surrounding landscape', not tied to red rock",
}


def main():
    feats = [json.loads(l) for l in open("data/features.jsonl") if l.strip()]
    out, n1 = [], 0
    for f in feats:
        rid, q = f["redfin_id"], f["quotes"].get("view", "")
        if rid in STRICT:
            s, why = 1, "named formation or unobstructed/panoramic/direct/360 red rock view in the description"
        elif not f["red_rock_view"]:
            s, why = 0, "original flag was 0"
        elif q.startswith("("):
            s, why = 0, "description silent; MLS View field only"
        elif f["view_score"] is None or f["view_score"] < 4:
            s, why = 0, f"view_score {f['view_score']} < 4 (hedged, partial or vague)"
        else:
            s, why = 0, REASON.get(rid, "generic phrase ('stunning/breathtaking red rock views') with no named formation, panoramic, unobstructed or direct")
        n1 += s
        out.append({"redfin_id": rid, "red_rock_strict": s, "red_rock_strict_reason": why})
    with open("data/red_rock_strict.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(o) for o in out) + "\n")
    missing = set(STRICT) - {f["redfin_id"] for f in feats}
    print(f"{n1} of {len(feats)} homes red_rock_strict=1 (original red_rock_view: {sum(f['red_rock_view'] for f in feats)})")
    print("ids not found:", missing or "none")


if __name__ == "__main__":
    main()
