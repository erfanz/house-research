#!/usr/bin/env python3
"""Red-rock view rating 0-3, written to data/red_rock_rating.jsonl (the originals red_rock_view / red_rock_strict stay).

Scale (agreed with the user):
  0  no red rock visible or claimed
  1  glimpse: distant, small in the frame, or partly hidden by trees
  2  clear red-rock view from the house or deck, but at a distance or wide and muted
  3  large red-rock formations dominate the view from the main living space or deck, close or wide and unobstructed

Sources:
  photo  the rating comes from the downloaded listing photos (data/red_rock_photo.jsonl, `draft_rating`), the best evidence.
  text   no photo review: capped at 2 and low confidence. 2 if the strict text flag is 1, 1 if only the loose flag is 1, else 0.
Photo ratings are the reviewer's reading of up to ~15 listing photos per home. Agents choose which photos to show, and
a view that is not photographed ('not seen') can still exist, so a photo rating is a floor for what was visible.
"""
import json
from pathlib import Path

# Ratings for homes reviewed early, before `draft_rating` was recorded in red_rock_photo.jsonl (same scale, same photos).
EARLY = {
    "123836328": 3,  # 170 Spirit Trl: lead photo shows a large red-rock butte through floor-to-ceiling windows
    "123064428": 3,  # 420 Shadow Rock Dr: large red spires directly behind the house, seen from balcony and living room
    "124179591": 3,  # 750 Dry Creek Rd: deck photos show large red buttes close by, nothing in the way
    "124316745": 1,  # 365 Kachina Dr: small peak seen through trees from ground level
    "123178229": 3,  # 31 Feather Way: large red mountain behind the house, a butte framed by the sitting-room windows
    "122990691": 3,  # 70 Ruby Dr: a wide red ridge through the great-room glass wall
    "124326888": 3,  # 2230 Edgewood Dr: a large red mountain rises directly behind the house
    "124333352": 1,  # 30 W Gunsmoke Rd: a red ridge above the courtyard wall, a distant formation upstairs; no large view
    "124115902": 3,  # 1275 Dry Creek Rd: large red cliffs immediately behind the house and pool
    "122672379": 3,  # 20 Paisano Rd: a large red butte and spire directly behind the house
    "162365414": 2,  # 10 Stetson Ct: a large red mountain close by, but interior windows show only trees
    "124340558": 2,  # 155 Goodrow Ln: distant red ridges and mesas, clear but muted
    "123043126": 1,  # 35 Color Cove Rd: two small red buttes over the courtyard wall in one photo
    "116672374": 3,  # 275 Apache Trl: large red formation near the house, framed from the porch and behind the pond
    "123139524": 1,  # 815 Kachina Dr: small red formations through a window, otherwise distant
    "123043410": 1,  # 36 W Dove Wing Dr: red cliffs only far off, interior windows show trees
}


def main():
    feats = [json.loads(l) for l in open("data/features.jsonl") if l.strip()]
    photo = {}
    p = Path("data/red_rock_photo.jsonl")
    if p.exists():
        for l in p.read_text().split("\n"):
            if l.strip():
                r = json.loads(l)
                photo[r["redfin_id"]] = r
    out, n = [], {"photo": 0, "text": 0}
    for f in feats:
        rid = f["redfin_id"]
        ph = photo.get(rid)
        rating = None
        if ph is not None:
            rating = ph.get("draft_rating", EARLY.get(rid))
        if rating is not None:
            src, conf = "photo", "high" if ph.get("red_rock_photo") == "confirmed" else "medium"
            why = ph["evidence"]
        else:
            src, conf = "text", "low"
            rating = 2 if f.get("red_rock_strict") else (1 if f.get("red_rock_view") else 0)
            why = f"text only: strict={f.get('red_rock_strict')}, loose={f.get('red_rock_view')}, view_score={f.get('view_score')}"
        n[src] += 1
        out.append({"redfin_id": rid, "red_rock_rating": rating, "source": src, "confidence": conf, "why": why})
    with open("data/red_rock_rating.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n")
    dist = {s: sum(1 for o in out if o["source"] == s and o["red_rock_rating"] == s_) for s in ("photo", "text") for s_ in (0,)}
    for s in ("photo", "text"):
        print(s, n[s], {k: sum(1 for o in out if o["source"] == s and o["red_rock_rating"] == k) for k in (0, 1, 2, 3)})


if __name__ == "__main__":
    main()
