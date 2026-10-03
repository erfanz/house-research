#!/bin/bash
# 512px-wide copies (data/images_512/<id>/NNN.jpg, quality 60) of batch-3 photos, for cheaper review.
cd "$(dirname "$0")"
for id in $(cat data/photo_shortlist_3.txt); do
  [ -d "data/images/$id" ] || continue
  mkdir -p "data/images_512/$id"
  for f in data/images/$id/*.jpg; do
    out="data/images_512/$id/$(basename "$f")"
    [ -s "$out" ] || sips -Z 512 -s formatOptions 60 "$f" --out "$out" >/dev/null 2>&1
  done
done
