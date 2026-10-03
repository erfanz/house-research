#!/bin/bash
# Make 640px-wide copies (data/images_small/<id>/NNN.jpg) of any downloaded photo that has no small copy yet.
cd "$(dirname "$0")"
for d in data/images/[0-9]*/; do
  id=$(basename "$d"); mkdir -p "data/images_small/$id"
  for f in "$d"*.jpg; do
    out="data/images_small/$id/$(basename "$f")"
    [ -s "$out" ] || sips -Z 640 "$f" --out "$out" >/dev/null 2>&1
  done
done
