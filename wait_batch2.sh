#!/bin/bash
# Exit when at least N more batch-2 homes are complete (or the downloader has stopped), so I get notified to review them.
cd "$(dirname "$0")"
N=${1:-8}
start=$(ls data/images | grep -c '^[0-9]')
until [ $(( $(ls data/images | grep -c '^[0-9]') - start )) -ge $N ] || ! pgrep -f "fetch_images.py --ids-file data/photo_shortlist_2.txt" >/dev/null; do sleep 60; done
