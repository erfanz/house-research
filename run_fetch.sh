#!/bin/bash
# Run fetch_descriptions.py to completion, cooling down whenever Redfin blocks us.
# fetch_descriptions.py is resumable (cached pages are skipped), so each retry continues where it stopped.
cd "$(dirname "$0")" || exit 1

INITIAL_COOLDOWN=${INITIAL_COOLDOWN:-1200}   # seconds to wait before the first attempt
BACKOFFS=(1200 2400 3600 3600 5400 5400)    # waits after each successive block

log() { echo "[$(date '+%F %T')] $*"; }

log "cooling down ${INITIAL_COOLDOWN}s before starting"
sleep "$INITIAL_COOLDOWN"

attempt=0
while true; do
  log "starting fetch (attempt $((attempt + 1)))"
  python3 fetch_descriptions.py "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "all houses fetched"
    exit 0
  fi
  if [ "$rc" -eq 2 ]; then
    log "Redfin rate-limited us; stopping. Try again in 24 hours, not sooner."
    exit 2
  fi
  if [ "$attempt" -ge "${#BACKOFFS[@]}" ]; then
    log "blocked more than ${#BACKOFFS[@]} times, giving up; rerun later (progress is saved)"
    exit 1
  fi
  wait_s=${BACKOFFS[$attempt]}
  attempt=$((attempt + 1))
  log "blocked or failing; cooling down ${wait_s}s"
  sleep "$wait_s"
done
