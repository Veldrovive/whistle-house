#!/usr/bin/env bash
# Run a training command and, if it crashes, rerun it with --resume until it finishes.
#
#   scripts/train-forever.sh uv run whistle-house warmstart --skip-fetch ... --name humnet
#
# The first attempt runs the command as given (add --resume yourself to continue an earlier
# run); every retry appends --resume, so it picks up after the last finished epoch
# (ml/train.py). Stops after MAX_TRIES attempts (default 20), waiting RETRY_WAIT seconds
# (default 60) between them. Ctrl-C twice (once for the run, once during the wait) to stop.
set -u
# macOS: no idle/system sleep while this runs (closing the lid still sleeps a MacBook unless an
# external display is attached; a sleep pauses training and often crashes the data loader on
# wake, which the retry below then resumes from).
if command -v caffeinate >/dev/null; then
  caffeinate -i -s -w $$ &
fi
max=${MAX_TRIES:-20}
wait_s=${RETRY_WAIT:-60}
try=1
"$@"
status=$?
while [ $status -ne 0 ] && [ $try -lt $max ]; do
  try=$((try + 1))
  echo "== $(date '+%F %T') exited with $status; retry $try/$max with --resume in ${wait_s}s"
  sleep "$wait_s" || exit 130
  "$@" --resume
  status=$?
done
echo "== $(date '+%F %T') finished with exit status $status after $try attempt(s)"
exit $status
