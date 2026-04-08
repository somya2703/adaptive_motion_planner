#!/bin/sh
# Fix ownership of /app/results before running as the planner user.
# Handles stale root-owned files from previous Docker runs.
if [ -d /app/results ]; then
    chown -R planner:planner /app/results 2>/dev/null || true
fi
exec gosu planner "$@"
