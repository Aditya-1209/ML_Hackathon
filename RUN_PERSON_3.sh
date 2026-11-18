#!/bin/sh
set -eu
cd "$(dirname "$0")"
if command -v python3.12 >/dev/null 2>&1; then
  python3.12 start_person3.py "$@"
elif [ -x /opt/homebrew/bin/python3.12 ]; then
  /opt/homebrew/bin/python3.12 start_person3.py "$@"
else
  python3 start_person3.py "$@"
fi
