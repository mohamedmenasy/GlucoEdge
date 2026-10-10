#!/usr/bin/env bash
# No-network guard, iOS edition. iOS has no INTERNET permission, so unlike
# Android's merged-manifest check this is only a source scan — a weaker,
# honest guarantee (documented in the README). Fails if any app or test
# source references a networking API.
set -euo pipefail
cd "$(dirname "$0")/.."

pattern='URLSession|import Network|NWConnection|CFSocket|getaddrinfo'
if matches=$(grep -rnE "$pattern" GlucoEdge GlucoEdgeTests --include='*.swift'); then
  echo "Networking API references found in iOS sources:" >&2
  echo "$matches" >&2
  exit 1
fi
echo "check_no_network: no networking API references in iOS sources."
