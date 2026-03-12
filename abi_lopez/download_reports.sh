#!/usr/bin/env bash
set -euo pipefail

VOLUME="reft-reports"
DEST="${1:-./modal_reports}"
BUNDLE="reports_bundle.tar.gz"

modal run bundle_reports.py
modal volume get "$VOLUME" "$BUNDLE" "./$BUNDLE"

mkdir -p "$DEST"
tar -xzf "./$BUNDLE" -C "$DEST"

printf "\nDownloaded reports to %s\n" "$DEST"
