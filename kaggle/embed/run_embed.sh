#!/usr/bin/env bash
# Runs the embedding worker on Kaggle after turbograph-openie finished, then
# downloads the embeddings into data/hipporag/turbohippo/.
#
#   KAGGLE_CONFIG_DIR=~/.kaggle4 kaggle/embed/run_embed.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
USER_NAME="$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d '\r ')"
KERNEL_SLUG="turbograph-embed"

K="$HERE/build/kernel"
rm -rf "$K" && mkdir -p "$K"
cp "$HERE/worker.py" "$K/"
cat > "$K/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/$KERNEL_SLUG",
  "title": "$KERNEL_SLUG",
  "code_file": "worker.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": ["$USER_NAME/turbograph-hipporag-corpora"],
  "competition_sources": [],
  "kernel_sources": ["$USER_NAME/turbograph-openie"]
}
EOF
(cd "$K" && kaggle kernels push -p .)
while true; do
  S="$(kaggle kernels status "$USER_NAME/$KERNEL_SLUG" 2>/dev/null | tail -1)"
  echo "$(date +%T) $S"
  case "$S" in *COMPLETE*|*complete*|*ERROR*|*error*|*CANCEL*|*cancel*) break;; esac
  sleep 120
done
OUT="$ROOT/data/hipporag/turbohippo"
mkdir -p "$OUT"
(cd "$OUT" && kaggle kernels output "$USER_NAME/$KERNEL_SLUG" -p . -o)
ls -la "$OUT"
