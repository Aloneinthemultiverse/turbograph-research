#!/usr/bin/env bash
# Pushes the benchmark corpora as a private Kaggle dataset, runs the OpenIE worker
# on a free 2x T4 GPU, waits, and downloads <dataset>.facts.jsonl into data/hipporag/.
#
#   KAGGLE_CONFIG_DIR=~/.kaggle4 kaggle/openie/run_openie.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
USER_NAME="$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d '\r ')"
DS_SLUG="turbograph-hipporag-corpora"
KERNEL_SLUG="turbograph-openie"

DS="$HERE/build/dataset"
rm -rf "$DS" && mkdir -p "$DS"
cp "$ROOT"/data/hipporag/*.corpus.tsv "$ROOT"/data/hipporag/*.questions.tsv "$DS/"
cat > "$DS/dataset-metadata.json" <<EOF
{"title": "$DS_SLUG", "id": "$USER_NAME/$DS_SLUG", "licenses": [{"name": "CC-BY-SA-4.0"}]}
EOF
if kaggle datasets status "$USER_NAME/$DS_SLUG" >/dev/null 2>&1; then
  (cd "$DS" && kaggle datasets version -p . -m "sync $(date +%F_%T)" -q)
else
  (cd "$DS" && kaggle datasets create -p . -q)
fi
for _ in $(seq 1 60); do
  [ "$(kaggle datasets status "$USER_NAME/$DS_SLUG" 2>/dev/null | tail -1 | tr -d '\r ')" = "ready" ] && break
  sleep 10
done

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
  "dataset_sources": ["$USER_NAME/$DS_SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
(cd "$K" && kaggle kernels push -p .)
echo "pushed $USER_NAME/$KERNEL_SLUG"
