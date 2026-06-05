#!/usr/bin/env bash
# Enable Ollama continuous batching by setting OLLAMA_NUM_PARALLEL on the
# systemd-managed ollama service, via a drop-in override (non-destructive: the
# original unit file is left untouched). Requires sudo.
#
# Why: by default the service runs serialized (NUM_PARALLEL=1), so client-side
# concurrency does nothing and the GPU sits mostly idle. Benchmarked on the
# RTX 5060 with waso-translator (q8), bench_translate.py:
#
#     default (NUM_PARALLEL=1):  ~2.7 sentences/s,  117 tok/s
#     NUM_PARALLEL=8, conc 16:  ~18.4 sentences/s,  424 tok/s   (~7x, 76% GPU)
#
# i.e. the 300k-sentence corpus drops from ~34h to ~4.5h. NUM_PARALLEL=8 fits
# the 8 GB card with the q8 model (~5.9 GB used incl. the 8 KV-cache slots);
# raise/lower N to taste, watching `nvidia-smi`.
#
# Usage:
#   bash scripts/enable_ollama_parallel.sh        # NUM_PARALLEL=8 (default)
#   bash scripts/enable_ollama_parallel.sh 4      # custom value
#
# Revert:
#   sudo rm /etc/systemd/system/ollama.service.d/parallel.conf
#   sudo systemctl daemon-reload && sudo systemctl restart ollama
set -euo pipefail

N="${1:-8}"
DROPIN_DIR=/etc/systemd/system/ollama.service.d
OVERRIDE="$DROPIN_DIR/parallel.conf"

echo "Setting OLLAMA_NUM_PARALLEL=$N on the ollama service (drop-in override)…"
sudo mkdir -p "$DROPIN_DIR"
sudo tee "$OVERRIDE" >/dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=$N"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "Restarted. Verifying the env is set…"
sleep 2
if systemctl show ollama -p Environment | grep -qi "NUM_PARALLEL=$N"; then
  echo "  OK: OLLAMA_NUM_PARALLEL=$N"
else
  echo "  WARNING: NUM_PARALLEL not visible in 'systemctl show ollama -p Environment'." >&2
fi

cat <<EOF

Ollama will now continuously batch up to $N requests. Run the bulk translate:
  .venv/bin/python scripts/translate_corpus.py --concurrency 16

(translate_corpus already defaults to no-few-shot + num_predict 24.)
EOF
