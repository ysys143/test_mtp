#!/bin/bash
set -u
LOG=~/compare-mtp4.log
: > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
wait_ready(){ log "  waiting for vLLM health..."
  for i in $(seq 1 180); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && { log "  READY (${i}x5s)"; return 0; }
    sleep 5; done; log "  TIMEOUT"; return 1; }
wait_gpu_free(){ log "  waiting for GPU release..."
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr -d ' ')
    [ "$used" -lt 2000 ] && { log "  GPU freed (${used}MB)"; return 0; }
    sleep 3; done; log "  WARN GPU=${used}MB"; return 1; }
run_phase(){ local label=$1 gamma=$2
  log "===== PHASE: $label (USE_MTP=1, GAMMA=$gamma) ====="
  sudo sed -i "s/^USE_MTP=.*/USE_MTP=1/" /etc/ops-llm-test.env
  sudo sed -i "s/^GAMMA=.*/GAMMA=$gamma/" /etc/ops-llm-test.env
  sudo systemctl restart ops-llm-test
  wait_ready || { log "  $label FAILED"; sudo journalctl -u ops-llm-test --no-pager -n 25|grep -iE 'error|memory'|tail -5|tee -a "$LOG"; return 1; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader|tee -a "$LOG"
  python3 ~/benchmark3.py "$label" 2>&1 | tee -a "$LOG"
  sudo systemctl stop ops-llm-test
  wait_gpu_free; log ""; }
log "######## DENSE 27B-FP8 GAMMA SWEEP ########"
run_phase "MTP-g2" 2
run_phase "MTP-g3" 3
log "######## DONE ########"
