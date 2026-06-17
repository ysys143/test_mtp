#!/bin/bash
set -e
source /etc/ops-llm-test.env

BASE_ARGS=(
  serve "${MODEL_ID}"
  --host 0.0.0.0
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
)

if [[ "${USE_MTP:-0}" == "1" ]]; then
  exec /home/opscheck/.local/bin/vllm "${BASE_ARGS[@]}" \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${GAMMA:-1}}"
else
  exec /home/opscheck/.local/bin/vllm "${BASE_ARGS[@]}"
fi
