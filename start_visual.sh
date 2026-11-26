#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mode="${1:-local}"
python_bin="${PYTHON:-python}"
if [[ $# -gt 0 ]]; then shift; fi
# This affects only the simulator/perception process, not the existing Ollama server.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NAV_EXECUTION_MODE="${NAV_EXECUTION_MODE:-async}"
export NAV_IMAGE_POLICY="${NAV_IMAGE_POLICY:-adaptive}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}dashscope.aliyuncs.com,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
case "$mode" in
  local)
    export NAV_VLM_BACKEND=ollama
    export NAV_VLM_BASE_URL=http://127.0.0.1:11434
    export NAV_VLM_API_KEY=''
    export NAV_VLM_MODEL="${NAV_VLM_MODEL:-qwen3.8:27b-q4_K_M}"
    export NAV_VLM_NUM_CTX="${NAV_VLM_NUM_CTX:-16384}"
    export NAV_VLM_TIMEOUT="${NAV_VLM_TIMEOUT:-300}"
    "$python_bin" -m scripts.utilities.warmup_vlm
    ;;
  cloud)
    export NAV_VLM_BACKEND=openai
    export NAV_VLM_MODEL=qwen3.8-max
    unset NAV_VLM_BASE_URL NAV_VLM_API_KEY NAV_VLM_TIMEOUT
    # Cloud endpoint/key remain in the existing ignored config/model_config.json.
    ;;
  *) echo 'Usage: bash start_visual.sh local|cloud [serve.py arguments]' >&2; exit 2 ;;
esac
exec "$python_bin" serve.py --port 8770 "$@"
