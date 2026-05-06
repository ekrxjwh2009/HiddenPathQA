# env_example.sh
# ==============
# Source this file (after editing) to configure paths, API keys, and the
# LLM backend before running run_all.py.
#
# Usage:
#   cp env_example.sh env.sh
#   # edit env.sh to fill in your keys and paths
#   source env.sh
#   python run_all.py --only FlatHOIL --limit 5

# ── Data paths (edit to point at your local copies of the dataset) ──
export HOLI_DATASET_PATH="./data/questions.jsonl"
export HOLI_GRAPH_PATH="./data/graphs.jsonl"
export HOLI_OUTPUT_DIR="./results"


# ── OpenAI API key (required for embeddings; required for OpenAI LLMs) ──
# Replace with your own key.  NEVER commit this file with a real key.
export OPENAI_API_KEY="[YOUR_KEY]"

# ── LLM backend selection ──────────────────────────────────────────
# Choose ONE preset.  See common.py:LLM_PRESETS for the full list.
#
# (1) OpenAI gpt-5-nano (closed-source baseline)
#export LLM_PRESET=[MODEL]
#
# (2) Qwen3-32B served via vLLM (recommended open-source baseline)
export LLM_PRESET=qwen3-32b
export VLLM_QWEN3_32B_URL="http://localhost:8000/v1"
#
# (3) Llama-3.1-8B-Instruct served via vLLM
# export LLM_PRESET=llama-3.1-8b
# export VLLM_LLAMA_8B_URL="http://localhost:8001/v1"

# ── Optional: override individual settings (preset bypass) ──
# export LLM_BACKEND=vllm
# export LLM_MODEL=Qwen/Qwen3-32B
# export LLM_BASE_URL=http://localhost:8000/v1
# export LLM_API_KEY=dummy
