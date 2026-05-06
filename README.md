

# HiddenPathQA
This repository is the official implementation of the two methods
proposed in our paper:

HiddenPathQA: A Benchmark for Knowledge Graph Question Answering
When Questions Hide Their Paths

The HiddenPathQA dataset (questions and per-question subgraphs) is
released separately from this repository.
What is in this repository
Two methods are released as part of this codebase:

- **Flat-HOIL** (`run_flat_holi.py`)  single-pass selection over the
full set of candidate relation chains (no depth loop).
- **Tier-HOIL** (`run_tier_holi.py`)  depth-tiered selection that grows
the search depth one hop at a time and only commits to an answer
when the current depth's evidence is sufficient.











> **Release scope.** .
> The benchmark dataset (questions and per-question subgraphs) is released
> separately; After you download the queestion and graph set, move them in to '/data' folder.

---

## 1. Requirements

```bash
pip install -r requirements.txt
```

The codebase depends on `openai>=1.0.0` (used as both an OpenAI-API
client *and* an OpenAI-compatible client for vLLM) and `tqdm`.

---

## 2. Configuration

All paths and keys are configured via environment variables.  No paths
or secrets are hardcoded in source.

```bash
cp env_example.sh env.sh
# edit env.sh: set OPENAI_API_KEY, dataset paths, and LLM_PRESET
source env.sh
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for embeddings (always OpenAI). Required for the `gpt-5-nano` preset. |
| `HOLI_DATASET_PATH` | Path to the questions JSONL. |
| `HOLI_GRAPH_PATH` | Path to the per-question subgraph JSONL (with type annotations). |
| `HOLI_OUTPUT_DIR` | Where results CSVs and step-by-step JSONL logs are written. |
| `LLM_PRESET` | One of `gpt-5-nano`, `qwen3-32b`, `llama-3.1-8b`. |
| `VLLM_QWEN3_32B_URL`, `VLLM_LLAMA_8B_URL` | Override the default vLLM endpoints (if you serve them on non-default ports). |

To run with a vLLM-served open-source model, start vLLM first, e.g.:

```bash
vllm serve Qwen/Qwen3-32B --port 8000
# in another shell:
source env.sh
python run_all.py
```

---

## 3. Quick start

```bash
# Run both methods on a 5-question smoke test
python run_all.py --limit 5

# Run only Flat-HOIL on a specific question id list
python run_all.py --only FlatHOIL --qids 10 20 30

# Tier-HOIL, full benchmark, depth budget 6
python run_all.py --only TierHOIL --tier_max_depth_limit 6
```

Both runners are **resumable**: each method writes
`answer_<method>.csv` incrementally, and a re-run skips question ids
that already have a row.

### Output files (under `HOLI_OUTPUT_DIR`)

| File | Contents |
|---|---|
| `answer_FlatHOIL.csv` / `answer_TierHOIL.csv` | One row per question: `question_id, prediction, gold_answer, is_correct, status, timestamp`. |
| `log_FlatHOIL.jsonl` / `log_TierHOIL.jsonl` | Step-by-step audit log: every LLM input/output, parsed indices, and decisions, suitable for replay or per-question debugging. |

### Running each method standalone

`run_flat_holi.py` and `run_tier_holi.py` can also be invoked directly,
which is useful when you only want to vary that method's
hyperparameters:

```bash
python run_flat_holi.py --max_attempts 2 --max_select 5 \
    --max_candidates_show 200 --max_raw_paths_show 50

python run_tier_holi.py --max_attempts 2 --max_depth_limit 6 \
    --max_select 5 --max_candidates_show 200 --max_raw_paths_show 150
```

---

## 4. Code layout

```
.
├── data
|     ├── questions.jsonl
      ├── graphs.jsonl
|
├── common.py                # LLM client, JSONL loaders, batch runner, MCQ adapter
├── local_subgraph_client.py # Local per-question subgraph backend +
│                            #   type-aware path-formatting helpers
├── run_flat_holi.py         # Flat-HOIL implementation
├── run_tier_holi.py         # Tier-HOIL implementation
├── run_all.py               # Convenience runner for both methods
├── env_example.sh           # Template environment file
└── requirements.txt
```

---

## 5. Method-id strings in logs

The internal `method` field of every JSONL log record uses the
public-release names:

- `"method": "FlatHOIL"`
- `"method": "TierHOIL"`

---


## 6. License

This project is released under the MIT License.  See `LICENSE` for details.
