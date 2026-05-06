"""
common.py
=========
Shared utilities for all runners (FlatHOIL / TierHOIL):
  - Config (paths, model names, API key) — all environment-driven
  - LLM client (OpenAI / vLLM via OpenAI-compatible endpoint)
  - JSONL data loader (MCQ dataset, per-question subgraph)
  - Batch runner with checkpoint-resume from CSV
  - MCQ adapter: open-ended LLM output → a/b/c/d/e
  - Evaluation summary

Path & key configuration is via environment variables only — no personal
paths or secrets are hardcoded.  See README and `env_example.sh`.
"""
import os, re, csv, json, math, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterable
from openai import OpenAI


# =====================================================================
# CONFIG (env-driven; override with HOLI_* environment variables)
# =====================================================================
# Path defaults assume the dataset/graph have been placed under ./data/.
# Override per-run via:
#   export HOLI_GRAPH_PATH=/path/to/question_graphs.jsonl
#   export HOLI_DATASET_PATH=/path/to/questions.jsonl
#   export HOLI_OUTPUT_DIR=/path/to/results
GRAPH_PATH = os.environ.get(
    "HOLI_GRAPH_PATH",
    "./data/question_graphs_gold_patched_with_types.jsonl",
)
DATASET_PATH = os.environ.get(
    "HOLI_DATASET_PATH",
    "./data/merged_questions_with_none_cleaned_fixed.jsonl",
)
OUTPUT_DIR = os.environ.get(
    "HOLI_OUTPUT_DIR",
    "./results",
)


# =====================================================================
# LLM BACKEND CONFIG (vLLM / OpenAI abstraction)
# =====================================================================
# All runners call common.chat() only; the backend is selected here.
#
# Preset system: short alias → (model, endpoint, max_tokens) bundle.
# Override via env var:
#   export LLM_PRESET=qwen3-32b              # alias
#   export LLM_PRESET=llama-3.1-8b
#   export LLM_PRESET=gpt-5-nano
#
# Or set individual env vars (preset bypass):
#   export LLM_BACKEND=vllm
#   export LLM_MODEL=Qwen/Qwen3-32B
#   export LLM_BASE_URL=http://localhost:8000/v1
#   export LLM_API_KEY=dummy
#
# Embedding always uses OpenAI (independent of LLM):
#   export OPENAI_API_KEY=sk-...
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# IMPORTANT — temperature constraint:
# OpenAI's GPT-5 family (gpt-5*, gpt-5.4-*, gpt-5.5) and o-series
# (o1, o3, o3-mini, o4-mini) are reasoning models and REJECT
# temperature != 1 in the chat-completions endpoint.  This codebase
# requires temperature=0 for deterministic, reproducible evaluation,
# so the OpenAI presets below all use NON-reasoning models that
# accept temperature=0:
#   - gpt-4o-mini       : the de-facto KGQA reference baseline
#   - gpt-4.1-nano      : OpenAI's smallest/cheapest non-reasoning model
#   - gpt-4.1-mini      : larger non-reasoning option
# A "gpt-5-nano-reasoning" preset is kept as a commented example for
# ablation; switching it on will trigger the temperature-stripping
# code path in `chat()` below.
# ─────────────────────────────────────────────────────────────────────
LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    # ── OpenAI non-reasoning (temperature=0 supported) ──
    "gpt-4o-mini": {
        "backend": "openai",
        "model": "gpt-4o-mini-2024-07-18",      # pinned snapshot for reproducibility
        "base_url": None,                       # OpenAI default
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 1024,
    },
    "gpt-4.1-nano": {
        "backend": "openai",
        "model": "gpt-4.1-nano-2025-04-14",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 1024,
    },
    "gpt-4.1-mini": {
        "backend": "openai",
        "model": "gpt-4.1-mini-2025-04-14",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 1024,
    },
    # ── OpenAI reasoning (temperature is force-stripped at call time) ──
    # Output is non-deterministic; only use for ablation, not the main
    # comparison.  Uncomment to enable.
    # "gpt-5-nano-reasoning": {
    #     "backend": "openai",
    #     "model": "gpt-5-nano",
    #     "base_url": None,
    #     "api_key_env": "OPENAI_API_KEY",
    #     "max_tokens": 1024,
    #     "is_reasoning": True,                 # consumed by `chat()`
    # },
    # ── Qwen (open, vLLM) ──
    "qwen3-32b": {
        "backend": "vllm",
        "model": "Qwen/Qwen3-32B",
        "base_url_env": "VLLM_QWEN3_32B_URL",
        "base_url_default": "http://localhost:8000/v1",
        "api_key": "dummy",                     # vLLM does not validate keys
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    # ── Llama (open, vLLM) ──
    "llama-3.1-8b": {
        "backend": "vllm",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url_env": "VLLM_LLAMA_8B_URL",
        "base_url_default": "http://localhost:8001/v1",
        "api_key": "dummy",
        "max_tokens": 2048,
    },
}

# Preset selection: env LLM_PRESET → fallback "gpt-4o-mini"
_DEFAULT_PRESET = os.environ.get("LLM_PRESET", "gpt-4o-mini").strip().lower()


def _resolve_llm_config() -> Dict[str, Any]:
    """
    Combine preset + env overrides into a single LLM config dict.

    Priority:
      1. preset's base values
      2. *_env keys inside the preset (override base if env var is set)
      3. individual env vars (LLM_BACKEND, LLM_MODEL, ...) take final
         precedence when ALL of LLM_BACKEND/LLM_MODEL are present
    """
    preset_name = os.environ.get("LLM_PRESET", _DEFAULT_PRESET).strip().lower()
    if preset_name not in LLM_PRESETS:
        valid = ", ".join(sorted(LLM_PRESETS))
        raise RuntimeError(
            f"Unknown LLM_PRESET={preset_name!r}.  Available: {valid}.  "
            f"Either set LLM_PRESET to one of these, or set "
            f"LLM_BACKEND/LLM_MODEL/LLM_BASE_URL directly."
        )
    cfg = dict(LLM_PRESETS[preset_name])
    cfg["preset_name"] = preset_name

    # Resolve base_url
    if cfg["backend"] == "vllm":
        env_var = cfg.get("base_url_env")
        if env_var and os.environ.get(env_var):
            cfg["base_url"] = os.environ[env_var]
        else:
            cfg["base_url"] = cfg.get("base_url_default")
    # OpenAI: base_url stays None (= default endpoint)

    # Resolve api_key
    if cfg["backend"] == "openai":
        env_var = cfg.get("api_key_env", "OPENAI_API_KEY")
        cfg["api_key"] = os.environ.get(env_var, "")
        if not cfg["api_key"]:
            raise RuntimeError(
                f"OpenAI backend requires {env_var} environment variable.  "
                f"Set it via:  export {env_var}=sk-..."
            )
    # vLLM uses the hardcoded 'dummy' key from the preset.

    # Individual env-var override (all-or-nothing for clarity)
    if all(os.environ.get(k) for k in ("LLM_BACKEND", "LLM_MODEL")):
        cfg["backend"] = os.environ["LLM_BACKEND"]
        cfg["model"] = os.environ["LLM_MODEL"]
        cfg["base_url"] = os.environ.get("LLM_BASE_URL") or cfg.get("base_url")
        cfg["api_key"] = os.environ.get("LLM_API_KEY") or cfg.get("api_key", "dummy")
        cfg["preset_name"] = "custom"

    return cfg


# Module-load time: resolve once
LLM_CONFIG = _resolve_llm_config()
MODEL_NAME = LLM_CONFIG["model"]
LLM_BACKEND = LLM_CONFIG["backend"]
LLM_BASE_URL = LLM_CONFIG.get("base_url")
LLM_API_KEY = LLM_CONFIG.get("api_key", "")
LLM_EXTRA_BODY: Dict[str, Any] = LLM_CONFIG.get("extra_body", {}) or {}


def _is_openai_reasoning_model(model_name: str, preset_flag: bool = False) -> bool:
    """
    True if `model_name` is an OpenAI reasoning model that rejects
    `temperature` (and other sampling params) at the chat-completions
    endpoint.  Detection is conservative: explicit preset flag wins;
    otherwise we match the GPT-5 family and o-series by name prefix.
    """
    if preset_flag:
        return True
    if not model_name:
        return False
    m = model_name.lower()
    # GPT-5 family: gpt-5, gpt-5-nano, gpt-5.4-*, gpt-5.5, etc.
    if m.startswith("gpt-5"):
        return True
    # o-series: o1, o1-mini, o3, o3-mini, o4-mini (but not gpt-4o!)
    if re.match(r"^o[1-9](?:-|$)", m):
        return True
    return False


# Whether the active model rejects `temperature` and other sampling params
IS_REASONING_MODEL = (
    LLM_BACKEND == "openai"
    and _is_openai_reasoning_model(
        MODEL_NAME,
        preset_flag=bool(LLM_CONFIG.get("is_reasoning", False)),
    )
)

# Embeddings: always OpenAI (independent of LLM backend).
EMBED_MODEL = "text-embedding-3-small"
EMBED_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Common LLM hyperparameters.
LLM_TEMPERATURE_EXPLORATION = 0.4   # not used by FlatHOIL/TierHOIL but kept for consistency
LLM_TEMPERATURE_REASONING   = 0.0   # deterministic — auto-stripped for reasoning models
LLM_MAX_TOKENS              = LLM_CONFIG.get("max_tokens", 512)

# QID subsample modulus (set > 1 to evaluate on every k-th question)
QID_SUBSAMPLE_MOD = int(os.environ.get("HOLI_QID_SUBSAMPLE_MOD", "1"))

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# Banner: which LLM are we running?
print(f"[common.py] LLM_PRESET={LLM_CONFIG['preset_name']}  "
      f"backend={LLM_BACKEND}  model={MODEL_NAME}  "
      f"base_url={LLM_BASE_URL or 'openai-default'}  "
      f"reasoning={IS_REASONING_MODEL}")
print(f"[common.py] EMBED_MODEL={EMBED_MODEL}")
if IS_REASONING_MODEL:
    print(f"[common.py] WARNING: '{MODEL_NAME}' is a reasoning model.  "
          f"`temperature` will be stripped from chat() calls.  Output is "
          f"NON-deterministic — use a non-reasoning model "
          f"(gpt-4o-mini / gpt-4.1-nano) for the main paper comparison.")


# =====================================================================
# CLIENTS (LLM client + Embedding client are kept separate)
# =====================================================================
_llm_client = None
_embed_client = None


def get_client() -> OpenAI:
    """LLM-call client.  base_url depends on backend."""
    global _llm_client
    if _llm_client is None:
        kwargs: Dict[str, Any] = {"api_key": LLM_API_KEY or "dummy"}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _llm_client = OpenAI(**kwargs)
    return _llm_client


def get_embed_client() -> OpenAI:
    """Embedding-only client.  Always OpenAI."""
    global _embed_client
    if _embed_client is None:
        if not EMBED_API_KEY:
            raise RuntimeError(
                "Embedding requires OPENAI_API_KEY environment variable.  "
                "Even when running LLMs via vLLM, embeddings still use OpenAI.  "
                "Set:  export OPENAI_API_KEY=sk-..."
            )
        _embed_client = OpenAI(api_key=EMBED_API_KEY)
    return _embed_client


def chat(prompt: str,
         temperature: float = None,
         max_tokens: int = None,
         system: Optional[str] = None,
         few_shot_messages: Optional[List[Dict[str, str]]] = None,
         retry: int = 3,
         retry_sleep: float = 2.0) -> str:
    """
    Single chat-completion call.  Backend-agnostic — works for both
    OpenAI and vLLM (OpenAI-compatible) endpoints.

    - Supports system + few-shot + user prompt structure.
    - Exponential-backoff retry on failure.
    - vLLM: preset's extra_body (e.g., enable_thinking) is auto-injected.

    Args:
        temperature: if None, uses LLM_TEMPERATURE_REASONING (0.0)
        max_tokens : if None, uses LLM_MAX_TOKENS (preset default)
    """
    if temperature is None:
        temperature = LLM_TEMPERATURE_REASONING
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS

    client = get_client()
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": prompt})

    # OpenAI prefers max_completion_tokens; vLLM uses max_tokens.
    # We send only the parameter the active backend understands.
    create_kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    # Reasoning models (GPT-5 family, o-series) reject the `temperature`
    # parameter in chat-completions.  Strip it for those, pass it for
    # everything else.  See _is_openai_reasoning_model() above.
    if not IS_REASONING_MODEL:
        create_kwargs["temperature"] = temperature
    if LLM_BACKEND == "openai":
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens
        if LLM_EXTRA_BODY:
            create_kwargs["extra_body"] = LLM_EXTRA_BODY

    last_err = None
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content
            if content is None:
                # vLLM may return empty content (esp. when the model
                # tries to think but thinking is OFF).  Normalize to "".
                return ""
            return content.strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {retry} retries: {last_err}")


def embed_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embedding call.  Always OpenAI (independent of LLM backend)."""
    client = get_embed_client()
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def top_k_indices(query_emb: List[float],
                  candidate_embs: List[List[float]],
                  k: int) -> List[int]:
    scores = [(cosine_sim(query_emb, e), i) for i, e in enumerate(candidate_embs)]
    scores.sort(reverse=True)
    return [i for _, i in scores[:k]]


# =====================================================================
# JSONL loaders
# =====================================================================
def load_jsonl(path: str, key: str = "question_id") -> Dict[int, Dict[str, Any]]:
    """Load a JSONL file into a dict keyed by `question_id`."""
    data: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row.get(key)
            if qid is not None:
                data[int(qid)] = row
    return data


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =====================================================================
# CSV utils (result store + resume)
# =====================================================================
def ensure_csv(path: str, headers: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def append_csv(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def load_done_qids(csv_path: str) -> set:
    done: set = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["question_id"]))
            except Exception:
                pass
    return done


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =====================================================================
# MCQ adapter
# =====================================================================
_CHOICE_LETTERS = ["a", "b", "c", "d", "e"]


def format_choices(sample: Dict[str, Any]) -> str:
    """Format MCQ choices as a labelled list."""
    return "\n".join(f"{k}. {sample[k]}" for k in _CHOICE_LETTERS)


def extract_mcq_letter(text: str) -> str:
    """
    Extract a/b/c/d/e from an open-ended LLM response.
    Handles "a", "(a)", "**a**", "Answer: A", "The answer is (c).",
    "{b}", and similar variants.  Returns 'fail' if nothing matches.
    """
    if not text:
        return "fail"
    low = text.strip().lower()
    if low in set(_CHOICE_LETTERS):
        return low
    # ToG-style {x}
    m = re.search(r"\{\s*([abcde])\s*\}", low)
    if m:
        return m.group(1)
    # "Answer: X"
    m = re.search(r"\b(?:answer|option|choose|pick|select)\s*(?:is\s*)?[:\-]?\s*\(?\s*([abcde])\s*\)?\b", low)
    if m:
        return m.group(1)
    # "(a)", "a.", "a:" at start
    m = re.match(r"^\s*\*?\*?\s*\(?\s*([abcde])\s*[\)\.\:]", low)
    if m:
        return m.group(1)
    # First leading letter
    m = re.match(r"^\s*\*?\*?\s*([abcde])\b", low)
    if m:
        return m.group(1)
    return "fail"


def mcq_via_embedding(open_ended_answer: str, sample: Dict[str, Any]) -> str:
    """
    Match an open-ended answer to one of {a..e} via embedding similarity.
    Used as a fallback when the original baseline emits free-text rather
    than a letter.
    """
    if not open_ended_answer or not open_ended_answer.strip():
        return "fail"
    try:
        choice_texts = [str(sample.get(k, "")) for k in _CHOICE_LETTERS]
        embs = embed_batch([open_ended_answer] + choice_texts)
        query = embs[0]
        cand = embs[1:]
        best = top_k_indices(query, cand, 1)
        if not best:
            return "fail"
        return _CHOICE_LETTERS[best[0]]
    except Exception:
        return "fail"


MCQ_INSTRUCTION_SUFFIX = (
    "\n\nFrom the choices below, select the single best answer. "
    "Respond with ONLY one letter: a, b, c, d, or e.\n"
    "Choices:\n{choices}\n\nAnswer (just one letter):"
)


def build_mcq_answer_prompt(question: str,
                            sample: Dict[str, Any],
                            reasoning_context: str = "") -> str:
    """Build the final MCQ prompt from a method's collected reasoning context."""
    parts = []
    if reasoning_context:
        parts.append(reasoning_context.strip())
    parts.append(f"Question: {question}")
    choices = format_choices(sample)
    parts.append(MCQ_INSTRUCTION_SUFFIX.format(choices=choices).strip())
    return "\n\n".join(parts)


# =====================================================================
# Batch runner (resume from CSV)
# =====================================================================
def select_qids(adj_or_graph_keys: Iterable[int],
                only_qids: Optional[List[int]] = None,
                subsample_mod: int = QID_SUBSAMPLE_MOD) -> List[int]:
    """Sub-sampling + user-specified-qid filter."""
    if only_qids is None:
        return sorted(q for q in adj_or_graph_keys if q % subsample_mod == 0)
    return sorted(int(q) for q in only_qids if int(q) % subsample_mod == 0)


def run_batch(method_name: str,
              method_fn: Callable,
              qids: List[int],
              dataset: Dict[int, Dict[str, Any]],
              output_dir: str = OUTPUT_DIR,
              wiki_client: Any = None,
              **method_kwargs) -> str:
    """
    Common run harness:
      - answer_<method>.csv: per-qid letter, gold, is_correct, status, ts
      - log_<method>.jsonl : raw LLM I/O at every step (for audit)

    Args:
        method_fn: signature = method_fn(qid, sample, log_path, **kwargs) -> str
        wiki_client: kept for compatibility with non-released baselines.
                     For FlatHOIL/TierHOIL it is None and unused.
    Returns:
        path to answer CSV
    """
    ans_csv   = os.path.join(output_dir, f"answer_{method_name}.csv")
    log_jsonl = os.path.join(output_dir, f"log_{method_name}.jsonl")
    ensure_csv(ans_csv, ["question_id", "prediction", "gold_answer",
                         "is_correct", "status", "timestamp"])
    done = load_done_qids(ans_csv)

    try:
        from tqdm.auto import tqdm
        pbar = tqdm([q for q in qids if q not in done], desc=f"[{method_name}]")
    except ImportError:
        pbar = [q for q in qids if q not in done]

    has_set_question = wiki_client is not None and hasattr(wiki_client, "set_question")

    for qid in pbar:
        if qid not in dataset:
            continue
        sample = dataset[qid]
        gold = (sample.get("answer") or "").strip().lower()

        if has_set_question:
            try:
                wiki_client.set_question(qid)
            except Exception as e:
                append_csv(ans_csv, [qid, "fail", gold, "",
                                      f"no_subgraph: {e}", now_str()])
                continue

        try:
            pred = method_fn(qid=qid, sample=sample,
                             log_path=log_jsonl, **method_kwargs)
            status = "done"
        except Exception as e:
            pred = "fail"
            status = f"error: {type(e).__name__}: {e}"
            # Print tracebacks for the first 3 errors per method (debug aid)
            _error_count_key = f"_{method_name}_errors"
            n_errs = run_batch.__dict__.setdefault(_error_count_key, 0)
            if n_errs < 3:
                import traceback as _tb
                print(f"\n[{method_name}] qid={qid} error ({n_errs+1}/3 shown):")
                _tb.print_exc()
                run_batch.__dict__[_error_count_key] = n_errs + 1
            try:
                append_jsonl(log_jsonl, {
                    "timestamp": now_str(), "question_id": qid,
                    "method": method_name, "step": "exception",
                    "error": status,
                })
            except Exception:
                pass

        is_correct = (int(pred == gold) if pred in _CHOICE_LETTERS and gold in _CHOICE_LETTERS
                      else "")
        append_csv(ans_csv, [qid, pred, gold, is_correct, status, now_str()])

    print(f"[{method_name}] done -> {ans_csv}")
    return ans_csv


def summarize(method_name: str, output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """Read answer_<method>.csv and print total/fail/accuracy."""
    ans_csv = os.path.join(output_dir, f"answer_{method_name}.csv")
    if not os.path.exists(ans_csv):
        print(f"[{method_name}] no results yet")
        return {"method": method_name, "total": 0, "fail": 0, "accuracy": None}
    rows = list(csv.DictReader(open(ans_csv, encoding="utf-8")))
    flags = [int(r["is_correct"]) for r in rows if r["is_correct"] in {"0", "1"}]
    fails = sum(1 for r in rows if r["prediction"] == "fail")
    acc = (sum(flags) / len(flags)) if flags else None
    summary = {"method": method_name,
               "total": len(rows),
               "fail": fails,
               "scored": len(flags),
               "correct": sum(flags) if flags else 0,
               "accuracy": acc}
    print(f"\n=== {method_name} ===")
    print(f"  total   : {summary['total']}")
    print(f"  fail    : {summary['fail']}")
    if acc is not None:
        print(f"  accuracy: {acc:.4f}  ({summary['correct']}/{summary['scored']})")
    else:
        print(f"  accuracy: N/A")
    return summary