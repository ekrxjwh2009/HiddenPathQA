"""
run_all.py
==========
Convenience runner for the two proposed methods of HOLI: Flat-HOLI and
Tier-HOLI.

Examples:
    python run_all.py
    python run_all.py --only FlatHOIL
    python run_all.py --only TierHOIL
    python run_all.py --only flat_holi tier_holi
    python run_all.py --qids 10 20 30
    python run_all.py --limit 5
    python run_all.py --skip TierHOIL

Paths and the LLM backend are configured via environment variables; see
README and `env_example.sh`.
"""
from __future__ import annotations
import argparse
import traceback
from typing import Dict, Any, List

from common import (
    load_jsonl, select_qids,
    DATASET_PATH, GRAPH_PATH, OUTPUT_DIR, run_batch, summarize,
)

from run_flat_holi import run_flat_holi_one
from run_tier_holi import run_tier_holi_one


# ──────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────
ALL_METHODS = ["FlatHOIL", "TierHOIL"]

# snake_case / case-insensitive aliases
_ALIASES: Dict[str, str] = {m.lower(): m for m in ALL_METHODS}
_ALIASES.update({
    "flat_holi":  "FlatHOIL",
    "tier_holi":  "TierHOIL",
    "flat":       "FlatHOIL",
    "tier":       "TierHOIL",
    "flathoil":   "FlatHOIL",
    "tierhoil":   "TierHOIL",
})


def _normalize_method_name(raw: str) -> str:
    if raw in ALL_METHODS:
        return raw
    return _ALIASES.get(raw.lower(), raw)


def _make_fn(name: str,
             dataset: Dict[int, Any],
             graph_data: Dict[int, Any],
             extra_kwargs: Dict[str, Any]):
    """Wrap a method to the (qid, sample, log_path, **kw) signature."""

    if name == "FlatHOIL":
        return lambda qid, sample, log_path, **kw: run_flat_holi_one(
            qid=qid, sample=sample,
            graph_row=graph_data.get(qid, {}),
            log_path=log_path,
            max_attempts=extra_kwargs.get("flat_max_attempts", 2),
            max_candidates_show=extra_kwargs.get("flat_max_candidates_show", 200),
            max_select=extra_kwargs.get("flat_max_select", 5),
            max_raw_paths_show=extra_kwargs.get("flat_max_raw_paths_show", 50),
        )

    if name == "TierHOIL":
        return lambda qid, sample, log_path, **kw: run_tier_holi_one(
            qid=qid, sample=sample,
            graph_row=graph_data.get(qid, {}),
            log_path=log_path,
            max_attempts=extra_kwargs.get("tier_max_attempts", 2),
            max_depth_limit=extra_kwargs.get("tier_max_depth_limit", 6),
            max_candidates_show=extra_kwargs.get("tier_max_candidates_show", 200),
            max_select=extra_kwargs.get("tier_max_select", 5),
            max_raw_paths_show=extra_kwargs.get("tier_max_raw_paths_show", 150),
            anonymized=extra_kwargs.get("tier_anonymized", False),
        )

    raise ValueError(f"Unknown method: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HOLI methods (Flat-HOLI, Tier-HOLI) on the KGQA benchmark."
    )
    parser.add_argument("--only", nargs="*", default=None,
                        help=f"Run only these methods. Choices: {ALL_METHODS} "
                             f"(snake_case / lowercase aliases supported)")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--qids", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)

    # ── Flat-HOLI hyperparameters ──
    parser.add_argument("--flat_max_attempts", type=int, default=2)
    parser.add_argument("--flat_max_candidates_show", type=int, default=200,
                        help="Flat-HOLI Step1 LLM batch size (relation chains "
                             "per LLM call).  Larger → fewer batches but a "
                             "longer prompt per call.")
    parser.add_argument("--flat_max_select", type=int, default=5)
    parser.add_argument("--flat_max_raw_paths_show", type=int, default=50,
                        help="Number of raw paths shown to the LLM at Step3.")

    # ── Tier-HOLI hyperparameters ──
    parser.add_argument("--tier_max_attempts", type=int, default=2)
    parser.add_argument("--tier_max_depth_limit", type=int, default=6,
                        help="Maximum search depth.  Default matches the "
                             "longest gold path in the benchmark.")
    parser.add_argument("--tier_max_candidates_show", type=int, default=200,
                        help="Tier-HOLI Step1 LLM batch size (path prefixes "
                             "per LLM call).")
    parser.add_argument("--tier_max_select", type=int, default=5)
    parser.add_argument("--tier_max_raw_paths_show", type=int, default=150)
    parser.add_argument("--tier_anonymized", action="store_true",
                        help="Use the anonymized-entity prompt block at Step3 "
                             "(only relevant when running on a graph whose "
                             "entity labels have been replaced by IDs).")

    args = parser.parse_args()

    # ── Resolve method list ────────────────────────────────────────
    if args.only:
        requested = [_normalize_method_name(x) for x in args.only]
        to_run = [m for m in requested if m in ALL_METHODS]
        unknown = [x for x in args.only
                   if _normalize_method_name(x) not in ALL_METHODS]
        if unknown:
            print(f"WARNING: unknown methods in --only: {unknown}.  "
                  f"Valid: {ALL_METHODS}")
    else:
        skip_norm = {_normalize_method_name(x) for x in args.skip}
        to_run = [m for m in ALL_METHODS if m not in skip_norm]

    if not to_run:
        print("No methods selected.  Exiting.")
        return

    print(f"Methods to run: {to_run}")

    # ── Load data ───────────────────────────────────────────────────
    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = load_jsonl(DATASET_PATH)
    print(f"  loaded {len(dataset)} questions")

    print(f"Loading graph from {GRAPH_PATH} ...")
    graph_data = load_jsonl(GRAPH_PATH)
    print(f"  loaded {len(graph_data)} graphs")

    qids: List[int] = select_qids(graph_data.keys(), only_qids=args.qids)
    if args.limit:
        qids = qids[:args.limit]
    print(f"Running on {len(qids)} questions")

    extra = vars(args)

    # ── Run ─────────────────────────────────────────────────────────
    results_by_method: Dict[str, str] = {}
    for name in to_run:
        print(f"\n{'='*60}\n[RUN] {name}\n{'='*60}")
        try:
            fn = _make_fn(name, dataset, graph_data, extra)
            csv_path = run_batch(name, fn, qids, dataset, args.output_dir)
            results_by_method[name] = csv_path
        except Exception as e:
            print(f"[{name}] FATAL ERROR: {e}")
            traceback.print_exc()
            continue

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}\nSUMMARY\n{'#'*60}")
    for method_name in results_by_method:
        summarize(method_name, args.output_dir)


if __name__ == "__main__":
    main()
