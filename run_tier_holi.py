"""
run_tier_holi.py
================
Tier-HOLI: type-Hierarchy / Ontology-Informed reasoning with a per-depth
(tiered) loop.  At each depth d, the model selects the most promising
relation-path *prefixes* of length d (Step1), the corresponding raw
paths are matched and depth-truncated (Step2), and Step3 either commits
to an answer a/b/c/d/e or returns 'unsure' to advance to depth d+1.

Key design choices:
  - 'unsure' at Step3 is *intentional* and is the core control signal
    for advancing to a deeper search depth — do not treat it as failure.
  - At every depth, an attempt loop (max_attempts) lets the model swap
    in a different set of prefixes while excluding previously-tried ones.
  - Type decoration: relation-path prefixes shown to the LLM at Step1
    carry entity-type annotations ("[Drug] --> (treats) --> [Disease]");
    raw paths shown at Step3 carry per-entity types ("aspirin:[Drug]").

Runner-compatible entry point:
    run_tier_holi_one(qid, sample, graph_row, log_path, **kw) -> str
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from common import (
    chat, append_jsonl, now_str,
    LLM_TEMPERATURE_REASONING, LLM_MAX_TOKENS,
    run_batch, load_jsonl, select_qids,
    DATASET_PATH, GRAPH_PATH, OUTPUT_DIR,
)
from local_subgraph_client import (
    format_typed_relation_chain, decorate_path_with_types,
)


# ────────────────────────────────────────────────────────────────────
# Default 하이퍼파라미터
# ────────────────────────────────────────────────────────────────────
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_MAX_DEPTH_LIMIT = 6
STEP1_MAX_CANDIDATES_SHOW = 200            # batch 크기
STEP3_MAX_RAW_PATHS_SHOW = 150             # ver3: 100 → ver4: 150
DEFAULT_MAX_SELECT = 5                     # ver3: 3 → ver4: 5

PROMPT_LOG_CHARS = 4000


# ────────────────────────────────────────────────────────────────────
# Path parsing utilities (ver3 그대로)
# ────────────────────────────────────────────────────────────────────
def parse_path_to_relations(path_str: str) -> List[str]:
    parts = [p.strip() for p in path_str.split("-->")]
    relations = []
    for i in range(1, len(parts), 2):
        rel = parts[i].strip("() ").strip()
        relations.append(rel)
    return relations


def get_depth(path_str: str) -> int:
    return len(parse_path_to_relations(path_str))


def get_relation_paths(candidate_paths: List[str], depth: int) -> List[List[str]]:
    """주어진 depth의 relation-path prefix 집합을 정렬된 list로."""
    rel_paths: Set[Tuple[str, ...]] = set()
    for p in candidate_paths:
        rels = parse_path_to_relations(p)
        if len(rels) >= depth:
            rel_paths.add(tuple(rels[:depth]))
    return [list(x) for x in sorted(rel_paths)]


def truncate_path(path_str: str, depth: int) -> str:
    parts = [p.strip() for p in path_str.split("-->")]
    keep = 2 * depth + 1
    return " --> ".join(parts[:keep])


def format_relation_chain(rel_path: List[str]) -> str:
    return " --> ".join(rel_path)


def match_raw_paths(candidate_paths: List[str],
                    selected_rel_paths: List[List[str]]) -> List[str]:
    if not selected_rel_paths:
        return []
    matched: List[str] = []
    for p in candidate_paths:
        rels = parse_path_to_relations(p)
        for sel in selected_rel_paths:
            d = len(sel)
            if len(rels) < d:
                continue
            if tuple(rels[:d]) == tuple(sel):
                matched.append(truncate_path(p, d))
                break
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in matched:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ────────────────────────────────────────────────────────────────────
# Type-decoration helpers (ver4)
# ────────────────────────────────────────────────────────────────────
def _typed_chain_str(rel_chain: List[str],
                     candidate_paths: List[str],
                     type_lookup: Dict[str, str]) -> str:
    if candidate_paths and type_lookup:
        return format_typed_relation_chain(rel_chain, candidate_paths, type_lookup)
    return format_relation_chain(rel_chain)


def _typed_paths(raw_paths: List[str],
                 type_lookup: Dict[str, str]) -> List[str]:
    if type_lookup:
        return [decorate_path_with_types(p, type_lookup) for p in raw_paths]
    return list(raw_paths)


# ────────────────────────────────────────────────────────────────────
# Step1 helpers
# ────────────────────────────────────────────────────────────────────
def extract_step1_indices(text: str,
                          n_paths: int,
                          max_select: int = DEFAULT_MAX_SELECT) -> List[int]:
    nums = re.findall(r"\d+", text)
    cleaned: List[int] = []
    seen: Set[int] = set()
    for x in nums:
        idx = int(x) - 1
        if idx < 0 or idx >= n_paths:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        cleaned.append(idx)
        if len(cleaned) >= max_select:
            break
    return cleaned


def _build_step1_prompt(question: str,
                        batch_rel_paths: List[List[str]],
                        prev_selected_for_display: List[List[str]],
                        depth: int, max_depth: int,
                        candidate_paths: Optional[List[str]],
                        type_lookup: Optional[Dict[str, str]],
                        max_select: int,
                        is_consolidation: bool,
                        batch_idx: int,
                        total_batches: int) -> str:
    """Step1 LLM prompt 빌더."""
    cps = candidate_paths or []
    tl = type_lookup or {}

    rel_lines: List[str] = []
    for i, path in enumerate(batch_rel_paths):
        display = _typed_chain_str(path, cps, tl)
        rel_lines.append(f"{i+1}. [{len(path)}-hop prefix] {display}")
    rel_text = "\n".join(rel_lines)

    prev_text = ""
    if prev_selected_for_display:
        prev_lines = [
            f"- {_typed_chain_str(p, cps, tl)}"
            for p in prev_selected_for_display
        ]
        prev_text = ("\nPreviously selected paths at this depth "
                     "(DO NOT repeat):\n" + "\n".join(prev_lines))

    # ver4: notation legend
    type_note = (
        "\nNotation legend:\n"
        "- Text in (parentheses) denotes a RELATION between two adjacent entities.\n"
        "- Text in [square brackets] denotes the TYPE of an entity from the\n"
        "  knowledge graph (e.g. [Drug], [Disease], [Person]).  '[unknown]'\n"
        "  means the type was not recorded for that slot.\n"
        "- Use the entity types together with the relations to judge whether\n"
        "  a path's structure can answer the question at this depth.\n"
        if (cps and tl) else
        "\nNotation legend:\n"
        "- Text in (parentheses) denotes a RELATION between two adjacent entities.\n"
    )

    if is_consolidation:
        header = (
            "You previously reviewed multiple batches of candidate prefixes "
            "and short-listed the most promising ones from each batch.  "
            f"Now from the {len(batch_rel_paths)} finalist prefixes below "
            f"(all at the current depth = {depth}), pick the best up to "
            f"{max_select}."
        )
    elif total_batches > 1:
        header = (
            f"You are reviewing batch {batch_idx + 1} of {total_batches} "
            f"of candidate relation-path prefixes for the CURRENT depth "
            f"(depth = {depth}).  This batch contains "
            f"{len(batch_rel_paths)} prefixes.  After all batches are "
            f"reviewed, your selections will be merged.  Pick up to "
            f"{max_select} from THIS BATCH ONLY."
        )
    else:
        header = (
            f"There are {len(batch_rel_paths)} candidate relation-path "
            f"prefixes for the CURRENT depth (depth = {depth}).  Pick up "
            f"to {max_select} most promising ones."
        )

    prompt = f"""You are solving a knowledge graph reasoning problem.

Question:
{question}

Current search depth:
- current depth = {depth}
- deeper rounds may still happen up to depth = {max_depth}

{header}

Candidate relation-path prefixes:
{rel_text}
{type_note}{prev_text}

Important:
- The current depth may still be too shallow.
- If the current-depth evidence seems incomplete, that is okay: deeper rounds may reveal additional evidence later.
- So do NOT force a path just because you must pick something. Pick only genuinely promising prefixes for the current depth.
- Prefer unexplored paths.
- Do NOT repeat previously selected paths.
- Return ONLY the path numbers, separated by commas.
- Example output: 1,3,5
- If only 1 or 2 paths look promising, return fewer.
- If none look useful, return: none
- Do not explain your choice.
"""
    return prompt


def _step1_one_call(question: str,
                    batch_rel_paths: List[List[str]],
                    prev_selected_for_display: List[List[str]],
                    depth: int, max_depth: int,
                    candidate_paths: Optional[List[str]],
                    type_lookup: Optional[Dict[str, str]],
                    qid: int, attempt: int,
                    batch_idx: int, total_batches: int,
                    is_consolidation: bool,
                    log_path: str,
                    max_select: int = DEFAULT_MAX_SELECT
                    ) -> List[List[str]]:
    """한 번의 step1 LLM 호출."""
    prompt = _build_step1_prompt(
        question=question,
        batch_rel_paths=batch_rel_paths,
        prev_selected_for_display=prev_selected_for_display,
        depth=depth, max_depth=max_depth,
        candidate_paths=candidate_paths,
        type_lookup=type_lookup,
        max_select=max_select,
        is_consolidation=is_consolidation,
        batch_idx=batch_idx,
        total_batches=total_batches,
    )

    raw_text = chat(
        prompt,
        temperature=LLM_TEMPERATURE_REASONING,
        max_tokens=LLM_MAX_TOKENS,
    )

    low = raw_text.strip().lower()
    if low == "none":
        idxs: List[int] = []
    else:
        idxs = extract_step1_indices(
            raw_text,
            n_paths=len(batch_rel_paths),
            max_select=max_select,
        )
    selected = [batch_rel_paths[i] for i in idxs]

    cps = candidate_paths or []
    tl = type_lookup or {}
    selected_typed_chains = [_typed_chain_str(p, cps, tl) for p in selected]

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL",
            "depth": depth, "attempt": attempt, "step": "step1",
            "batch_idx": batch_idx, "total_batches": total_batches,
            "is_consolidation": is_consolidation,
            "n_paths_in_batch": len(batch_rel_paths),
            "raw_llm_output": raw_text,
            "parsed_indices_1based": [i + 1 for i in idxs],
            "selected_relation_paths": selected,
            "selected_typed_chains": selected_typed_chains,   # ver4
            "prompt_excerpt": prompt[:PROMPT_LOG_CHARS],      # ver4
            "prompt_total_chars": len(prompt),
        })
    return selected


# ────────────────────────────────────────────────────────────────────
# Step1 — BATCHED selection over current-depth prefixes
# ────────────────────────────────────────────────────────────────────
def step1_select_paths(question: str,
                       rel_paths: List[List[str]],
                       prev_selected: List[List[str]],
                       qid: int,
                       depth: int,
                       max_depth: int,
                       attempt: int,
                       log_path: str,
                       candidate_paths: Optional[List[str]] = None,
                       type_lookup: Optional[Dict[str, str]] = None,
                       max_candidates_show: int = STEP1_MAX_CANDIDATES_SHOW,
                       max_select: int = DEFAULT_MAX_SELECT
                       ) -> List[List[str]]:
    """Step1: 현재 depth 의 prefix 후보들 중 promising top-max_select 선택."""
    # 1) prev_selected 제외
    prev_set: Set[Tuple[str, ...]] = {tuple(p) for p in prev_selected}
    remaining: List[List[str]] = [p for p in rel_paths if tuple(p) not in prev_set]

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL",
            "depth": depth, "attempt": attempt, "step": "step1_setup",
            "n_total_rel_paths": len(rel_paths),
            "n_prev_selected_excluded": len(rel_paths) - len(remaining),
            "n_remaining": len(remaining),
            "batch_size": max_candidates_show,
            "max_select": max_select,
            "expected_batches": (len(remaining) + max_candidates_show - 1)
                                 // max(1, max_candidates_show)
                                 if remaining else 0,
        })

    if not remaining:
        return []

    # 2) batch 분할
    batches: List[List[List[str]]] = [
        remaining[i:i + max_candidates_show]
        for i in range(0, len(remaining), max_candidates_show)
    ]
    total_batches = len(batches)

    # 3) 단일 batch
    if total_batches == 1:
        return _step1_one_call(
            question=question,
            batch_rel_paths=batches[0],
            prev_selected_for_display=prev_selected,
            depth=depth, max_depth=max_depth,
            candidate_paths=candidate_paths,
            type_lookup=type_lookup,
            qid=qid, attempt=attempt,
            batch_idx=0, total_batches=1,
            is_consolidation=False,
            log_path=log_path,
            max_select=max_select,
        )

    # 4) Multi-batch
    finalists: List[List[str]] = []
    finalists_seen: Set[Tuple[str, ...]] = set()
    for b_idx, batch in enumerate(batches):
        sel = _step1_one_call(
            question=question,
            batch_rel_paths=batch,
            prev_selected_for_display=prev_selected,
            depth=depth, max_depth=max_depth,
            candidate_paths=candidate_paths,
            type_lookup=type_lookup,
            qid=qid, attempt=attempt,
            batch_idx=b_idx, total_batches=total_batches,
            is_consolidation=False,
            log_path=log_path,
            max_select=max_select,
        )
        for p in sel:
            t = tuple(p)
            if t not in finalists_seen:
                finalists_seen.add(t)
                finalists.append(p)

    if log_path:
        cps = candidate_paths or []
        tl = type_lookup or {}
        finalists_typed = [_typed_chain_str(p, cps, tl) for p in finalists[:10]]
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL",
            "depth": depth, "attempt": attempt, "step": "step1_finalists",
            "n_finalists": len(finalists),
            "finalists_preview": finalists[:10],
            "finalists_typed_preview": finalists_typed,    # ver4
        })

    if len(finalists) <= max_select:
        return finalists

    # 6) Consolidation
    return _step1_one_call(
        question=question,
        batch_rel_paths=finalists,
        prev_selected_for_display=prev_selected,
        depth=depth, max_depth=max_depth,
        candidate_paths=candidate_paths,
        type_lookup=type_lookup,
        qid=qid, attempt=attempt,
        batch_idx=0, total_batches=total_batches,
        is_consolidation=True,
        log_path=log_path,
        max_select=max_select,
    )


# ────────────────────────────────────────────────────────────────────
# Step2 — selected prefix와 매칭되는 raw path 추출 (depth로 truncate)
# ────────────────────────────────────────────────────────────────────
def run_step2(candidate_paths: List[str],
              selected_rel_paths: List[List[str]],
              qid: int, depth: int, attempt: int,
              log_path: str,
              type_lookup: Optional[Dict[str, str]] = None,
              ) -> List[str]:
    raw_paths = match_raw_paths(candidate_paths, selected_rel_paths)
    tl = type_lookup or {}

    if log_path:
        raw_typed_preview = _typed_paths(raw_paths[:30], tl)
        selected_typed_chains = [
            _typed_chain_str(p, candidate_paths, tl) for p in selected_rel_paths
        ]
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL",
            "depth": depth, "attempt": attempt, "step": "step2",
            "selected_relation_paths": selected_rel_paths,
            "selected_typed_chains": selected_typed_chains,    # ver4
            "num_raw_paths": len(raw_paths),
            "raw_paths_preview": raw_paths[:30],
            "raw_paths_typed_preview": raw_typed_preview,      # ver4
        })
    return raw_paths


# ────────────────────────────────────────────────────────────────────
# Step3 — depth-aware MCQ answer (commit OR 'unsure' for next depth)
# ────────────────────────────────────────────────────────────────────
def extract_step3_answer(text: str) -> str:
    low = text.strip().lower()
    if low in ["a", "b", "c", "d", "e", "unsure"]:
        return low
    m = re.match(r"^\s*(?:\*\*)?\s*([abcde])\s*[\.\)\:\-]?", low)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:answer|final answer|i choose|choose|pick|option)\s*(?:is|:)?\s*([abcde])\b", low)
    if m:
        return m.group(1)
    if "unsure" in low or "insufficient" in low or "not enough" in low:
        return "unsure"
    m = re.search(r"\b([abcde])\b", low)
    if m:
        return m.group(1)
    return "unsure"


# 익명화된 graph (E_XXXX) 일 때만 step3 prompt에 추가하는 안내 블록.
ANONYMIZED_NOTE = """\
Note: All entity names have been replaced with anonymous IDs (E_XXXX).
Do NOT try to interpret the meaning of IDs.
Instead, focus purely on structural matching:
- If an ID in the KG path appears in a choice option, treat that as direct evidence for that choice.
- If the path connects the question's IDs through relevant relations to a choice's ID, that is sufficient evidence to commit to an answer.
"""


def step3_answer(question: str,
                 choices: Dict[str, str],
                 raw_paths: List[str],
                 qid: int, depth: int, max_depth: int, attempt: int,
                 log_path: str,
                 type_lookup: Optional[Dict[str, str]] = None,
                 anonymized: bool = False,
                 max_raw_paths_show: int = STEP3_MAX_RAW_PATHS_SHOW) -> str:
    """
    Step3: depth-aware MCQ answer.

    ver4 변경:
      - notation legend 가 (parentheses)=relation, [brackets]=type 명시.
      - log 에 type-decorated raw path + prompt excerpt 저장.
      - 'unsure' 메커니즘은 유지 (depth-loop 의 핵심 신호).
    """
    tl = type_lookup or {}
    decorated_paths = _typed_paths(list(raw_paths[:max_raw_paths_show]), tl)
    path_text = "\n".join(f"- {p}" for p in decorated_paths)
    note_block = ANONYMIZED_NOTE if anonymized else ""

    type_note = (
        "Notation legend:\n"
        "- Text in (parentheses) denotes a RELATION (knowledge-graph predicate).\n"
        "- Text in [square brackets] following 'entity:[Type]' denotes the TYPE\n"
        "  of that entity (e.g. 'Aspirin:[Drug]').  When the type was not\n"
        "  recorded, the entity name itself is repeated in the brackets\n"
        "  (e.g. 'Cereblon:[Cereblon]') — treat that as 'type unknown'.\n\n"
        if tl else
        "Notation legend:\n"
        "- Text in (parentheses) denotes a RELATION (knowledge-graph predicate).\n\n"
    )

    prompt = f"""
Question:
{question}

Choices:
a. {choices['a']}
b. {choices['b']}
c. {choices['c']}
d. {choices['d']}
e. {choices['e']}

Current search depth:
- current depth = {depth}
- deeper rounds may still happen up to depth = {max_depth}

{note_block}{type_note}Current evidence from knowledge graph paths
(the answer entity may appear at any position along a path, including
intermediate nodes — not only at the end):
{path_text}

Decision rule:
- If the CURRENT evidence is already sufficient to justify one answer strongly, choose one option (a/b/c/d/e).
- If the CURRENT evidence is incomplete, ambiguous, or you think a deeper search could still change the answer, respond with: unsure
- Do NOT guess just because an answer choice looks plausible.
- Be conservative: deeper depths may reveal missing links.

Return format:
- Return the option letter first if you commit to an answer.
- Otherwise return exactly: unsure
"""

    raw_text = chat(
        prompt,
        temperature=LLM_TEMPERATURE_REASONING,
        max_tokens=LLM_MAX_TOKENS,
    )
    ans = extract_step3_answer(raw_text)

    if log_path:
        raw_typed_preview = _typed_paths(list(raw_paths[:10]), tl)
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL",
            "depth": depth, "attempt": attempt, "step": "step3",
            "raw_llm_output": raw_text,
            "parsed_answer": ans,
            "num_input_raw_paths": len(raw_paths),
            "num_paths_shown_to_llm": min(len(raw_paths), max_raw_paths_show),
            "raw_paths_preview": raw_paths[:10],
            "raw_paths_typed_preview": raw_typed_preview,      # ver4
            "prompt_excerpt": prompt[:PROMPT_LOG_CHARS],       # ver4
            "prompt_total_chars": len(prompt),
        })
    return ans


# ────────────────────────────────────────────────────────────────────
# Main entry point: single qid (run_batch가 호출)
# ────────────────────────────────────────────────────────────────────
def run_tier_holi_one(qid: int,
                          sample: Dict[str, Any],
                          graph_row: Dict[str, Any],
                          log_path: str = "",
                          max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                          max_depth_limit: int = DEFAULT_MAX_DEPTH_LIMIT,
                          max_candidates_show: int = STEP1_MAX_CANDIDATES_SHOW,
                          max_select: int = DEFAULT_MAX_SELECT,
                          max_raw_paths_show: int = STEP3_MAX_RAW_PATHS_SHOW,
                          anonymized: bool = False,
                          **_) -> str:
    """
    Returns: 'a'/'b'/'c'/'d'/'e' or 'fail'

    depth-loop + attempt-loop 구조는 ver3 그대로 유지.
    """
    graph = (graph_row or {}).get("graph", {}) or {}
    candidate_paths = graph.get("candidate_paths", [])
    type_lookup = graph.get("type_lookup_by_label", {}) or {}
    if not candidate_paths:
        if log_path:
            append_jsonl(log_path, {
                "timestamp": now_str(), "question_id": qid,
                "method": "TierHOIL", "step": "final",
                "prediction": "fail", "status": "no_candidate_paths",
            })
        return "fail"

    max_depth = min(max(get_depth(p) for p in candidate_paths), max_depth_limit)

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL", "step": "start",
            "version": "release-1.0",
            "question": sample.get("question", ""),
            "num_candidate_paths": len(candidate_paths),
            "max_depth": max_depth,
            "type_lookup_size": len(type_lookup),
            "max_attempts": max_attempts,
            "max_select": max_select,
            "max_candidates_show": max_candidates_show,
            "max_raw_paths_show": max_raw_paths_show,
        })

    for depth in range(1, max_depth + 1):
        rel_paths = get_relation_paths(candidate_paths, depth)
        if log_path:
            append_jsonl(log_path, {
                "timestamp": now_str(), "question_id": qid,
                "method": "TierHOIL",
                "depth": depth, "step": "depth_start",
                "num_relation_paths": len(rel_paths),
            })
        if not rel_paths:
            continue

        prev_selected: List[List[str]] = []
        for attempt in range(1, max_attempts + 1):
            selected = step1_select_paths(
                question=sample["question"],
                rel_paths=rel_paths,
                prev_selected=prev_selected,
                qid=qid, depth=depth, max_depth=max_depth,
                attempt=attempt, log_path=log_path,
                candidate_paths=candidate_paths,
                type_lookup=type_lookup,
                max_candidates_show=max_candidates_show,
                max_select=max_select,
            )
            if not selected:
                continue
            prev_selected.extend(selected)

            raw_paths = run_step2(
                candidate_paths=candidate_paths,
                selected_rel_paths=selected,
                qid=qid, depth=depth, attempt=attempt,
                log_path=log_path,
                type_lookup=type_lookup,
            )
            if not raw_paths:
                continue

            ans = step3_answer(
                question=sample["question"],
                choices={k: sample[k] for k in "abcde"},
                raw_paths=raw_paths,
                qid=qid, depth=depth, max_depth=max_depth,
                attempt=attempt, log_path=log_path,
                type_lookup=type_lookup,
                anonymized=anonymized,
                max_raw_paths_show=max_raw_paths_show,
            )
            if ans in ("a", "b", "c", "d", "e"):
                if log_path:
                    append_jsonl(log_path, {
                        "timestamp": now_str(), "question_id": qid,
                        "method": "TierHOIL",
                        "depth": depth, "attempt": attempt, "step": "final",
                        "prediction": ans, "status": "done",
                    })
                return ans
            # 'unsure' → 다음 attempt 로

    # 모든 depth 실패
    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "TierHOIL", "step": "final",
            "prediction": "fail", "status": "exhausted",
        })
    return "fail"


# ────────────────────────────────────────────────────────────────────
# Standalone CLI
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max_depth_limit", type=int, default=DEFAULT_MAX_DEPTH_LIMIT)
    parser.add_argument("--max_candidates_show", type=int,
                        default=STEP1_MAX_CANDIDATES_SHOW)
    parser.add_argument("--max_select", type=int, default=DEFAULT_MAX_SELECT)
    parser.add_argument("--max_raw_paths_show", type=int,
                        default=STEP3_MAX_RAW_PATHS_SHOW)
    parser.add_argument("--anonymized", action="store_true",
                        help="익명화된 graph (E_XXXX)일 때 step3 prompt에 안내 추가")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_jsonl(DATASET_PATH)
    print(f"Loading graph:   {GRAPH_PATH}")
    graph_data = load_jsonl(GRAPH_PATH)
    print(f"  dataset={len(dataset)}, graph={len(graph_data)}")

    qids = select_qids(graph_data.keys(), only_qids=args.qids)
    if args.limit:
        qids = qids[:args.limit]
    print(f"Running TierHOIL on {len(qids)} questions")
    print(f"  max_attempts={args.max_attempts}  max_depth={args.max_depth_limit}  "
          f"max_select={args.max_select}  max_candidates_show={args.max_candidates_show}  "
          f"max_raw_paths_show={args.max_raw_paths_show}")

    def _fn(qid, sample, log_path, **kw):
        return run_tier_holi_one(
            qid=qid, sample=sample,
            graph_row=graph_data.get(qid, {}),
            log_path=log_path,
            max_attempts=args.max_attempts,
            max_depth_limit=args.max_depth_limit,
            max_candidates_show=args.max_candidates_show,
            max_select=args.max_select,
            max_raw_paths_show=args.max_raw_paths_show,
            anonymized=args.anonymized,
        )

    run_batch("TierHOIL", _fn, qids, dataset, OUTPUT_DIR)
    
# python run_tier_holi.py --max_attempts 2 --max_select 7 --max_depth_limit 6 --max_candidates_show 200 --max_raw_paths_show 150 