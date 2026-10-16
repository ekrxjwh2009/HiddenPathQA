"""
run_flat_holi.py
================
Flat-HOLI: type-Hierarchy / Ontology-Informed reasoning over a flat set
of candidate relation chains (no depth loop).  All candidate paths are
considered together in a single Step1 batch (with consolidation when the
batch overflows), then matched raw paths are scored in Step3.

Key design choices:
  - No 'unsure' escape in Step3: the model must commit to a/b/c/d/e.
    This avoids the well-known 'e' / abstain bias when the LLM is
    uncertain.
  - Type decoration: relation chains shown to the LLM at Step1 carry
    entity-type annotations ("[Drug] --> (treats) --> [Disease]"); raw
    paths shown at Step3 carry per-entity types ("aspirin:[Drug]").
  - Retry-on-uncertainty: if Step3 returns 'e' or 'unsure', restart from
    Step1 with the previously-selected paths excluded ("DO NOT repeat").
    Up to `max_attempts` total.  The last-attempt answer is committed
    as-is even if still 'e'/'unsure'.

Runner-compatible entry point:
    run_flat_holi_one(qid, sample, graph_row, log_path, **kw) -> str
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
DEFAULT_MAX_CANDIDATES_SHOW = 200          # batch 크기 (rel-chain 개수)
DEFAULT_MAX_SELECT = 5                     # ver3: 3 → ver4: 5
DEFAULT_MAX_RAW_PATHS_SHOW = 50            # ver3: 30 → ver4: 50

# Prompt 본문에서 발췌하여 log 에 저장할 최대 글자수
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


def get_all_relation_paths(candidate_paths: List[str]) -> List[List[str]]:
    """모든 candidate path의 unique relation chain 추출 (depth별 분리 X)."""
    rel_paths: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()
    for p in candidate_paths:
        rels = tuple(parse_path_to_relations(p))
        if not rels or rels in seen:
            continue
        seen.add(rels)
        rel_paths.append(list(rels))
    return rel_paths


def format_relation_chain(rel_path: List[str]) -> str:
    return " --> ".join(rel_path)


def match_raw_paths(candidate_paths: List[str],
                    selected_rel_paths: List[List[str]]) -> List[str]:
    """선택된 relation chain과 매칭되는 raw path를 truncate 없이 반환."""
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
                matched.append(p)   # truncate X — full raw path
                break
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in matched:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ────────────────────────────────────────────────────────────────────
# Step1 helpers
# ────────────────────────────────────────────────────────────────────
def extract_step1_indices(text: str,
                          n_paths: int,
                          max_select: int = DEFAULT_MAX_SELECT) -> List[int]:
    """LLM 응답에서 1-based index 들을 0-based 로 파싱."""
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


def _typed_chain_str(rel_chain: List[str],
                     candidate_paths: List[str],
                     type_lookup: Dict[str, str]) -> str:
    """type-decorated chain 을 만들거나 (type_lookup 없으면) plain string 반환."""
    if candidate_paths and type_lookup:
        return format_typed_relation_chain(rel_chain, candidate_paths, type_lookup)
    return format_relation_chain(rel_chain)


def _typed_paths(raw_paths: List[str],
                 type_lookup: Dict[str, str]) -> List[str]:
    if type_lookup:
        return [decorate_path_with_types(p, type_lookup) for p in raw_paths]
    return list(raw_paths)


def _build_step1_prompt(question: str,
                        batch_rel_paths: List[List[str]],
                        prev_selected_for_display: List[List[str]],
                        candidate_paths: Optional[List[str]],
                        type_lookup: Optional[Dict[str, str]],
                        max_select: int,
                        is_consolidation: bool,
                        batch_idx: int,
                        total_batches: int,
                        attempt: int,
                        max_attempts: int) -> str:
    """
    Step1 LLM prompt 빌더.  단일 호출 / batch / consolidation 모두 동일 포맷.
    ver4 변경:
      - notation legend 가 (parentheses)=relation, [brackets]=entity type
        을 명시.
      - retry attempt 인 경우 attempt 번호와 prev_selected 가 강조됨.
    """
    cps = candidate_paths or []
    tl = type_lookup or {}

    rel_lines: List[str] = []
    for i, path in enumerate(batch_rel_paths):
        display = _typed_chain_str(path, cps, tl)
        rel_lines.append(f"{i+1}. [{len(path)}-hop] {display}")
    rel_text = "\n".join(rel_lines)

    prev_text = ""
    if prev_selected_for_display:
        prev_lines = [
            f"- {_typed_chain_str(p, cps, tl)}"
            for p in prev_selected_for_display
        ]
        prev_text = (
            "\nPreviously selected paths from earlier attempts (DO NOT repeat these "
            "and DO NOT pick paths that share the same relation prefix unless\n"
            "you have a strong new reason — earlier attempts already explored them):\n"
            + "\n".join(prev_lines)
        )

    # ver4: notation legend.  (괄호) = relation, [대괄호] = entity type.
    type_note = (
        "\nNotation legend:\n"
        "- Text in (parentheses) denotes a RELATION between two adjacent entities.\n"
        "- Text in [square brackets] denotes the TYPE of an entity from the\n"
        "  knowledge graph (e.g. [Drug], [Disease], [Person]).  '[unknown]'\n"
        "  means the type was not recorded for that slot.\n"
        "- Use the entity types together with the relations to judge whether\n"
        "  a path's structure can answer the question.\n"
        if (cps and tl) else
        "\nNotation legend:\n"
        "- Text in (parentheses) denotes a RELATION between two adjacent entities.\n"
    )

    # retry 안내 (attempt > 1)
    retry_banner = ""
    if attempt > 1:
        retry_banner = (
            f"\n[RETRY ATTEMPT {attempt}/{max_attempts}] "
            f"The previous attempt's paths did not lead to a confident answer "
            f"(the LLM either replied 'e' or 'unsure').  This time, pick a "
            f"DIFFERENT set of relation paths that explore other parts of the "
            f"knowledge graph.\n"
        )

    if is_consolidation:
        header = (
            "You previously reviewed multiple batches of candidate paths "
            "and short-listed the most promising ones from each batch.  "
            f"Now from the {len(batch_rel_paths)} finalist paths below, "
            f"pick the best up to {max_select} for answering the question."
        )
    elif total_batches > 1:
        header = (
            f"You are reviewing batch {batch_idx + 1} of {total_batches} "
            f"of candidate relation paths.  This batch contains "
            f"{len(batch_rel_paths)} paths.  After all batches are "
            f"reviewed, your selections will be merged.  Pick up to "
            f"{max_select} from THIS BATCH ONLY."
        )
    else:
        header = (
            f"There are {len(batch_rel_paths)} candidate relation paths.  "
            f"Pick up to {max_select} most promising ones."
        )

    prompt = f"""You are solving a knowledge graph reasoning problem.

Question:
{question}
{retry_banner}
{header}

Candidate relation paths:
{rel_text}
{type_note}{prev_text}

Rules:
- The paths may have different hop lengths.
- Prefer semantically relevant paths.
- Do NOT repeat previously selected paths.
- Return ONLY the path numbers, separated by commas.
- Example output: 1,3,5
- If only 1 or 2 paths seem promising, return fewer.
- Do not explain your choice.
"""
    return prompt


def _step1_one_call(question: str,
                    batch_rel_paths: List[List[str]],
                    prev_selected_for_display: List[List[str]],
                    candidate_paths: Optional[List[str]],
                    type_lookup: Optional[Dict[str, str]],
                    qid: int, attempt: int,
                    batch_idx: int, total_batches: int,
                    is_consolidation: bool,
                    log_path: str,
                    max_select: int = DEFAULT_MAX_SELECT,
                    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                    ) -> List[List[str]]:
    """한 번의 step1 LLM 호출.  Returns: selected rel_paths."""
    prompt = _build_step1_prompt(
        question=question,
        batch_rel_paths=batch_rel_paths,
        prev_selected_for_display=prev_selected_for_display,
        candidate_paths=candidate_paths,
        type_lookup=type_lookup,
        max_select=max_select,
        is_consolidation=is_consolidation,
        batch_idx=batch_idx,
        total_batches=total_batches,
        attempt=attempt,
        max_attempts=max_attempts,
    )

    raw_text = chat(
        prompt,
        temperature=LLM_TEMPERATURE_REASONING,
        max_tokens=LLM_MAX_TOKENS,
    )
    idxs = extract_step1_indices(
        raw_text,
        n_paths=len(batch_rel_paths),
        max_select=max_select,
    )
    selected = [batch_rel_paths[i] for i in idxs]

    # ver4: type-decorated representation 도 같이 저장 (LLM 이 실제 본 것)
    cps = candidate_paths or []
    tl = type_lookup or {}
    selected_typed_chains = [_typed_chain_str(p, cps, tl) for p in selected]

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "FlatHOIL",
            "attempt": attempt, "step": "step1",
            "batch_idx": batch_idx, "total_batches": total_batches,
            "is_consolidation": is_consolidation,
            "n_paths_in_batch": len(batch_rel_paths),
            "raw_llm_output": raw_text,
            "parsed_indices_1based": [i + 1 for i in idxs],
            "selected_relation_paths": selected,           # raw list-of-lists
            "selected_typed_chains": selected_typed_chains, # ver4: typed display
            "prompt_excerpt": prompt[:PROMPT_LOG_CHARS],   # ver4: audit
            "prompt_total_chars": len(prompt),
        })
    return selected


# ────────────────────────────────────────────────────────────────────
# Step1 — BATCHED selection over all rel_paths
# ────────────────────────────────────────────────────────────────────
def step1_select_paths(question: str,
                       rel_paths: List[List[str]],
                       prev_selected: List[List[str]],
                       qid: int, attempt: int,
                       log_path: str,
                       candidate_paths: Optional[List[str]] = None,
                       type_lookup: Optional[Dict[str, str]] = None,
                       max_candidates_show: int = DEFAULT_MAX_CANDIDATES_SHOW,
                       max_select: int = DEFAULT_MAX_SELECT,
                       max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                       ) -> List[List[str]]:
    """
    Step1: rel_paths 에서 가장 promising 한 chain 최대 max_select 개 선택.

    ver4 변경:
      - prev_selected 누적 처리는 ver3 그대로지만, retry attempt 일 때
        prompt 에서 prev_selected 를 "이번에는 다른 걸 골라라" 라고 강조.
    """
    # 1) prev_selected 제외 (tuple 기반 비교)
    prev_set: Set[Tuple[str, ...]] = {tuple(p) for p in prev_selected}
    remaining: List[List[str]] = [p for p in rel_paths if tuple(p) not in prev_set]

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "FlatHOIL",
            "attempt": attempt, "step": "step1_setup",
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

    # 3) 단일 batch — 단일 호출
    if total_batches == 1:
        return _step1_one_call(
            question=question,
            batch_rel_paths=batches[0],
            prev_selected_for_display=prev_selected,
            candidate_paths=candidate_paths,
            type_lookup=type_lookup,
            qid=qid, attempt=attempt,
            batch_idx=0, total_batches=1,
            is_consolidation=False,
            log_path=log_path,
            max_select=max_select,
            max_attempts=max_attempts,
        )

    # 4) Multi-batch — finalists 수집
    finalists: List[List[str]] = []
    finalists_seen: Set[Tuple[str, ...]] = set()
    for b_idx, batch in enumerate(batches):
        sel = _step1_one_call(
            question=question,
            batch_rel_paths=batch,
            prev_selected_for_display=prev_selected,
            candidate_paths=candidate_paths,
            type_lookup=type_lookup,
            qid=qid, attempt=attempt,
            batch_idx=b_idx, total_batches=total_batches,
            is_consolidation=False,
            log_path=log_path,
            max_select=max_select,
            max_attempts=max_attempts,
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
            "method": "FlatHOIL",
            "attempt": attempt, "step": "step1_finalists",
            "n_finalists": len(finalists),
            "finalists_preview": finalists[:10],
            "finalists_typed_preview": finalists_typed,    # ver4
        })

    # 5) finalists 가 max_select 이하면 consolidation 불필요
    if len(finalists) <= max_select:
        return finalists

    # 6) Consolidation
    return _step1_one_call(
        question=question,
        batch_rel_paths=finalists,
        prev_selected_for_display=prev_selected,
        candidate_paths=candidate_paths,
        type_lookup=type_lookup,
        qid=qid, attempt=attempt,
        batch_idx=0, total_batches=total_batches,
        is_consolidation=True,
        log_path=log_path,
        max_select=max_select,
        max_attempts=max_attempts,
    )


# ────────────────────────────────────────────────────────────────────
# Step2 — selected chain 매칭 raw path
# ────────────────────────────────────────────────────────────────────
def run_step2(candidate_paths: List[str],
              selected_rel_paths: List[List[str]],
              qid: int, attempt: int, log_path: str,
              type_lookup: Optional[Dict[str, str]] = None,
              ) -> List[str]:
    raw_paths = match_raw_paths(candidate_paths, selected_rel_paths)
    tl = type_lookup or {}

    if log_path:
        # ver4: type-decorated raw path preview 추가
        raw_typed_preview = _typed_paths(raw_paths[:30], tl)
        selected_typed_chains = [
            _typed_chain_str(p, candidate_paths, tl) for p in selected_rel_paths
        ]
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "FlatHOIL",
            "attempt": attempt, "step": "step2",
            "selected_relation_paths": selected_rel_paths,
            "selected_typed_chains": selected_typed_chains,   # ver4
            "num_raw_paths": len(raw_paths),
            "raw_paths_preview": raw_paths[:30],
            "raw_paths_typed_preview": raw_typed_preview,     # ver4
        })
    return raw_paths


# ────────────────────────────────────────────────────────────────────
# Step3 — MCQ answer (commit; no 'unsure' escape in prompt)
# ────────────────────────────────────────────────────────────────────
def extract_step3_answer(text: str) -> str:
    """
    ver4: prompt 에서 'unsure' 옵션을 제거했지만, LLM 이 자발적으로 unsure-style
    응답을 할 수도 있어서 fallback parser 는 'unsure' 도 인식.  단 main loop 에서
    'unsure' / 'e' 모두 retry trigger 로 처리.
    """
    low = text.strip().lower()
    if low in ["a", "b", "c", "d", "e"]:
        return low
    if low in ["unsure", "none", "n/a"]:
        return "unsure"
    m = re.match(r"^\s*(?:\*\*)?\s*([abcde])\s*[\.\)\:\-]?", low)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:answer|final answer|i choose|choose|pick|option)\s*(?:is|:)?\s*([abcde])\b", low)
    if m:
        return m.group(1)
    if "unsure" in low or "insufficient" in low or "not enough" in low:
        return "unsure"
    # 마지막 fallback: 첫 등장하는 a/b/c/d/e 한 글자
    m = re.search(r"\b([abcde])\b", low)
    if m:
        return m.group(1)
    return "unsure"


def step3_answer(question: str,
                 choices: Dict[str, str],
                 raw_paths: List[str],
                 qid: int, attempt: int,
                 log_path: str,
                 type_lookup: Optional[Dict[str, str]] = None,
                 max_raw_paths_show: int = DEFAULT_MAX_RAW_PATHS_SHOW,
                 ) -> str:
    """
    Step3: 최종 답변.

    ver4 변경:
      - "If insufficient, answer: unsure" 줄 삭제.
      - "You must commit to exactly one option" 강조.
      - notation legend 에 (parentheses)=relation, [brackets]=type 명시.
      - log 에 type-decorated raw path 와 prompt excerpt 저장.
    """
    tl = type_lookup or {}
    decorated_paths = _typed_paths(list(raw_paths[:max_raw_paths_show]), tl)
    path_text = "\n".join(f"- {p}" for p in decorated_paths)

    # ver4: notation legend
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

    # ver4: 'If insufficient, answer: unsure' 줄 삭제, commit 강조
    prompt = f"""
Question:
{question}

Choices:
a. {choices['a']}
b. {choices['b']}
c. {choices['c']}
d. {choices['d']}
e. {choices['e']}

{type_note}Knowledge graph paths (the answer entity may appear at any position
along a path, including intermediate nodes — not only at the end):
{path_text}

Decision rule:
- You MUST commit to exactly one option among a / b / c / d / e.
- Pick the choice whose entity (or its synonym/parent class) best matches
  the entities and types observed along the paths above.
- The answer entity may be at an intermediate hop; the path does not need
  to end at the answer.
- Do NOT explain.  Return only the option letter (a, b, c, d, or e).
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
            "method": "FlatHOIL",
            "attempt": attempt, "step": "step3",
            "raw_llm_output": raw_text,
            "parsed_answer": ans,
            "num_input_raw_paths": len(raw_paths),
            "num_paths_shown_to_llm": min(len(raw_paths), max_raw_paths_show),
            "raw_paths_preview": raw_paths[:10],
            "raw_paths_typed_preview": raw_typed_preview,    # ver4
            "prompt_excerpt": prompt[:PROMPT_LOG_CHARS],     # ver4
            "prompt_total_chars": len(prompt),
        })
    return ans


# ────────────────────────────────────────────────────────────────────
# Main entry point: single qid (run_batch가 호출)
# ────────────────────────────────────────────────────────────────────
def run_flat_holi_one(qid: int,
                      sample: Dict[str, Any],
                      graph_row: Dict[str, Any],
                      log_path: str = "",
                      max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      max_candidates_show: int = DEFAULT_MAX_CANDIDATES_SHOW,
                      max_select: int = DEFAULT_MAX_SELECT,
                      max_raw_paths_show: int = DEFAULT_MAX_RAW_PATHS_SHOW,
                      **_) -> str:
    """
    Returns: 'a' / 'b' / 'c' / 'd' / 'e'   (no 'fail' under normal operation)

    ver4 main-loop 흐름 (사용자 요청 (5) 반영):
      attempt = 1
      while attempt <= max_attempts:
          step1 → step2 → step3
          if step3 ans ∈ {a, b, c, d}:
              return ans                       # confident commit
          if step3 ans ∈ {e, unsure}:
              if attempt < max_attempts:
                  prev_selected ← prev_selected ∪ this_attempt_paths
                  attempt += 1                 # retry with different paths
                  continue
              else:
                  return ans (or 'e' if 'unsure')  # last-attempt commit
          # step1/2 가 비어 있으면 다음 attempt
    """
    graph = (graph_row or {}).get("graph", {}) or {}
    candidate_paths = graph.get("candidate_paths", [])
    type_lookup = graph.get("type_lookup_by_label", {}) or {}
    if not candidate_paths:
        if log_path:
            append_jsonl(log_path, {
                "timestamp": now_str(), "question_id": qid,
                "method": "FlatHOIL", "step": "final",
                "prediction": "fail", "status": "no_candidate_paths",
            })
        return "fail"

    rel_paths = get_all_relation_paths(candidate_paths)
    if not rel_paths:
        if log_path:
            append_jsonl(log_path, {
                "timestamp": now_str(), "question_id": qid,
                "method": "FlatHOIL", "step": "final",
                "prediction": "fail", "status": "no_relation_paths",
            })
        return "fail"

    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "FlatHOIL", "step": "start",
            "version": "release-1.0",
            "question": sample.get("question", ""),
            "num_candidate_paths": len(candidate_paths),
            "num_unique_relation_paths": len(rel_paths),
            "type_lookup_size": len(type_lookup),
            "max_attempts": max_attempts,
            "max_select": max_select,
            "max_candidates_show": max_candidates_show,
            "max_raw_paths_show": max_raw_paths_show,
        })

    prev_selected: List[List[str]] = []
    last_ans: str = "unsure"     # 마지막 attempt 의 step3 답을 fallback 으로

    for attempt in range(1, max_attempts + 1):
        selected = step1_select_paths(
            question=sample["question"],
            rel_paths=rel_paths,
            prev_selected=prev_selected,
            qid=qid, attempt=attempt, log_path=log_path,
            candidate_paths=candidate_paths,
            type_lookup=type_lookup,
            max_candidates_show=max_candidates_show,
            max_select=max_select,
            max_attempts=max_attempts,
        )
        if not selected:
            # step1 가 아무것도 못 골랐으면 retry 의 의미가 없음 → 다음 attempt
            # (단 prev_selected 에 추가할 것이 없으므로 동일 결과 반복 가능
            #  → 이 경우 그대로 break)
            if log_path:
                append_jsonl(log_path, {
                    "timestamp": now_str(), "question_id": qid,
                    "method": "FlatHOIL",
                    "attempt": attempt, "step": "step1_empty",
                    "note": "step1 returned no path; breaking attempt loop",
                })
            break

        # 다음 attempt 에서 같은 path 못 고르도록 누적
        prev_selected.extend(selected)

        raw_paths = run_step2(
            candidate_paths=candidate_paths,
            selected_rel_paths=selected,
            qid=qid, attempt=attempt, log_path=log_path,
            type_lookup=type_lookup,
        )
        if not raw_paths:
            if log_path:
                append_jsonl(log_path, {
                    "timestamp": now_str(), "question_id": qid,
                    "method": "FlatHOIL",
                    "attempt": attempt, "step": "step2_empty",
                    "note": "no raw paths matched; trying next attempt",
                })
            continue

        ans = step3_answer(
            question=sample["question"],
            choices={k: sample[k] for k in "abcde"},
            raw_paths=raw_paths,
            qid=qid, attempt=attempt, log_path=log_path,
            type_lookup=type_lookup,
            max_raw_paths_show=max_raw_paths_show,
        )
        last_ans = ans

        # ver4 핵심: 'a/b/c/d' → commit, 'e' or 'unsure' → retry (마지막 attempt 제외)
        if ans in ("a", "b", "c", "d"):
            if log_path:
                append_jsonl(log_path, {
                    "timestamp": now_str(), "question_id": qid,
                    "method": "FlatHOIL",
                    "attempt": attempt, "step": "final",
                    "prediction": ans, "status": "committed_confident",
                })
            return ans

        # ans ∈ {'e', 'unsure'}
        if attempt < max_attempts:
            if log_path:
                append_jsonl(log_path, {
                    "timestamp": now_str(), "question_id": qid,
                    "method": "FlatHOIL",
                    "attempt": attempt, "step": "retry_trigger",
                    "this_attempt_answer": ans,
                    "n_paths_blocked_for_next_attempt": len(prev_selected),
                    "note": ("retrying step1 with prev_selected exclusion "
                             "because answer was 'e' or 'unsure'"),
                })
            continue
        # 마지막 attempt → commit 강제
        final_ans = ans if ans in ("a", "b", "c", "d", "e") else "e"
        if log_path:
            append_jsonl(log_path, {
                "timestamp": now_str(), "question_id": qid,
                "method": "FlatHOIL",
                "attempt": attempt, "step": "final",
                "prediction": final_ans,
                "status": "committed_last_attempt",
                "note": ("answer was 'e' or 'unsure' on last attempt; "
                         "committing as-is (no further retries)"),
            })
        return final_ans

    # 정상적으로 step3 도달 못한 경우 (step1 empty 등) → last_ans 가 'unsure'
    final_ans = last_ans if last_ans in ("a", "b", "c", "d", "e") else "e"
    if log_path:
        append_jsonl(log_path, {
            "timestamp": now_str(), "question_id": qid,
            "method": "FlatHOIL", "step": "final",
            "prediction": final_ans,
            "status": "exhausted_no_step3",
            "note": "loop exited without reaching step3 successfully",
        })
    return final_ans


# ────────────────────────────────────────────────────────────────────
# Standalone CLI
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max_candidates_show", type=int,
                        default=DEFAULT_MAX_CANDIDATES_SHOW)
    parser.add_argument("--max_select", type=int, default=DEFAULT_MAX_SELECT)
    parser.add_argument("--max_raw_paths_show", type=int,
                        default=DEFAULT_MAX_RAW_PATHS_SHOW)
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
    print(f"Running FlatHOIL on {len(qids)} questions")
    print(f"  max_attempts={args.max_attempts}  max_select={args.max_select}  "
          f"max_candidates_show={args.max_candidates_show}  "
          f"max_raw_paths_show={args.max_raw_paths_show}")

    def _fn(qid, sample, log_path, **kw):
        return run_flat_holi_one(
            qid=qid, sample=sample,
            graph_row=graph_data.get(qid, {}),
            log_path=log_path,
            max_attempts=args.max_attempts,
            max_candidates_show=args.max_candidates_show,
            max_select=args.max_select,
            max_raw_paths_show=args.max_raw_paths_show,
        )

    run_batch("FlatHOIL", _fn, qids, dataset, OUTPUT_DIR)
    
    
# python run_flat_holi.py --max_attempts 2 --max_select 7 --max_candidates_show 200 --max_raw_paths_show 70 
