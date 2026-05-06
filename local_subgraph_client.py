"""
local_subgraph_client.py

"""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


# Type relation labels expected from the graph builder's saved type triples.
# Kept here so callers can reason about which relations represent typing
# information without having to import the builder's constants.
TYPE_RELATION_LABELS: Tuple[str, ...] = ("instance of", "subclass of")


def _parse_candidate_path(path_str: str) -> List[Tuple[str, str, str]]:
    """'A --> (rel) --> B --> (rel2) --> C' """
    parts = [p.strip() for p in path_str.split("-->") if p.strip()]
    triples = []
    for i in range(0, len(parts) - 2, 2):
        h = parts[i]
        rel_raw = parts[i + 1].strip()
        t = parts[i + 2]
        rel = rel_raw[1:-1].strip() if (rel_raw.startswith("(") and rel_raw.endswith(")")) else rel_raw
        if h and rel and t:
            triples.append((h, rel, t))
    return triples


# ─────────────────────────────────────────────────────────────────────
# Type-aware path display helpers (Tier-HOIL / Flat-HOIL )
# ─────────────────────────────────────────────────────────────────────
def _parse_path_into_seqs(path_str: str) -> Tuple[List[str], List[str]]:
    """
    'A --> (rel1) --> B --> (rel2) --> C' 
        entities = ['A', 'B', 'C'], relations = ['rel1', 'rel2'] .

    """
    parts = [p.strip() for p in path_str.split("-->") if p.strip()]
    entities: List[str] = []
    relations: List[str] = []
    for i, tok in enumerate(parts):
        if i % 2 == 0:
            entities.append(tok)
        else:
            rel = tok
            if rel.startswith("(") and rel.endswith(")"):
                rel = rel[1:-1].strip()
            relations.append(rel)
    return entities, relations


def format_typed_relation_chain(rel_chain: List[str],
                                candidate_paths: List[str],
                                type_lookup: Dict[str, str],
                                placeholder: str = "?",
                                unknown_label: str = "unknown") -> str:

    n = len(rel_chain)
    type_chains: List[Tuple[str, ...]] = []

    for path_str in candidate_paths:
        entities, relations = _parse_path_into_seqs(path_str)
        if len(relations) < n:
            continue
        if tuple(relations[:n]) != tuple(rel_chain):
            continue
        seq: List[str] = []
        for i in range(n + 1):
            if i < len(entities):
                t = type_lookup.get(entities[i], placeholder)
                if not t or t == placeholder:
                    t = unknown_label
                seq.append(t)
            else:
                seq.append(unknown_label)
        type_chains.append(tuple(seq))

    if not type_chains:
        chosen: Tuple[str, ...] = tuple([unknown_label] * (n + 1))
    else:
        cnt = Counter(type_chains)
        max_count = max(cnt.values())
        ties = sorted(t for t, c in cnt.items() if c == max_count)
        chosen = ties[0]   # alphabetical first

    parts = [f"[{chosen[0]}]"]
    for i, rel in enumerate(rel_chain):
        parts.append(f"({rel})")
        parts.append(f"[{chosen[i + 1]}]")
    return " --> ".join(parts)


def decorate_path_with_types(path_str: str,
                             type_lookup: Dict[str, str],
                             placeholder: str = "?") -> str:


    parts = [p.strip() for p in path_str.split("-->")]
    out: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            ent = part
            if not ent:
                out.append(part)
                continue
            t = type_lookup.get(ent, placeholder)
            if not t or t == placeholder:
                t = ent   # redundant self-type for visual consistency
            out.append(f"{ent}:[{t}]")
        else:
            out.append(part)
    return " --> ".join(out)


# ─────────────────────────────────────────────────────────────────────
# Module-level helpers for baselines that bypass query_all
# (KARPA, FiDeLiS read graph["candidate_paths"] directly)
# ─────────────────────────────────────────────────────────────────────
def make_type_triple_paths(graph_row: Dict[str, Any],
                           skip_placeholder: bool = True,
                           placeholder_label: str = "?") -> List[str]:

    paths: List[str] = []
    seen: Set[Tuple[str, str, str]] = set()
    graph = (graph_row or {}).get("graph", {}) or {}
    for t in graph.get("type_triples", []) or []:
        src = (t.get("src_label") or "").strip()
        rel = (t.get("rel_label") or "").strip()
        dst = (t.get("type_label") or "").strip()
        if not (src and rel and dst):
            continue
        if skip_placeholder and dst == placeholder_label:
            continue
        key = (src, rel, dst)
        if key in seen:
            continue
        seen.add(key)
        paths.append(f"{src} --> ({rel}) --> {dst}")
    return paths


def get_candidate_paths_with_types(graph_row: Dict[str, Any]) -> List[str]:

    graph = (graph_row or {}).get("graph", {}) or {}
    base = list(graph.get("candidate_paths", []) or [])
    return base + make_type_triple_paths(graph_row)


# ─────────────────────────────────────────────────────────────────────
# Main client
# ─────────────────────────────────────────────────────────────────────
class LocalSubgraphClient:

    def __init__(self, graph_data: Dict[int, Dict[str, Any]]):
        """
        Args:
            graph_data: {question_id: graph_row}
                graph_row = {"question_id": int,
                             "graph": {"start_graphs": [...],
                                       "candidate_paths": [...],
                                       "type_triples": [...],          # NEW
                                       "type_lookup_by_label": {...}}} # NEW
        """
        self._graph_data = graph_data
        self.current_qid: int = None
        # per-question state 
        self._adj_outgoing: Dict[str, List[Tuple[str, str]]] = {}
        self._adj_incoming: Dict[str, List[Tuple[str, str]]] = {}
        self._qid_to_label: Dict[str, str] = {}
        self._label_to_qid: Dict[str, str] = {}
        self._type_lookup_by_label: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # per-question subgraph
    # ------------------------------------------------------------------
    def set_question(self, qid: int) -> None:

        self.current_qid = qid
        self._adj_outgoing = {}
        self._adj_incoming = {}
        self._qid_to_label = {}
        self._label_to_qid = {}
        self._type_lookup_by_label = {}

        row = self._graph_data.get(qid, {}) or {}
        graph = row.get("graph", {}) or {}

        
        for sg in graph.get("start_graphs", []) or []:
            sq = (sg.get("start_qid") or "").strip()
            sl = (sg.get("start_label") or "").strip()
            if sq and sl:
                self._qid_to_label[sq] = sl
                self._label_to_qid[sl] = sq

        
        for sg in graph.get("start_graphs", []) or []:
            for path_str in sg.get("candidate_paths", []) or []:
                for (h, rel, t) in _parse_candidate_path(path_str):
                    self._add_edge(h, rel, t)


        for t in graph.get("type_triples", []) or []:
            src_label  = (t.get("src_label")  or "").strip()
            rel_label  = (t.get("rel_label")  or "").strip()
            type_label = (t.get("type_label") or "").strip()
            if not (src_label and rel_label and type_label):
                continue
            self._add_edge(src_label, rel_label, type_label)


            src_qid  = (t.get("src_qid")  or "").strip()
            type_qid = (t.get("type_qid") or "").strip()
            if src_qid:
                self._qid_to_label.setdefault(src_qid, src_label)
                self._label_to_qid.setdefault(src_label, src_qid)
            if type_qid:
                self._qid_to_label.setdefault(type_qid, type_label)
                self._label_to_qid.setdefault(type_label, type_qid)


        self._type_lookup_by_label = dict(graph.get("type_lookup_by_label", {}) or {})


        if not self._adj_outgoing and not self._qid_to_label:
            raise ValueError(
                f"LocalSubgraphClient: no subgraph for question_id={qid}"
            )

    def _add_edge(self, head: str, rel: str, tail: str) -> None:
        """outgoing + incoming index 에 (head -rel-> tail) 추가."""
        self._adj_outgoing.setdefault(head, [])
        if (rel, tail) not in self._adj_outgoing[head]:
            self._adj_outgoing[head].append((rel, tail))
        self._adj_incoming.setdefault(tail, [])
        if (rel, head) not in self._adj_incoming[tail]:
            self._adj_incoming[tail].append((rel, head))

    # ------------------------------------------------------------------
    # entity identifier resolution (qid ↔ label)
    # ------------------------------------------------------------------
    def _resolve_to_label(self, x: str) -> str:

        if x in self._qid_to_label:
            return self._qid_to_label[x]
        return x

    # ------------------------------------------------------------------
    # type lookup (explicit-use API for Tier-HOIL / Flat-HOIL)
    # ------------------------------------------------------------------
    def get_type(self, entity: str,
                 default: Optional[str] = None,
                 placeholder: str = "?") -> Optional[str]:

        label = self._resolve_to_label(entity)
        t = self._type_lookup_by_label.get(label)
        if t is None or t == placeholder:
            return default
        return t

    def all_type_lookup(self) -> Dict[str, str]:

        return dict(self._type_lookup_by_label)

    # ------------------------------------------------------------------
    # wiki_client.query_all 
    # ------------------------------------------------------------------
    def query_all(self, action: str, *args):
        if action == "get_all_relations_of_an_entity":
            entity = args[0]
            label = self._resolve_to_label(entity)
            head_rels = sorted({rel for rel, _ in self._adj_outgoing.get(label, [])})
            tail_rels = sorted({rel for rel, _ in self._adj_incoming.get(label, [])})
            return {"head": head_rels, "tail": tail_rels}

        if action == "label2pid":

            rel_label = args[0]
            return {rel_label}

        if action == "label2qid":
            lab = args[0]
            if lab in self._label_to_qid:
                return {self._label_to_qid[lab]}
            return {lab}   # label-as-qid

        if action == "get_tail_entities_given_head_and_relation":
            entity = args[0]
            rel_label = args[1]
            label = self._resolve_to_label(entity)

            tails = []
            for rel, t_label in self._adj_outgoing.get(label, []):
                if rel == rel_label:
                    tails.append({
                        "qid": self._label_to_qid.get(t_label, t_label),
                        "label": t_label,
                    })

            heads = []
            for rel, h_label in self._adj_incoming.get(label, []):
                if rel == rel_label:
                    heads.append({
                        "qid": self._label_to_qid.get(h_label, h_label),
                        "label": h_label,
                    })

            return {"head": heads, "tail": tails}

        if action == "get_tail_values_given_head_and_relation":

            return []

        if action == "qid2label":
            x = args[0]
            if x in self._qid_to_label:
                return {self._qid_to_label[x]}
            return {x}   


        return "Not Found!"

    # ------------------------------------------------------------------
    # to_topic_entity_dict 대체 (polymorphic dispatch)
    # ------------------------------------------------------------------
    def build_topic_entity_dict(self, entities: List[str]) -> Dict[str, str]:

        topic: Dict[str, str] = {}
        for e in entities:
            e = (e or "").strip()
            if not e:
                continue
            if e in self._qid_to_label:
             
                topic[e] = self._qid_to_label[e]
            elif e in self._label_to_qid:
          
                topic[self._label_to_qid[e]] = e
            else:
           
                topic[e] = e
        return topic


# --------------------------------------------------------------------------
# Singleton factory (wiki_client.py )
# --------------------------------------------------------------------------
_CACHED_INSTANCE: LocalSubgraphClient = None

def get_local_subgraph_client() -> LocalSubgraphClient:

    global _CACHED_INSTANCE
    if _CACHED_INSTANCE is None:

        from common import load_jsonl, GRAPH_PATH
        graph_data = load_jsonl(GRAPH_PATH)
        _CACHED_INSTANCE = LocalSubgraphClient(graph_data)
    return _CACHED_INSTANCE


# --------------------------------------------------------------------------
# Helper: graph_row  start entity label 
# --------------------------------------------------------------------------
def extract_topic_labels(graph_row: Dict[str, Any]) -> List[str]:

    labels: List[str] = []
    graph = (graph_row or {}).get("graph", {}) or {}
    for sg in graph.get("start_graphs", []) or []:
        lab = (sg.get("start_label") or "").strip()
        if lab and lab not in labels:
            labels.append(lab)
    return labels