from collections import defaultdict
from typing import List, Set, Dict, Tuple, Union, Sequence, Iterable
import math
import numpy as np
from sympy.physics.vector.tests.test_printing import alpha

from knowledge_graph.ic_metrics import resnik_bma, _mean_ic
from knowledge_graph.ner import ClinicalEntityLinker


def dist(
    target: Set[str],
    neg_target: Set[str],
    valid_refs: List[Set[str]],
    neg_valid_refs: List[Set[str]],
    depth_sets: List[Set[str]],
    max_depth: int,
    set_to_ids: Dict[frozenset, Set[str]],
    linker: ClinicalEntityLinker,
    reachable: Set[str],
    wrapper,
    dist_from_t,
    *,
    type: str = "umls",
    **kwargs,
) -> List[Tuple[str, Set[str], Union[int, float]]]:
    """
    Generic distance dispatcher.
    Returns a list of (reference_id, reference_CUI_set, distance)
    Extra hyper‑parameters are forwarded via **kwargs.
    """
    if type == "umls":
        return umls_distance(target, valid_refs, depth_sets, 10, set_to_ids,linker, reachable, dist_from_t, wrapper.cui_to_vid)
    elif type == "jac":
        return jaccard_distance(target, valid_refs, set_to_ids)
    elif type == "dice":
        return dice_distance(target, valid_refs, set_to_ids)
    elif type == "overlap":
        return overlap_distance(target, valid_refs, set_to_ids)
    elif type == "tversky":
        return tversky_distance(target, valid_refs, set_to_ids, linker)

    elif type == "depth":
        return depth_weighted_distance(
            target, valid_refs, depth_sets, max_depth, set_to_ids,
            gamma=kwargs.get("gamma", 1.0)
        )
    elif type == "umls_neg":
        return umls_neg(target, neg_target, valid_refs, neg_valid_refs, set_to_ids,linker, wrapper, depth_sets, kwargs["alpha"], kwargs["beta"])
    elif type == "sym_tversky":
        return sym_tversky(target, neg_target, valid_refs, set_to_ids, linker, wrapper, depth_sets, dist_from_t, kwargs["alpha"],
                             kwargs["beta"])
    else:
        raise NotImplementedError(f"{type} is not implemented")


# ours

def umls_distance(target, valid_refs, depth_sets, max_depth, set_to_ids, linker, reachable, dist_from_t, cui_to_vid, decay_base=0.5):
    scored = []
    for ref_cuis in valid_refs:

        add_fwd = [
            sum(apply_filter(linker, ds & ref_cuis))
            for ds in depth_sets
        ]

        depth_sets_R = [set() for _ in range(max_depth)]
        ref_vids = {cui_to_vid[c] for c in ref_cuis if c in cui_to_vid}

        for t_cui in target:
            # shortest distance t → any member of R
            d = min(
                (dist_from_t[t_cui].get(r_vid, max_depth + 1) for r_vid in ref_vids),
                default=max_depth + 1
            )
            if 1 <= d <= max_depth:
                depth_sets_R[d - 1].add(t_cui)

        add_rev = [
            sum(apply_filter(linker, ds))  # ds already subset of target
            for ds in depth_sets_R
        ]

        try:
            max_add_idx_fwd = max(i for i, v in enumerate(add_fwd) if v)
        except ValueError:
            max_add_idx_fwd = 0
        try:
            max_add_idx_rev = max(i for i, v in enumerate(add_rev) if v)
        except ValueError:
            max_add_idx_rev = 0

        max_pair_index = max(max_add_idx_fwd, max_add_idx_rev)
        weights = [math.exp(-decay_base * d) for d in range(max_pair_index + 1, -1, -1)]

        unreachables = sum(apply_filter(linker, ref_cuis - reachable))


        weighted_fwd = [(d+1) * add_fwd[d] for d in range(max_pair_index+1)]
        weighted_fwd.append(unreachables * (max_pair_index + 1))

        weighted_fwd = sum(w * h for w, h in zip(weights, weighted_fwd))

        weighted_rev = [(d+1) * add_rev[d] for d in range(max_pair_index + 1)]
        weighted_rev.append(unreachables * (max_pair_index + 1))

        weighted_rev = sum(w * h for w, h in zip(weights, weighted_rev))


        alpha = weighted_fwd
        beta = weighted_rev

        #a  = sum(apply_filter(linker, ref_cuis - target))
        #b = sum(apply_filter(linker, target - ref_cuis))
        denominator = sum(apply_filter(linker, ref_cuis & target)) + alpha + beta
        if denominator == 0:
            sim = 0
        else:
            sim = sum(apply_filter(linker, ref_cuis & target)) / denominator
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


def apply_filter(linker: ClinicalEntityLinker, target):
    stys = [linker.cui2sty[cui] for cui in target]
    weights = [linker.importance_scores_sty.get(sty, 0) for sty in stys]
    return weights

def filter_cuis(linker: ClinicalEntityLinker, target):
    stys = [linker.cui2sty[cui] for cui in target]
    weights = [linker.importance_scores_sty.get(sty, 0) for sty in stys]
    cuis = []
    target = list(target)
    for i in range(len(target)):
        if weights[i] > 0:
           cuis.append(target[i])
    return set(cuis)

def weight_by_relation(
    reference_cuis: set[str],
    linker:          ClinicalEntityLinker,
    *,
    pred_map:        dict[int, int],
    vid_to_cui:      dict[int, str],
    cui_to_vid:      dict[str, int],
    rel_dict:        dict[tuple[str, str],
                         list[str]],
    max_depth: int = 0,
) -> int:

    syn_rels = {"RQ", "SY"}
    if not reference_cuis:
        return 0

    adds = 0

    for cui in reference_cuis:
        vid = cui_to_vid.get(cui)
        if vid is None:
            adds += 1
            continue

        cur = vid
        depth = 0
        while True:
            pred = pred_map.get(cur, -1)
            if pred == -1:
                if depth == 0:
                    depth = 0.5
                break  # reached a root

            rels = (rel_dict.get((vid_to_cui[pred], vid_to_cui[cur]))
                    or rel_dict.get((vid_to_cui[cur], vid_to_cui[pred])))

            if  len(set(rels) - syn_rels) == 0:
                depth = 1 if depth == 0 else depth
                cur = pred
                continue
            #pars = rels.count("PAR")
            #rb = rels.count("RB")
            #chds = rels.count("CHD")
            #rn = rels.count("RN")

            depth += 1
            cur = pred

        adds += depth

    return adds

def filter_synonyms(reference_cuis: set[str],
                    cui_to_vid: dict[str, int],
                    vid_to_cui: dict[int, str],
                    pred_map: dict[int, int],
                    rel_dict):
    syn_rels = {"RL", "RQ", "SY"}
    if not reference_cuis:
        return 0

    adds = 0

    for cui in reference_cuis:
        vid = cui_to_vid.get(cui)
        if vid is None:
            adds += 1
            continue

        all_syns = True
        cur = vid
        add = 1
        while True:
            pred = pred_map.get(cur, -1)
            if pred == -1:
                break  # reached a root

            rels = (rel_dict.get((vid_to_cui[pred], vid_to_cui[cur]))
                    or rel_dict.get((vid_to_cui[cur], vid_to_cui[pred])))

            if not any(r in syn_rels for r in rels):
                all_syns = False
            cur = pred

        if all_syns:
            add = 0
        adds += add

    return adds


def add_synonyms(reference_cuis: set[str],
                 target_cuis: set[str],
                 cui_to_vid: dict[str, int],
                 vid_to_cui: dict[int, str],
                 pred_map: dict[int, int],
                 rel_dict):
    if not reference_cuis:
        return reference_cuis

    def edge_rels(a_vid: int, b_vid: int) -> tuple[str, ...]:
        """Return relations on the edge (a,b), order-agnostic."""
        a, b = vid_to_cui[a_vid], vid_to_cui[b_vid]
        return rel_dict.get((a, b)) or rel_dict.get((b, a)) or ()

    def reaches_target(start_vid: int, relation_code: str) -> bool:
        """Walk up while every edge contains `relation_code`."""
        cur = start_vid
        while True:
            if vid_to_cui[cur] in target_cuis:
                return True
            parent = pred_map.get(cur, -1)
            if parent == -1 or relation_code not in edge_rels(parent, cur):
                return False
            cur = parent

    adds = {
        cui
        for cui in reference_cuis
        if (vid := cui_to_vid.get(cui)) is not None
           and (
                   reaches_target(vid, "SY")
                   #or reaches_target(vid, "PAR")
                   #or reaches_target(vid, "CHD")
           )
    }

    return target_cuis | adds



# jaccard distance

def jaccard_distance(target, valid_refs, set_to_ids):
    scored = []
    for ref_cuis in valid_refs:
        sim = len(ref_cuis & target) / len(ref_cuis | target) if (ref_cuis | target) else 0
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored



# Sorensen–Dice distance
def dice_distance(target, valid_refs, set_to_ids):
    scored = []
    for ref_cuis in valid_refs:
        denom = len(target) + len(ref_cuis)
        sim = (2 * len(target & ref_cuis) / denom) if denom else 0
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


# Overlap distance
def overlap_distance(target, valid_refs, set_to_ids):
    scored = []
    for ref_cuis in valid_refs:
        denom = min(len(target), len(ref_cuis))
        sim = (len(target & ref_cuis) / denom) if denom else 0
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


# Depth weighed distance
def depth_weighted_distance(
    target, valid_refs, depth_sets, max_depth, set_to_ids, *, gamma=1.0
):
    """
    depth_sets[d]  = CUIs discovered exactly at distance d from *target* in BFS.
    Weight for depth d:  w_d = exp(‑γ·d).  Larger γ => faster decay.
    Distance  = 1 / (1 + sim)   (so more overlap at close depths → smaller cost)
    """
    weights = [math.exp(-gamma * d) for d in range(max_depth + 1)]
    scored = []
    for ref_cuis in valid_refs:
        per_depth_hits = [len(ds & ref_cuis) for ds in depth_sets]
        sim = sum(w * n for w, n in zip(weights, per_depth_hits))
        cost = 1 / (1 + sim)
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored



def ic_dist(target, valid_refs, set_to_ids, ic_graph_wrapper, type="lin"):
    target = ic_graph_wrapper.translate(target)
    valid_refs = ic_graph_wrapper.translate(valid_refs)
    if type == "lin":
        return lin(target, valid_refs, set_to_ids, ic_graph_wrapper)
    else:
        raise NotImplementedError(f"{type} is not implemented")

def lin(target, valid_refs, set_to_ids, ic_graph_wrapper):
    scored = []
    if len(scored) % 10 == 0:
        print(f"scored {len(scored)} entries")
    for ref_cuis in valid_refs:
        if len(ref_cuis) == 0 and len(target) == 0:
            cost = 0
        elif len(ref_cuis) == 0 or len(target) == 0:
            cost = 1
        else:
            numerator = 2 * resnik_bma(ic_graph_wrapper.graph, target, ref_cuis, ic_graph_wrapper.ic_scores)
            target_ic = _mean_ic(target, ic_graph_wrapper.ic_scores)
            ref_ic = _mean_ic(ref_cuis, ic_graph_wrapper.ic_scores)
            denominator = target_ic + ref_ic
            sim = 0 if denominator == 0 else numerator / denominator
            cost = 1 - sim

        for ref_id in set_to_ids[frozenset(ic_graph_wrapper.translate_back(ref_cuis))]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


def tversky_distance(target, valid_refs, set_to_ids, linker, alpha=0.1, beta=0.9):
    scored = []
    for ref_cuis in valid_refs:
        #if sum(apply_filter(linker, ref_cuis)) < sum(apply_filter(linker, target)):
        #    alpha_tmp = alpha
        #    alpha = beta
        #    beta = alpha_tmp

        denominator = sum(apply_filter(linker, ref_cuis & target)) + alpha * sum(apply_filter(linker, ref_cuis - target)) + beta * sum(apply_filter(linker, target - ref_cuis))
        sim = sum(apply_filter(linker, ref_cuis & target)) / denominator
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored



def umls_neg(target, neg_target, valid_refs, neg_valid_refs, set_to_ids, linker, wrapper, depth_sets, alpha, beta):
    scored = []
    if neg_target is not None:
        contradictions = sum(apply_filter(linker, target & neg_target))
    else:
        contradictions = 0
    for ref_cuis, neg_ref_cuis in zip(valid_refs, neg_valid_refs):
        target_exp = add_synonyms(ref_cuis,
                                  target_cuis=target,
                                  pred_map=wrapper.pred_map,
                                  vid_to_cui=wrapper.vid_to_cui,
                                  cui_to_vid=wrapper.cui_to_vid,
                                  rel_dict=wrapper.rel_dict)

        intersect = sum(apply_filter(linker, ref_cuis & target_exp))
        adds = sum(apply_filter(linker, ref_cuis - target_exp))
        removes = sum(apply_filter(linker, target_exp - ref_cuis))
        denominator = (intersect + adds + removes) + contradictions
        denominator = denominator if denominator > 0 else 1
        sim = intersect / denominator

        intersect_neg = sum(apply_filter(linker, neg_ref_cuis & neg_target))
        adds_neg = sum(apply_filter(linker, neg_ref_cuis - neg_target))
        removes_neg = sum(apply_filter(linker, neg_target - neg_ref_cuis))
        denominator = (intersect_neg + adds_neg + removes_neg) + contradictions
        denominator = denominator if denominator > 0 else 1
        sim_neg = intersect / denominator
        cost = 1 - (alpha * sim + beta * sim_neg)
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


def sym_tversky(target, neg_target, valid_refs, set_to_ids, linker, wrapper, depth_sets, dist_from_t, alpha, beta):
    scored = []

    if neg_target is not None:
        contradictions = sum(apply_filter(linker, target & neg_target))
    else:
        contradictions = 0
    for ref_cuis in valid_refs:

        target_exp = add_synonyms(ref_cuis,
                                  target_cuis=target,
                                  pred_map=wrapper.pred_map,
                                  vid_to_cui=wrapper.vid_to_cui,
                                  cui_to_vid=wrapper.cui_to_vid,
                                  rel_dict=wrapper.rel_dict)


        diff_forward = sum(apply_filter(linker, ref_cuis - target_exp))
        diff_backward = sum(apply_filter(linker, target_exp - ref_cuis))

        a = min(diff_backward, diff_forward)
        b = max(diff_backward, diff_forward)

        intersect = sum(apply_filter(linker, ref_cuis & target))
        #adds = alpha * sum(apply_filter(linker, ref_cuis - target_exp))
        #removes = beta * sum(apply_filter(linker, target_exp - ref_cuis))
        #denominator = (intersect + adds + removes) + contradictions
        denominator = intersect + beta * (alpha * a + (1-alpha) * b) + contradictions
        denominator = denominator if denominator > 0 else 1
        sim = intersect / denominator
        cost = 1 - sim
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored


def weight_by_relation_v2(
    reference_cuis: set[str],
    linker:          ClinicalEntityLinker,
    *,
    pred_map:        dict[int, int],
    vid_to_cui:      dict[int, str],
    cui_to_vid:      dict[str, int],
    rel_dict:        dict[tuple[str, str], list[str]],
    depth_of:        dict[int, int],
    unreachable_cost: int = 5,
) -> float:
    """
    Like v2, but makes CHD (child-of) edges cheaper.

    • Synonyms (‘SY’, ‘RQ’) still cost 0.
    • ‘CHD’ costs 0.5 per edge.
    • All other non-synonym relations (PAR, RB, RN, …) cost 1.
    • CuIs that are missing or unreachable add `unreachable_cost`.
    """
    if not reference_cuis:
        return 0.0

    syn_rels    = {"RQ", "SY"}
    edge_weights = defaultdict(lambda: 1.0, {"CHD": 0.5})   # <── tweak here

    total = 0.0

    for cui in reference_cuis:
        vid = cui_to_vid.get(cui)

        # ── 1. Not in graph
        if vid is None:
            total += unreachable_cost
            continue

        # ── 2. In graph but unreachable from any source
        if depth_of.get(vid, -1) == -1:
            total += unreachable_cost
            continue

        # ── 3. Walk up predecessor chain
        cost = 0.0
        cur  = vid
        while True:
            pred = pred_map.get(cur, -1)
            if pred == -1:                        # reached a BFS root
                break

            rels = (rel_dict.get((vid_to_cui[pred], vid_to_cui[cur]))
                    or rel_dict.get((vid_to_cui[cur], vid_to_cui[pred]))
                    or [])

            # strip out synonym codes
            non_syn_rels = [r for r in rels if r not in syn_rels]
            if not non_syn_rels:                  # synonym-only edge → free
                cur = pred
                continue

            # cheapest non-synonym relation on this edge
            edge_cost = min(edge_weights[r] for r in non_syn_rels)
            cost += edge_cost
            cur = pred

        total += cost

    return total


