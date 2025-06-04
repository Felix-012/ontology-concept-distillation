from typing import List, Set, Dict, Tuple, Union, Sequence, Iterable
import math
import numpy as np
from knowledge_graph.ic_metrics import resnik_bma, _mean_ic
from knowledge_graph.ner import ClinicalEntityLinker


def dist(
    target: Set[str],
    neg_target: Set[str],
    valid_refs: List[Set[str]],
    depth_sets: List[Set[str]],
    max_depth: int,
    set_to_ids: Dict[frozenset, Set[str]],
    linker: ClinicalEntityLinker,
    reachable: Set[str],
    wrapper,
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
        return umls_distance(target, neg_target, valid_refs, set_to_ids,linker, wrapper, kwargs["alpha"], kwargs["beta"])
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
    else:
        raise NotImplementedError(f"{type} is not implemented")


# ours

def umls_distance(target, neg_target, valid_refs, set_to_ids, linker, wrapper, alpha, beta):
    #target_str = ' '.join([linker.cui2str[cui] for cui in target])
    #print(f"Target: {target_str}")
    scored = []
    contradictions = sum(apply_filter(linker, target & neg_target))
    for ref_cuis in valid_refs:

        #ref_str = ' '.join([linker.cui2str[cui] for cui in ref_cuis])
        #print(f"Target: {ref_str}")
        '''
        add_weights = filter_synonyms(filter_cuis(linker, ref_cuis - target),
                                         pred_map=wrapper.pred_map,
                                         vid_to_cui=wrapper.vid_to_cui,
                                         cui_to_vid=wrapper.cui_to_vid,
                                         rel_dict=wrapper.rel_dict)
        remove_weights = filter_synonyms(filter_cuis(linker, target-ref_cuis),
                                         pred_map=wrapper.pred_map,
                                         vid_to_cui=wrapper.vid_to_cui,
                                         cui_to_vid=wrapper.cui_to_vid,
                                         rel_dict=wrapper.rel_dict)
        adds = add_weights
        removes = remove_weights
        '''

        #old_len = len(target)
        target_exp = add_synonyms(ref_cuis,
                                target_cuis=target,
                                pred_map=wrapper.pred_map,
                                vid_to_cui=wrapper.vid_to_cui,
                                cui_to_vid=wrapper.cui_to_vid,
                                rel_dict=wrapper.rel_dict)
        #if old_len != len(target_exp):
        #    print(f"added {len(target_exp) - old_len} synonyms")

        #target_exp = target
        # if sum(apply_filter(linker, ref_cuis)) < sum(apply_filter(linker, target)):
        #    alpha_tmp = alpha
        #    alpha = beta
        #    beta = alpha_tmp
        intersect = sum(apply_filter(linker, ref_cuis & target_exp))
        adds = alpha * sum(apply_filter(linker, ref_cuis - target_exp))
        removes =  beta * sum(apply_filter(linker, target_exp - ref_cuis))
        denominator = (intersect + adds + removes) + contradictions
        denominator = denominator if denominator > 0 else 1
        sim = intersect / denominator
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
    depth_of:        dict[int, int],
    rel_dict:        dict[tuple[str, str],
                         list[str]],
    min_weight:      int        = 0,
    max_depth: int = 0,
) -> int:

    syn_rels = {"RL", "RQ", "SY"}
    if not reference_cuis:
        return 0

    adds = 0

    for cui in reference_cuis:
        vid = cui_to_vid.get(cui)
        if vid is None:
            adds += 1
            continue

        last_child = False
        last_parent = False
        cur = vid
        depth = 1
        while True:
            pred = pred_map.get(cur, -1)
            if pred == -1:
                break  # reached a root

            rels = (rel_dict.get((vid_to_cui[pred], vid_to_cui[cur]))
                    or rel_dict.get((vid_to_cui[cur], vid_to_cui[pred])))

            if any(r in syn_rels for r in rels):
                cur = pred
                continue

            if "CHD" in rels:
                last_child = True
                if last_parent:
                    last_parent = False
                    cur = pred
                    continue

            if "PAR" in rels:
                last_parent = True
                if last_child:
                    last_child = False
                    cur = pred
                    continue

            last_parent = False
            last_child = False

            depth += 1
            cur = pred


        adds += depth/max_depth

        print(f"{linker.cui2str[cui]}")
        print(f"Depth: {depth/max_depth}")

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

    syn_rels = {"SY"}
    if not reference_cuis:
        return reference_cuis

    adds = set()

    for cui in reference_cuis:
        vid = cui_to_vid.get(cui)
        if vid is None:
            continue

        cur = vid
        while True:
            if vid_to_cui[cur] in target_cuis:
                adds.add(cui)
                break

            pred = pred_map.get(cur, -1)

            if pred == -1:
                break  # reached a root without hitting target

            rels = (rel_dict.get((vid_to_cui[pred], vid_to_cui[cur]))
                or rel_dict.get((vid_to_cui[cur], vid_to_cui[pred]))
                or ())

            if not any(r in syn_rels for r in rels):
                break

            cur = pred

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
    if len(scored) % 100 == 0:
        print(f"scored {len(scored)} entries")
    for ref_cuis in valid_refs:
        if len(ref_cuis) == 0 and len(target) == 0:
            cost = 0
        elif len(ref_cuis) == 0 or len(target) == 0:
            cost = 1
        else:
            numerator = 2 * resnik_bma(ic_graph_wrapper.graph, ic_graph_wrapper.handle, target, ref_cuis, ic_graph_wrapper.ic_scores)
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



import math
from collections import defaultdict


'''
def umls_distance(
        target,
        valid_refs,
        depth_sets,
        max_depth,
        set_to_ids,
        linker,
        reachable,
        dist_from_t,
        cui_to_vid,
        decay_base=0.9,
        alpha=1.0,
        beta=1.0,
):
    """
    Weighted-Jaccard (Ruzicka) distance between the CUI set `target`
    and every candidate in `valid_refs`, where 'weights' are produced
    from your depth-based decay scheme.
    """
    scored = []

    # --- pre-compute an exponential kernel that we can reuse -----------
    #exp_kernel = [math.exp(-decay_base * d)  # d = max_depth, …, 0
    #              for d in reversed(range(max_depth + 1))]       # plus the “unreachable” bucket

    for ref_cuis in valid_refs:

        # ------------------------------------------------------------------
        # 1) how many *reference* CUIs sit at each distance 1…max_depth
        #    away from *any* target CUI  (forward direction)
        # ------------------------------------------------------------------
        add_fwd = [
            sum(apply_filter(linker, ds & ref_cuis))
            for ds in depth_sets                          # depth_sets[d] is distance d+1
        ]

        # ------------------------------------------------------------------
        # 2) how many *target* CUIs sit at each distance 1…max_depth
        #    away from *any* reference CUI  (reverse direction)
        # ------------------------------------------------------------------
        depth_sets_R = [set() for _ in range(max_depth)]
        ref_vids = {cui_to_vid[c] for c in ref_cuis if c in cui_to_vid}

        for t_cui in target:
            d = min(
                (dist_from_t[t_cui].get(r_vid, max_depth + 1) for r_vid in ref_vids),
                default=max_depth + 1
            )
            if 1 <= d <= max_depth:
                depth_sets_R[d - 1].add(t_cui)

        add_rev = [
            sum(apply_filter(linker, ds))                  # ds ⊆ target
            for ds in depth_sets_R
        ]

        # ------------------------------------------------------------------
        # 3) unreachable references (distance > max_depth)
        # ------------------------------------------------------------------
        unreachable = sum(apply_filter(linker, ref_cuis - reachable))

        try:
            max_add_idx_fwd = max(i for i, v in enumerate(add_fwd) if v)
        except ValueError:
            max_add_idx_fwd = 0
        try:
            max_add_idx_rev = max(i for i, v in enumerate(add_rev) if v)
        except ValueError:
            max_add_idx_rev = 0

        max_pair_index = max(max_add_idx_fwd, max_add_idx_rev)

        # ------------------------------------------------------------------
        # 4) convert those counts into *weights* with the exponential kernel
        # ------------------------------------------------------------------
        # forward side
        A = sum((d + 1) * add_fwd[d] for d in range(max_depth)) + (max_pair_index + 1) * unreachable

        # reverse side
        B = sum((d + 1) * add_rev[d]
                for d in range(max_depth)) \
            + (max_pair_index + 1) * unreachable

        # ------------------------------------------------------------------
        # 5) literal intersection (distance = 0)
        # ------------------------------------------------------------------
        I = sum(apply_filter(linker, ref_cuis & target))  # weight 1 per shared CUI

        # ------------------------------------------------------------------
        # 6) weighted-Jaccard & distance
        # ------------------------------------------------------------------
        a = sum(apply_filter(linker, ref_cuis - target))
        b = sum(apply_filter(linker, target - ref_cuis))
        denom = I + alpha * A + beta * B
        sim   = 0.0 if denom == 0 else I / denom
        cost  = 1.0 - sim

        # ------------------------------------------------------------------
        # 7) collect scores for every reference id that shares this CUI set
        # ------------------------------------------------------------------
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))

    return scored
    
'''


