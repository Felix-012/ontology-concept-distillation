from typing import List, Set, Dict, Tuple, Union
import math

from knowledge_graph.ic_metrics import resnik_bma, _mean_ic


def dist(
    target: Set[str],
    valid_refs: List[Set[str]],
    depth_sets: List[Set[str]],
    max_depth: int,
    set_to_ids: Dict[frozenset, Set[str]],
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
        return umls_distance(target, valid_refs, depth_sets, max_depth, set_to_ids)
    elif type == "jac":
        return jaccard_distance(target, valid_refs, set_to_ids)
    elif type == "dice":
        return dice_distance(target, valid_refs, set_to_ids)
    elif type == "overlap":
        return overlap_distance(target, valid_refs, set_to_ids)

    elif type == "depth":
        return depth_weighted_distance(
            target, valid_refs, depth_sets, max_depth, set_to_ids,
            gamma=kwargs.get("gamma", 1.0)
        )
    else:
        raise NotImplementedError(f"{type} is not implemented")


# ours
def umls_distance(target, valid_refs, depth_sets, max_depth, set_to_ids):
    scored = []
    for ref_cuis in valid_refs:
        remove = len(target - (ref_cuis & target))
        add_per_depth = [len(ds & ref_cuis) for ds in depth_sets]
        weighted_adds = sum((d + 1) * add_per_depth[d] for d in range(max_depth))
        cost = remove + weighted_adds + abs(remove - sum(add_per_depth))
        for ref_id in set_to_ids[ref_cuis]:
            scored.append((ref_id, ref_cuis, cost))
    return scored

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

