import random
from typing import Set, List, Tuple, Union


def dist(*args, type) -> List[Tuple[str, Set[str], Union[float, int]]]:
    if type == "umls":
        return umls_distance(*args)
    elif type == "jac":
        return jaccard_distance(*args)
    else:
        raise NotImplementedError(f"{type} is not implemented")

# our new method
def umls_distance(target, valid_refs, depth_sets, max_depth, set_to_ids) -> List[Tuple[str, Set[str], int]]:
    scored: List[Tuple[str, Set[str], int]] = []
    for ref_cuis in valid_refs:
            remove = len(target - (ref_cuis & target))

            add_per_depth = [len(ds & ref_cuis) for ds in depth_sets]

            weighted_adds = sum((d + 1) * add_per_depth[d] for d in range(max_depth))

            cost = (
                    remove
                    + weighted_adds
                    + abs(remove - sum(add_per_depth))
            )

            ref_ids = set_to_ids[ref_cuis]
            for ref_id in ref_ids:
                scored.append((ref_id, ref_cuis, cost))
    return scored

def jaccard_distance(target, valid_refs, depth_sets, max_depth, set_to_ids) -> List[Tuple[str, Set[str], float]]:
    scored: List[Tuple[str, Set[str], float]] = []
    for ref_cuis in valid_refs:
            cost = len(ref_cuis & target) / len(ref_cuis | target)
            ref_ids = set_to_ids[ref_cuis]
            for ref_id in ref_ids:
                scored.append((ref_id, ref_cuis, cost))
    return scored

