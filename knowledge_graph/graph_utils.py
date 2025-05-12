import os
import random

from knowledge_graph.graph import build_graph, GraphWrapper

import random
from typing import FrozenSet, Union, Optional

import pickle
from collections import defaultdict
import cudf, cugraph
import cupy
from pylibcugraph import bfs, ResourceHandle, SGGraph, GraphProperties
import cupy as cp

from knowledge_graph.ner import Mention


def _bfs_depth(handle: ResourceHandle, sg: SGGraph, sources: cudf.Series, depth: int = 2):
    distances, _, vertices = bfs(handle, sg, sources,
                                 False,
                                 depth,
                                 False,
                                 False)
    return distances, vertices




from typing import Dict, Iterable, List, Set, Tuple

def k_closest_reference_reports(
    handle: ResourceHandle,
    sg: SGGraph,
    cui_to_vid: Dict[str, int],
    vid_to_cui: Dict[int, str],
    reference_reports: Iterable[Tuple[str, Set[str]]],   # (dicom_id, CUIs)
    target_reports: Iterable[Tuple[str, Iterable[str]]], # (dicom_id, CUIs)
    set_to_indices: Dict[FrozenSet, str],
    k: int = 1,
    max_depth: int = 3,
    allow_empty_references: bool = False
) -> Dict[str, List[Tuple[str, Set[str], int]]]:

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")

    frozen_refs = [
        (ref_id, frozenset(cuis))
        for ref_id, cuis in reference_reports
        if allow_empty_references or cuis
    ]

    results: Dict[str, List[Tuple[str, Set[str], int]]] = {}

    for target_id, target_cuis in target_reports:
        target = frozenset(target_cuis)
        if not target:
            ret = []
            for i in range(k):
                ret.append((random.choice(set_to_indices[frozenset()]), frozenset(), 0))
            results[target_id] = ret
            return results

        try:
            vids = {cui_to_vid[cui] for cui in target}
        except KeyError as e:
            raise ValueError(f"CUI {e.args[0]!r} from target_report not in id_map")

        distances, vertices = _bfs_depth(handle, sg, cudf.Series(list(vids)), depth=max_depth)


        depth_sets: List[Set[str]] = []
        for d in range(1, max_depth + 1):
            vids_at_d = cupy.asnumpy(vertices[distances == d])
            depth_sets.append({vid_to_cui[v] for v in vids_at_d})


        reachable: Set[str] = set(target)
        for s in depth_sets:
            reachable |= s

        valid_refs = [
            (ref_id, ref_cuis)
            for ref_id, ref_cuis in frozen_refs
            if reachable.issuperset(ref_cuis)
        ]

        scored: List[Tuple[str, Set[str], int]] = []
        for ref_id, ref_cuis in valid_refs:
            remove = len(target - (ref_cuis & target))

            add_per_depth = [len(ds & ref_cuis) for ds in depth_sets]

            weighted_adds = sum((d + 1) * add_per_depth[d] for d in range(max_depth))

            cost = (
                remove
                + weighted_adds
                + abs(remove - sum(add_per_depth))
            )

            scored.append((ref_id, set(ref_cuis), cost))

        scored.sort(key=lambda x: (x[2], x[0]))
        results[target_id] = scored[:k]

    return results


def construct_gpu_graph(G: cugraph.Graph, drop=None) -> tuple[SGGraph, ResourceHandle]:
    handle = ResourceHandle()
    G_sym = G.to_directed()
    srcs = G_sym.edgelist.edgelist_df['src']
    dsts = G_sym.edgelist.edgelist_df['dst']
    if drop:
        keep_mask = (~cp.isin(cp.asarray(srcs), cp.asarray(drop)) & (~cp.isin(cp.asarray(dsts), cp.asarray(drop))))
        srcs = srcs[keep_mask]
        dsts = dsts[keep_mask]

    del G_sym

    return SGGraph(
        handle,
        GraphProperties(is_multigraph=G.is_multigraph()),
        srcs,
        dsts
    ), handle



def topk_indices(results_by_id: dict[str, list[tuple[str, set[str], int]]],  # id -> [(neighbour_id, set, cost), ...]
                 id_to_index: Dict[str, int],  # id -> List of nearest neighbors indices
                 k: int,
                 shuffle: bool = False,
                 random_generator: random.Random = random.Random(4200)) -> Dict[str, List[int]]:

    top_k_indices = {}
    for id in results_by_id.keys():
        if shuffle:
            random_generator.shuffle(results_by_id[id])
        total_indices = []
        for triple in results_by_id[id]:
                total_indices.append(id_to_index[triple[0]])
        top_k_indices[id] = total_indices[:k]
    return top_k_indices


def preprocess_mentions(mentions: List[List[Mention]]) \
        -> Tuple[List[Set[str]], List[Set[str]], Dict[FrozenSet[str], List[int]]]:
    report_list = []
    neg_report_list = []
    set_to_indices = defaultdict(list)

    for idx, mentions_per_text in enumerate(mentions):
        has_observation = any(
            m.category.startswith("Observation") and m.assertion == "present" for m in mentions_per_text)

        cuis_set = {
            m.cui
            for m in mentions_per_text
            if m.assertion != "absent"
               and (has_observation or not m.category.startswith("Anatomy"))  # ← drop anatomy if no OBS
        }

        negated_cuis_set = {
            m.cui
            for m in mentions_per_text
            if m.assertion == "absent"

        }

        report_list.append(cuis_set)
        neg_report_list.append(negated_cuis_set)
        set_to_indices[frozenset(cuis_set)].append(idx)
    return report_list, neg_report_list, set_to_indices



def prepare_graph(mention_path: Union[str, os.PathLike],
                  rff_file_path: Union[str, os.PathLike],
                  id_to_index: Optional[Dict[str, int]] = None) -> GraphWrapper:

    mentions = pickle.load(open(mention_path, "rb"))

    report_list, neg_report_list, set_to_indices = preprocess_mentions(mentions)

    keep_cuis = set().union(*report_list).union(*neg_report_list)

    G, id_map = build_graph(rff_file_path, keep_cuis=keep_cuis)
    report_list_full = report_list.copy()
    neg_report_list_full = neg_report_list.copy()
    return GraphWrapper(G,
                        id_map,
                        set_to_indices,
                        report_list_full,
                        neg_report_list_full,
                        id_to_index)
