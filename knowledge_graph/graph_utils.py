import os

from knowledge_graph.cost_functions import dist, ic_dist
from knowledge_graph.graph import build_graph, GraphWrapper

import random
from typing import FrozenSet, Union, Dict, Tuple, Iterable, List, Set, Optional

import pickle
from collections import defaultdict
import cudf, cugraph
import cupy
from pylibcugraph import bfs, ResourceHandle, SGGraph, GraphProperties
import cupy as cp

from knowledge_graph.ic_metrics import ICGraphWrapper
from knowledge_graph.ner import Mention, get_mentions


def _bfs_depth(handle: ResourceHandle, sg: SGGraph, sources: cudf.Series, depth: int = 2):
    distances, _, vertices = bfs(handle, sg, sources,
                                 False,
                                 depth,
                                 False,
                                 False)
    return distances, vertices





def k_closest_reference_reports(
    handle: ResourceHandle,
    sg: SGGraph,
    cui_to_vid: Dict[str, int],
    vid_to_cui: Dict[int, str],
    reference_reports: Iterable[Tuple[str, Set[str]]],   # (dicom_id, CUIs)
    target_reports: Iterable[Tuple[str, Iterable[str]]], # (dicom_id, CUIs)
    set_to_ids: Dict[FrozenSet, str],
    k: int = 1,
    max_depth: int = 3,
    allow_empty_references: bool = False,
    cost_function: str = "umls",
    shuffle: bool  =True,
    ic_graph_wrapper: ICGraphWrapper = None,
) -> Dict[str, List[Tuple[str, Set[str], int]]]:

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")


    unique_refs = {}

    for target_id, cuis in reference_reports:
        if not allow_empty_references:
            cuis = {cui for cui in cuis if len(cui) > 0}
        unique_refs[target_id] = frozenset(cuis)




    results: Dict[str, List[Tuple[str, Set[str], int]]] = {}

    for target_id, target_cuis in target_reports:

        target = frozenset(target_cuis)

        if not target:
            ret = []
            for i in range(k):
                ret.append((random.choice(set_to_ids[frozenset()]), frozenset(), 0))
            results[target_id] = ret
            continue


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


        valid_refs = [cui_set for cui_set in unique_refs[target_id] if reachable.issuperset(cui_set)]
        if ic_graph_wrapper is None:
            scored = dist(target, valid_refs, depth_sets, max_depth, set_to_ids, type=cost_function)
        else:
            scored = ic_dist(target, valid_refs, set_to_ids, ic_graph_wrapper, type=cost_function)
        if shuffle:
            random.shuffle(scored)
        scored.sort(key=lambda x: (x[2]))
        results[target_id] = scored[:k]

    return results


def construct_gpu_graph(G: cugraph.Graph, drop=None) -> tuple[SGGraph, ResourceHandle]:
    handle = ResourceHandle()
    G_sym = G.to_directed()
    srcs = G_sym.edgelist.edgelist_df['src']
    dsts = G_sym.edgelist.edgelist_df['dst']
    if drop is not None:
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
                 k: int) -> Dict[str, List[int]]:

    top_k_indices = {}
    for id in results_by_id.keys():
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

        cuis_set = frozenset(cuis_set)

        negated_cuis_set = {
            m.cui
            for m in mentions_per_text
            if m.assertion == "absent"

        }

        negated_cuis_set = frozenset(negated_cuis_set)

        report_list.append(cuis_set)
        neg_report_list.append(negated_cuis_set)
        set_to_indices[frozenset(cuis_set)].append(idx)
    return report_list, neg_report_list, set_to_indices



def prepare_graph(mention_path: Union[str, os.PathLike],
                  impressions,
                  rff_file_path: Union[str, os.PathLike],
                  ids: List[str],
                  *,
                  additional_cuis: Optional[Set] = None) -> GraphWrapper:


    mentions = get_mentions(mention_path, impressions)

    report_list, neg_report_list, set_to_indices = preprocess_mentions(mentions)

    keep_cuis = set().union(*report_list).union(*neg_report_list)
    if additional_cuis is not None:
        keep_cuis = keep_cuis | set(additional_cuis)

    set_to_id = defaultdict(list)
    for rpt, i in zip(report_list, ids):
        set_to_id[rpt].append(i)

    id_to_index = dict(zip(ids, list(range(len(ids)))))

    G, id_map = build_graph(rff_file_path, keep_cuis=keep_cuis)

    return GraphWrapper(G,
                        id_map,
                        set_to_indices,
                        report_list,
                        neg_report_list,
                        id_to_index,
                        set_to_id)


def bfs_trim_subgraph(handle: ResourceHandle,
                      sg: SGGraph,
                      seed_cuis: Iterable,
                      max_depth: int,
                      *,
                      edges) -> cudf.DataFrame:
    seeds = cudf.Series(cp.asarray(list(seed_cuis), dtype=cp.int32), name="sources")
    bfs_df = bfs(handle, sg, seeds, False, max_depth, False, False)
    distances = bfs_df[0]
    mask = distances < cp.iinfo(distances.dtype).max
    reached = cp.flatnonzero(mask)

    mask = edges["src"].isin(reached) | edges["dst"].isin(reached)
    sub_edges = edges[mask]

    return sub_edges


def get_k_nearest_image_paths_from_report(report: str, k: int) -> List[List[str]]:
    pass

