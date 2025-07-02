import os
from itertools import repeat

from knowledge_graph.cost_functions import dist, ic_dist
from knowledge_graph.graph import build_graph, GraphWrapper

import random
from typing import FrozenSet, Union, Dict, Tuple, Iterable, List, Set, Optional

from collections import defaultdict
import cudf, cugraph
import cupy
from pylibcugraph import bfs, ResourceHandle, SGGraph, GraphProperties
import cupy as cp

from knowledge_graph.ic_metrics import ICGraphWrapper
from knowledge_graph.ner import Mention, get_mentions, ClinicalEntityLinker


def _bfs_depth(handle: ResourceHandle, sg: SGGraph, sources: cudf.Series, depth: int = 2):
    distances, predecessors, vertices = bfs(handle, sg, sources,
                                 False,
                                 depth,
                                 True,
                                 False)
    return distances, predecessors, vertices





def k_closest_reference_reports(
    handle: ResourceHandle,
    sg: SGGraph,
    cui_to_vid: Dict[str, int],
    vid_to_cui: Dict[int, str],
    reference_reports: Iterable[Tuple[str, Set[str]]],   # (dicom_id, CUIs)
    target_reports: Iterable[Tuple[str, Iterable[str]]], # (dicom_id, CUIs)
    set_to_ids: Dict[FrozenSet, str],
    k: int = 1,
    max_depth: int = -1,
    allow_empty_references: bool = False,
    cost_function: str = "umls",
    ic_graph_wrapper: ICGraphWrapper = None,
    graph_wrapper: GraphWrapper = None,
    linker: ClinicalEntityLinker = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    neg_reference_reports: List[Tuple[str, Set[str]]] = None,
    neg_target = None,
    ids_to_index = None,
    neg_set_to_ids = None
) -> Dict[str, List[Tuple[str, Set[str], int]]]:

    if k <= 0:
        raise ValueError("k must be a positive integer")

    unique_refs = {}
    unique_neg_refs = {}

    for target_id, cuis in reference_reports:
        if not allow_empty_references:
            cuis = {cui for cui in cuis if len(cui) > 0}
        unique_refs[target_id] = frozenset(cuis)

    if neg_reference_reports:
        for target_id, cuis in neg_reference_reports:
            cuis_neg = {cui for cui in cuis if len(cui) > 0}
            unique_neg_refs[target_id] = frozenset(cuis_neg)


    results: Dict[str, List[Tuple[str, Set[str], int]]] = {}

    blank = (None, None)
    neg_iter = neg_target if neg_target is not None else repeat(blank)

    for (target_id, target_cuis), (neg_target_id, neg_target_cuis) in zip(target_reports, neg_iter):

        target = frozenset(target_cuis)
        if neg_target is not None:
            neg_target = frozenset(neg_target_cuis)

        if not target and not neg_target:
            ret = []
            for i in range(k):
                ret.append((random.choice(set_to_ids[frozenset()]), frozenset(), 0))
            results[target_id] = ret
            continue

        try:
            vids = {cui_to_vid[cui] for cui in target}

            neg_vids = None
            if neg_target is not None:
                neg_vids = {cui_to_vid[cui] for cui in neg_target}
        except KeyError as e:
            raise ValueError(f"CUI {e.args[0]!r} from target_report not in id_map")

        depth_sets: list[set[str]] = []  # CUIs newly discovered at depth d
        if vids:
            distances, predecessors, vertices = _bfs_depth(
                handle, sg, cudf.Series(list(vids)), depth=max_depth
            )

            vert_np = cupy.asnumpy(vertices)
            pred_np = cupy.asnumpy(predecessors)
            dist_np = cupy.asnumpy(distances)
            dist_np[dist_np >= cupy.iinfo(dist_np.dtype).max] = -1
            graph_wrapper.depth_of = dict(zip(vert_np, dist_np))
            graph_wrapper.pred_map = dict(zip(vert_np, pred_np))  # {vertex → predecessor}

            for d in range(1, max_depth + 1):
                vids_at_d = vert_np[dist_np == d]

                depth_sets.append({vid_to_cui[v] for v in vids_at_d})

        neg_depth_sets: List[Set[str]] = []
        if neg_vids:
            distances, predecessors, vertices = _bfs_depth(handle, sg, cudf.Series(list(neg_vids)), depth=max_depth)
            for d in range(1, max_depth + 1):
                vids_at_d = cupy.asnumpy(vertices[distances == d])
                neg_depth_sets.append({vid_to_cui[v] for v in vids_at_d})


        if not target:
            candidates = set_to_ids[frozenset()]
            i = 0
            refs = sorted(unique_refs[target_id], key=len)
            if len(candidates) == 0:
                new_refs = []
                while len(new_refs) == 0:
                    i += 1
                    for ref in refs:
                        if len(ref) > i:
                            break
                        else:
                            new_refs.append(ref)
                unique_refs[target_id] = new_refs
            else:
                cand_indices = [ids_to_index[candidate] for candidate in candidates]
                neg_candidates = [neg_reference_reports[0][1][cand_index] for cand_index in
                                  cand_indices]  # TODO: rewrite this so it works for aribitrary k

                restricted_neg_set_to_ids = defaultdict(list)
                for cid in candidates:
                    cuiset = neg_reference_reports[0][1][ids_to_index[cid]]
                    restricted_neg_set_to_ids[cuiset].append(cid)

                scores_neg = dist(neg_target, None, neg_candidates, neg_candidates, neg_depth_sets, max_depth,
                                  restricted_neg_set_to_ids, linker, None, graph_wrapper, None, type=cost_function,
                                  alpha=alpha, beta=beta)
                scores_neg.sort(key=lambda x: (x[2], x[0]))
                scores_neg = scores_neg[:k]
                scores_neg = [(triple[0], frozenset(), triple[2]) for triple in scores_neg]
                results[target_id] = scores_neg
                continue

        reachable = None

        if target:
            reachable: Set[str] = set(target)
            for s in depth_sets:
                reachable |= s

        neg_reachable = None
        if neg_target:
            neg_reachable: Set[str] = set(neg_target)
            for s in neg_depth_sets:
                neg_reachable |= s

        #valid_refs = [cui_set for cui_set in unique_refs[target_id] if reachable.issuperset(cui_set)]
        if target:
            valid_refs = [cui_set for cui_set in unique_refs[target_id] if len(reachable & cui_set) != 0]
        else:
            valid_refs = unique_refs[target_id]

        if not valid_refs:
            missing_sets = [cui_set - reachable for cui_set in unique_refs[target_id]]
            min_missing = min(len(m) for m in missing_sets)
            best_missing_sets = [m for m in missing_sets if len(m) == min_missing]
            to_add = best_missing_sets[0]
            reachable |= to_add
            valid_refs = [cui_set for cui_set in unique_refs[target_id] if reachable.issuperset(cui_set)]

        valid_id_set = {
            id_
            for ref in valid_refs
            for id_ in set_to_ids[ref]
        }
        if neg_reference_reports:
            valid_neg_refs = [
                neg_set
                for neg_set in unique_neg_refs[target_id]
                if valid_id_set.intersection(neg_set_to_ids.get(neg_set, []))
            ]

        if ic_graph_wrapper is None:
            scored = dist(target, neg_target, valid_refs, valid_neg_refs, depth_sets, max_depth, set_to_ids, linker, reachable, graph_wrapper, None, type=cost_function, alpha=alpha, beta=beta)
            if neg_target:
                scored_neg = dist(neg_target, None, valid_neg_refs, valid_neg_refs, neg_depth_sets, max_depth, neg_set_to_ids, linker, reachable, graph_wrapper, None, type=cost_function, alpha=alpha, beta=beta)
                neg_by_id = {t[0]: t for t in scored_neg}  # quick lookup by id
                paired = [(pos, neg_by_id[pos[0]])
                          for pos in scored
                          if pos[0] in neg_by_id]
                paired.sort(key=lambda p: (p[0][2], p[1][2]))

                if len(paired) != 0:
                    scored, _ = zip(*paired)
                else:
                    scored.sort(key=lambda x: (x[2], x[1]))

            else:
                scored.sort(key=lambda x: (x[2], x[0]))
        else:
            scored = ic_dist(target, valid_refs, set_to_ids, ic_graph_wrapper, type=cost_function)
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
            if type(m.cui) == str and  m.assertion != "absent"
               and (has_observation or not m.category.startswith("Anatomy"))  # ← drop anatomy if no OBS
        }

        cuis_set_text = {
            m.cui_text
            for m in mentions_per_text
            if type(m.cui_text) == str and m.assertion != "absent"
               and (has_observation or not m.category.startswith("Anatomy"))
        }

        cuis_set = frozenset(cuis_set | cuis_set_text)

        negated_cuis_set = {
            m.cui
            for m in mentions_per_text
            if type(m.cui) == str and m.assertion == "absent"

        }

        negated_cuis_set_text = {
            m.cui_text
            for m in mentions_per_text
            if type(m.cui_text) == str and m.assertion == "absent"

        }

        negated_cuis_set = frozenset(negated_cuis_set | negated_cuis_set_text)

        report_list.append(cuis_set)
        neg_report_list.append(negated_cuis_set)
        set_to_indices[frozenset(cuis_set)].append(idx)
    return report_list, neg_report_list, set_to_indices



def prepare_graph(mention_path: Union[str, os.PathLike],
                  impressions,
                  rff_file_path: Union[str, os.PathLike],
                  ids: List[str],
                  linker: ClinicalEntityLinker,
                  *,
                  additional_cuis: Optional[Set] = None) -> GraphWrapper:


    mentions = get_mentions(linker, mention_path, impressions)

    report_list, neg_report_list, set_to_indices = preprocess_mentions(mentions)

    keep_cuis = set().union(*report_list).union(*neg_report_list)
    if additional_cuis is not None:
        keep_cuis = keep_cuis | set(additional_cuis)

    set_to_id = defaultdict(list)
    neg_set_to_id = defaultdict(list)
    for rpt, i in zip(report_list, ids):
        set_to_id[rpt].append(i)

    for rpt, i in zip(neg_report_list, ids):
        neg_set_to_id[rpt].append(i)

    id_to_index = dict(zip(ids, list(range(len(ids)))))

    G, id_map, rel_dict = build_graph(rff_file_path, keep_cuis=keep_cuis)

    return GraphWrapper(G,
                        id_map,
                        set_to_indices,
                        report_list,
                        neg_report_list,
                        id_to_index,
                        set_to_id,
                        neg_set_to_id,
                        rel_dict)


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



