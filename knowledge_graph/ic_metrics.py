from __future__ import annotations

import functools
from typing import Set, FrozenSet, Dict

import cudf
import cugraph
import cupy as cp
import pylibcugraph
from pylibcugraph import bfs, ResourceHandle

import knowledge_graph.graph_utils

class ICGraphWrapper:
    def __init__(self, graph, handle, ic_scores, cui_to_vid, vid_to_cui):
        self.graph = graph
        self.cui_to_vid = cui_to_vid
        self.vid_to_cui = vid_to_cui
        self.ic_scores = ic_scores
        self.handle = handle

    @staticmethod
    def _translate_container(container, mapping):
        return {mapping[v] for v in container if v in mapping}

    def translate(self, original_set):
        if isinstance(original_set, set) or isinstance(original_set, frozenset):
            return self._translate_container(original_set, self.cui_to_vid)

        if isinstance(original_set, (list, tuple)):
            return [self._translate_container(s, self.cui_to_vid)
                    for s in original_set]

        raise TypeError(
            "original_set must be a set or a list of sets")

    def translate_back(self, translated_set):
        if isinstance(translated_set, set) or isinstance(translated_set, frozenset):
            return self._translate_container(translated_set, self.vid_to_cui)

        if isinstance(translated_set, (list, tuple)):
            return [self._translate_container(s, self.vid_to_cui)
                    for s in translated_set]

        raise TypeError(
            "translated_set must be a set[int] or a list of sets[int]")


def _intrinsic_ic(G: cugraph.Graph):
    indeg  = G.in_degree()
    indeg  = indeg.set_index('vertex')['degree']
    leaf_set = set(indeg[indeg == 0].index.to_pandas())
    cnt    = cudf.Series(cp.zeros(len(indeg), dtype=cp.int64),
                         index=indeg.index, name='cnt')
    cnt[leaf_set] = 1

    remaining_children = indeg.copy(deep=True)


    e_df  = G.view_edge_list()[['src', 'dst']]
    e_df.columns = ['child', 'parent']

    frontier = remaining_children[remaining_children == 0].index

    while len(frontier):
        f_cnt = cnt.loc[frontier].rename("leaf_from_child")
        contrib = (
            e_df.merge(f_cnt, left_on="child", right_index=True)
            .groupby("parent")["leaf_from_child"]
            .sum()
        )
        idx = contrib.index
        cnt_vals = cnt.loc[idx].values
        cnt.loc[idx] = cnt_vals + contrib.values

        processed_edges = e_df["child"].isin(frontier)
        parents_touched = e_df.loc[processed_edges, "parent"]

        child_done_cnt = parents_touched.value_counts()  # how many per parent
        idx = child_done_cnt.index
        remaining_children_vals = remaining_children.loc[idx].values
        remaining_children.loc[idx] = remaining_children_vals - child_done_cnt.values

        remaining_children.loc[frontier] = -1  # sentinel: already done

        frontier = remaining_children[remaining_children == 0].index

    total_leaves = int(cnt[leaf_set].sum())

    prob = cnt / total_leaves
    prob[cnt == 0] = 1.0  # isolated root(s): avoid log(0)

    ic = -cp.log(prob)
    return ic


# ---------------------------------------------------------------------------
# Ancestor cache helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _build_ancestor_cache(hash_key: int,
                         handle: ResourceHandle,
                         graph: pylibcugraph.SGGraph,
                         seeds: tuple[int, ...]) -> Dict[int, FrozenSet[int]]:
    """Run one multi‑source BFS and return {src: frozenset(ancestors)}.

    The *hash_key* must uniquely identify the *graph* for the lifetime of the
    cache (e.g. ``id(graph)`` or a user‑provided checksum).  By including it in
    the lru_cache arguments we automatically invalidate the cache when the
    caller switches to a different graph instance.
    """

    ancestors_map: Dict[int, FrozenSet[int]] = {}

    # NB: `cp.iinfo` works for both int32 and int64 distances returned by BFS
    for seed in seeds:
        dist, _pred, verts = pylibcugraph.bfs(
            handle,
            graph,
            cudf.Series([seed]),  # single source
            False,  # direction_optimizing
            -1,  # depth_limit
            False, False  # (legacy) normalize / do_expensive_check
        )

        max_val = cp.iinfo(dist.dtype).max
        reachable = verts[dist < max_val]

        # Move the reachable-vertex vector to host and store as an immutable set
        ancestors_map[int(seed)] = frozenset(cp.asnumpy(reachable))

    return ancestors_map


# ---------------------------------------------------------------------------
# Resnik helpers
# ---------------------------------------------------------------------------

def _mica_ic(ic_map: cp.ndarray, anc_a: FrozenSet[int], anc_b: FrozenSet[int]) -> float:
    """Return IC of the most informative common ancestor (MICA).

    *ic_map* is a CuPy array indexed by integer concept id.  Using a dense array
    gives true O(1) lookup inside the GPU intersection kernel below.
    """
    common = anc_a & anc_b
    if not common:
        return 0.0
    idx = cp.asarray(list(common))
    return float(ic_map[idx].max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resnik_bma(graph: cugraph.Graph,
                    handle: cugraph.Graph,
                    setA: Set[int],
                    setB: Set[int],
                    ic_map: cp.ndarray,
                    *,
                    graph_key: int = None) -> float:
    """Compute Resnik Best‑Match Average (BMA) between *setA* and *setB*.

    Parameters
    ----------
    graph, handle
        The cuGraph graph object and its resource handle.
    setA, setB
        Sets of integer concept ids (e.g. CUIs).
    ic_map
        Dense array mapping concept id ➜ information content.
    graph_key
        Optional stable identifier for *graph*; defaults to ``id(graph)``.  Use
        this if you frequently rebuild the graph object but the topology is the
        same (to leverage the cache).
    """
    if not setA or not setB:
        return 0.0

    # ------------------------------------------------------------------
    # 1. Ancestor cache (single GPU traversal for |A∪B| seeds)
    # ------------------------------------------------------------------
    seeds = tuple(sorted(setA | setB))
    key = graph_key if graph_key is not None else id(graph)
    ancestor_map = _build_ancestor_cache(key, handle, graph, seeds)

    # ------------------------------------------------------------------
    # 2. Vectorised best‑match max per row/col
    # ------------------------------------------------------------------
    def best_match(s1: Set[int], s2: Set[int]) -> cp.ndarray:
        bm = cp.empty(len(s1), dtype=cp.float32)
        for i, a in enumerate(s1):
            # Compute IC scores against all *s2*; keep max
            scores = cp.asarray([
                _mica_ic(ic_map, ancestor_map[a], ancestor_map[b]) for b in s2
            ], dtype=cp.float32)
            bm[i] = scores.max() if scores.size else 0.0
        return bm

    scores_row = best_match(setA, setB)
    scores_col = best_match(setB, setA)

    return 0.5 * (scores_row.mean() + scores_col.mean())

def bfs_level_dag(
    G: cugraph.Graph,
    seed_cuis,
    max_depth
):

    sg, handle = knowledge_graph.graph_utils.construct_gpu_graph(G)

    seeds = cudf.Series(cp.asarray(list(seed_cuis), dtype=cp.int32), name="sources")
    bfs_df = bfs(handle, sg, seeds, False, max_depth, False, False)
    distances = bfs_df[0]
    levels = cudf.Series(distances, name="distance")  # index = vertex id
    reached = levels[levels != cp.iinfo(distances.dtype).max].index  # vertices actually reached                                  # v

    edges = G.view_edge_list()[["src", "dst"]]

    mask = edges["src"].isin(reached) & edges["dst"].isin(reached)
    edges = edges[mask]

    # join once to grab levels; no alignment problem because both columns are unique
    edges = (
        edges.merge(levels.rename("lvl_src"), left_on="src", right_index=True)
             .merge(levels.rename("lvl_dst"), left_on="dst", right_index=True)
    )

    dag_edges = edges[edges["lvl_dst"] > edges["lvl_src"]][["src", "dst"]]

    dag_edges = dag_edges.drop_duplicates(subset=["src", "dst"], ignore_index=True)

    unique_nodes = list(cudf.concat([dag_edges["src"], dag_edges["dst"]]).unique().values_host)
    unique_nodes.sort()

    old_to_new = {int(old): idx for idx, old in enumerate(unique_nodes)}
    new_to_old = {idx: int(old) for idx, old in enumerate(unique_nodes)}

    dag_edges["src"] = dag_edges["src"].map(old_to_new).dropna()
    dag_edges["dst"] = dag_edges["dst"].map(old_to_new).dropna()

    G_dag = cugraph.Graph(directed=True)
    G_dag.from_cudf_edgelist(dag_edges, source="src", destination="dst", renumber=False)

    return G_dag, old_to_new, new_to_old

def construct_ic_graph_wrapper(mrhier_path, keep_cuis, max_expl_depth) -> ICGraphWrapper:
    """
    Build an IC-weighted GPU graph from the UMLS MRHIER table
    (instead of the pruned MRREL table used previously).

    Parameters
    ----------
    mrhier_path : str
        Path to `MRHIER.RRF` (pipe-delimited, *no* header row).
    keep_cuis : Collection[str] | None
        CUIs that **must** appear in the final graph (isolated self-loops
        are added if a CUI has no hierarchical neighbours).
    max_expl_depth : int
        Maximum BFS depth to explore from each `keep_cui`.

    Returns
    -------
    ICGraphWrapper
        Wrapper with the sub-DAG, RAPIDS handle, intrinsic-IC scores
        and CUI ↔ vertex-id maps.
    """
    # ------------------------------------------------------------------
    # 1.  Read MRHIER and extract child-parent CUI pairs
    # ------------------------------------------------------------------
    COLS = ["CUI", "AUI", "CXN", "PAUI", "SAB", "RELA",
            "PTR", "HCD", "HXD", "HSG"]
    mrhier = cudf.read_csv(
        mrhier_path,
        sep="|",
        names=COLS,
        usecols=["CUI", "AUI", "PAUI", "SAB"],
        dtype={"CUI": "str", "AUI": "str", "PAUI": "str", "SAB": "str"}
    ).to_pandas()

        # map AUI → CUI (needed to obtain the parent's CUI from PAUI)
    aui_to_cui = (
        mrhier[["AUI", "CUI"]]
        .drop_duplicates(subset="AUI", keep="first")
        .set_index("AUI")["CUI"]
    )

    # merge to get parent CUI; rows with unknown PAUI are dropped
    mrhier["parent"] = mrhier["PAUI"].map(aui_to_cui)
    mrhier = mrhier.dropna(subset=["parent"])

    # child / parent (exclude self-loops for now)
    mrhier = mrhier[mrhier["CUI"] != mrhier["parent"]]
    mrhier["child"] = mrhier["CUI"]



    # ------------------------------------------------------------------
    # 2.  Convert to cuDF and build an edge list
    # ------------------------------------------------------------------
    edges = cudf.from_pandas(
        mrhier[["child", "parent"]].rename(columns={"child": "src", "parent": "dst"})
    )

    # ------------------------------------------------------------------
    # 3.  Create contiguous vertex ids
    # ------------------------------------------------------------------
    id_map = cudf.DataFrame({"CUI": cudf.concat([edges["src"], edges["dst"]]).unique()})
    id_map["vid"] = cp.arange(len(id_map), dtype="int32")

    new_vids = None
    if keep_cuis is not None:
        all_keep = cudf.Series(list(keep_cuis), dtype="str")
        missing = all_keep[~all_keep.isin(id_map["CUI"])]
        if len(missing) > 0:
            start = int(id_map["vid"].max()) + 1
            new_vids = cp.arange(start, start + len(missing), dtype="int32")
            new_id_map = cudf.DataFrame({"CUI": missing, "vid": new_vids})
            id_map = cudf.concat([id_map, new_id_map], ignore_index=True)

    cui_to_vid = dict(zip(id_map["CUI"].values_host, id_map["vid"].values_host))
    vid_to_cui = {v: k for k, v in cui_to_vid.items()}

    edges = edges.assign(
        src=edges["src"].map(cui_to_vid),
        dst=edges["dst"].map(cui_to_vid)
    )

    # add isolated self-loops for keep_cuis that had no neighbours
    if new_vids is not None and len(new_vids) > 0:
        isolated = cudf.DataFrame({"src": new_vids, "dst": new_vids})
        edges = cudf.concat([edges, isolated], ignore_index=True)

    # ------------------------------------------------------------------
    # 4.  Build the GPU graph
    # ------------------------------------------------------------------
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=False)

    keep_vids = {cui_to_vid[cui] for cui in keep_cuis if cui in cui_to_vid}

    # ------------------------------------------------------------------
    # 5.  Extract the BFS-level DAG around the keep CUIs and compute IC
    # ------------------------------------------------------------------
    #G_ic, old_to_new, new_to_old = bfs_level_dag(G, keep_vids, max_expl_depth)
    #cui_to_vid = {cui: old_to_new[vid] for cui, vid in cui_to_vid.items() if vid in old_to_new}
    #vid_to_cui = {new_vid: vid_to_cui[old_vid] for new_vid, old_vid in new_to_old.items()}

    ic_scores = _intrinsic_ic(G)

    sg, handle = knowledge_graph.graph_utils.construct_gpu_graph(G)

    return ICGraphWrapper(sg, handle, ic_scores,
                          cui_to_vid=cui_to_vid,
                          vid_to_cui=vid_to_cui)



def _mean_ic(concepts, ic_scores):
    return sum(ic_scores[c] for c in concepts) / len(concepts)


'''
data = initialize_data(csv_path="/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_balanced_train.csv",
                       image_base_path="vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/",
                       split_value=None,
                       split_column="split",
                       report_column="impression",
                       id_column="dicom_id",
                       image_path_column="path")
'''




