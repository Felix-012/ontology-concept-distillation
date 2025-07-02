from __future__ import annotations

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List

import cudf
import cugraph
import cupy as cp
from pylibcugraph import bfs

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


# Global cache: graph_id → {child: [parent, …]}
_parent_index_cache: Dict[int, Dict[int, List[int]]] = {}


def _build_parent_index(graph: cugraph.Graph) -> Dict[int, List[int]]:
    """Return a mapping *child → [parent, …]* built from the edges of *graph*.

    Expects each edge to point **child → parent**.  If your graph stores the
    opposite direction, call ``graph.reverse(copy=False)`` first or adapt this
    function.
    """
    df = graph.view_edge_list()
    src_h = df['src'].values_host
    dst_h = df['dst'].values_host

    out: Dict[int, List[int]] = defaultdict(list)
    for child, parent in zip(src_h, dst_h):
        out[int(child)].append(int(parent))
    return out


def _ancestors(cui: int, parent_index: Dict[int, List[int]]) -> FrozenSet[int]:
    """Inclusive ancestor set of *cui*."""
    visited: Set[int] = set()
    stack: List[int] = [cui]
    while stack:
        v = stack.pop()
        for p in parent_index.get(v, ()):
            if p not in visited:
                visited.add(p)
                stack.append(p)
    visited.add(cui)
    return frozenset(visited)


def _mica_ic(
    ic_map: cp.ndarray,
    anc_a: FrozenSet[int],
    anc_b: FrozenSet[int],
) -> float:
    """Information content of the most informative common ancestor (MICA)."""
    common = anc_a & anc_b
    if not common:
        return 0.0
    idx = cp.asarray(list(common))
    return float(ic_map[idx].max())

def resnik_bma(
    graph: cugraph.Graph,
    setA: Set[int],
    setB: Set[int],
    ic_map: cp.ndarray,
    *,
    graph_key: int  = None,
) -> float:
    """Compute Resnik Best-Match Average (BMA) without BFS.

    Parameters
    ----------
    graph
        cuGraph **directed** ontology graph (*child → parent*).
    setA, setB
        Sets of integer concept ids (e.g. CUIs).
    ic_map
        Dense CuPy array mapping concept id → information content.
    graph_key
        Stable id for *graph*.  Defaults to ``id(graph)``; supply your own if
        you rebuild the graph object frequently but the topology is unchanged.
    """
    if not setA or not setB:
        return 0.0

    key = graph_key if graph_key is not None else id(graph)
    if key not in _parent_index_cache:
        _parent_index_cache[key] = _build_parent_index(graph)
    parent_index = _parent_index_cache[key]

    # Per-call cache avoids repeated global LRU look-ups inside the hot loop
    anc_cache: Dict[int, FrozenSet[int]] = {}

    def ancestors(cui: int) -> FrozenSet[int]:
        if cui in anc_cache:
            return anc_cache[cui]
        fs = _ancestors(cui, parent_index)
        anc_cache[cui] = fs
        return fs

    def best_match(row: List[int], col: List[int]) -> cp.ndarray:
        bm = cp.empty(len(row), dtype=cp.float32)
        anc_col = [ancestors(b) for b in col]  # pre-gather
        for i, a in enumerate(row):
            anc_a = ancestors(a)
            scores = cp.asarray(
                [_mica_ic(ic_map, anc_a, anc_b) for anc_b in anc_col],
                dtype=cp.float32,
            )
            bm[i] = scores.max() if scores.size else 0.0
        return bm

    scores_row = best_match(list(setA), list(setB))
    scores_col = best_match(list(setB), list(setA))
    return float(0.5 * (scores_row.mean() + scores_col.mean()))

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


    edges = cudf.from_pandas(
        mrhier[["child", "parent"]].rename(columns={"child": "src", "parent": "dst"})
    )

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

    if new_vids is not None and len(new_vids) > 0:
        isolated = cudf.DataFrame({"src": new_vids, "dst": new_vids})
        edges = cudf.concat([edges, isolated], ignore_index=True)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=False)

    keep_vids = {cui_to_vid[cui] for cui in keep_cuis if cui in cui_to_vid}


    ic_scores = _intrinsic_ic(G)

    return ICGraphWrapper(G, None, ic_scores,
                          cui_to_vid=cui_to_vid,
                          vid_to_cui=vid_to_cui)



def _mean_ic(concepts, ic_scores):
    return sum(ic_scores[c] for c in concepts) / len(concepts)





