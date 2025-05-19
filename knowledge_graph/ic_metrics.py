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


def _ancestors(graph, handle, vertex):
    bfs_df = bfs(handle, graph, cudf.Series(vertex), False, -1, False, False)
    mask = bfs_df[0] < cp.iinfo(bfs_df[0].dtype).max
    ancestors = bfs_df[2][mask]
    return frozenset(ancestors.get())

def _resnik(graph, handle, cui1, cui2, ic_map):
    anc1, anc2 = _ancestors(graph, handle, cui1), _ancestors(graph, handle, cui2)
    common = anc1 & anc2
    if not common:
        return 0.0
    return max(ic_map[n] for n in common)

def resnik_bma(graph, handle, setA, setB, ic_map):
    scores_row = [max(_resnik(graph, handle, a, b, ic_map) for b in setB) for a in setA]
    scores_col = [max(_resnik(graph, handle, b, a, ic_map) for a in setA) for b in setB]
    return 0.5 * (sum(scores_row) / len(setA) + sum(scores_col) / len(setB))

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

def construct_ic_graph_wrapper(mrrel_path, keep_cuis, max_expl_depth) -> ICGraphWrapper:
    HIER_REL = ["PAR", "CHD", "RN", "RB", ]
    cols = ["CUI1","AUI1","STYPE1","REL","CUI2","AUI2","STYPE2",
            "RELA","RUI","SRUI","SAB","SL","RG","DIR","SUPPRESS","CVF"]
    mrrel = cudf.read_csv(mrrel_path,
                          sep="|",
                          names=cols,
                          usecols=["CUI1", "REL", "CUI2", "SAB"],
                          dtype={"CUI1": "str", "CUI2": "str", "REL": "str", "SAB": "str"})

    mrrel = mrrel.to_pandas()

    mrrel = mrrel[mrrel.REL.isin(HIER_REL)]
    mrrel = mrrel[mrrel["CUI1"] != mrrel["CUI2"]]

    mrrel["child"] = mrrel["CUI1"]
    mrrel["parent"] = mrrel["CUI2"]

    mask_second_is_child = mrrel["REL"].isin(["CHD", "RB"])
    mrrel.loc[mask_second_is_child, ["child", "parent"]] = mrrel.loc[mask_second_is_child, ["CUI2", "CUI1"]].values

    mrrel = cudf.from_pandas(mrrel)

    edges = mrrel[["child", "parent"]].rename(columns={"child": "src", "parent": "dst"})

    id_map = cudf.DataFrame({
        "CUI": cudf.concat([edges["src"], edges["dst"]]).unique()
    })
    id_map["vid"] = cp.arange(len(id_map), dtype="int32")  # contiguous 0 … N-1

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
    vid_to_cui = dict(zip(id_map["vid"].values_host, id_map["CUI"].values_host))

    edges = edges.assign(
        src=edges["src"].map(cui_to_vid),
        dst=edges["dst"].map(cui_to_vid)
    )

    if new_vids is not None and len(new_vids) > 0:
        # create a self‐loop for every missing vid
        isolated_vertices = cudf.DataFrame({
            "src": new_vids,
            "dst": new_vids
        })
        # tack them onto your real edges
        edges = cudf.concat([edges, isolated_vertices], ignore_index=True)

    G = cugraph.Graph(directed=True)  # directed = True by default
    G.from_cudf_edgelist(edges,
                         source="src",
                         destination="dst",
                         renumber=False)

    keep_cuis = {cui_to_vid[cui] for cui in keep_cuis if cui in cui_to_vid.keys()}

    G_ic, old_to_new, new_to_old = bfs_level_dag(G, keep_cuis, max_expl_depth)
    cui_to_vid = {cui: old_to_new[old_vid] for cui, old_vid in cui_to_vid.items() if old_vid in old_to_new.keys()}
    vid_to_cui = {new_vid: vid_to_cui[old_vid] for new_vid, old_vid in new_to_old.items() if
                  old_vid in vid_to_cui.keys()}
    ic_scores = _intrinsic_ic(G_ic)

    sg, handle = knowledge_graph.graph_utils.construct_gpu_graph(G_ic)

    return ICGraphWrapper(sg, handle, ic_scores, cui_to_vid=cui_to_vid, vid_to_cui=vid_to_cui)


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




