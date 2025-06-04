import os
from collections import defaultdict
from typing import Optional, Union, Tuple, Dict, Set

import cudf, cugraph, cupy as cp
import pylibcugraph.graphs
from pylibcugraph import SGGraph


def build_graph(mrrel_path: Union[str, os.PathLike],
                keep_cuis: Optional[Set[str]] = None,
                require_both: bool = True
               ) -> Tuple[cugraph.Graph, cudf.DataFrame, Dict[Tuple[int, int], list[str]]]:
    wanted_rels = ["SY", "CHD", "RQ", "RL", "PAR", "RB", "RN"]

    rel = cudf.read_csv(
        mrrel_path,
        sep="|",
        header=None,
        usecols=[0, 3, 4],
        names=["CUI1", "REL", "CUI2"],
        dtype="str"
    )

    mask_not_self = rel["CUI1"] != rel["CUI2"]
    mask_wanted_rel = rel["REL"].isin(wanted_rels)
    rel = rel[mask_not_self & mask_wanted_rel].drop_duplicates(["CUI1", "CUI2", "REL"])
    #rel = rel[mask_not_self].drop_duplicates(["CUI1", "CUI2", "REL"])

    if keep_cuis is not None:
        keep_series = cudf.Series(list(keep_cuis), dtype="str")
        mask = rel["CUI1"].isin(keep_series)
        mask &= rel["CUI2"].isin(keep_series) if require_both else rel["CUI2"].isin(keep_series)
        rel = rel[mask]

    id_map = cudf.DataFrame({
        "CUI": cudf.concat([rel["CUI1"], rel["CUI2"]]).unique()
    })
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


    relations_df = rel.to_pandas()
    edge_rel_map = defaultdict(list)
    for src, r, dst  in relations_df.itertuples(index=False):
        key = tuple(sorted((src, dst)))
        edge_rel_map[key].append(r)


    rel = rel.merge(id_map.rename(columns={"CUI": "CUI1", "vid": "src"}), on="CUI1") \
             .merge(id_map.rename(columns={"CUI": "CUI2", "vid": "dst"}), on="CUI2") \
             [["src", "dst", "REL"]]

    if new_vids is not None and len(new_vids) > 0:
        isolated_vertices = cudf.DataFrame({
            "src": new_vids,
            "dst": new_vids
        })
        for v in map(int, new_vids):
            edge_rel_map[(v, v)].append("SELF")
        rel = cudf.concat([rel[["src", "dst"]], isolated_vertices], ignore_index=True)
    else:
        rel = rel[["src", "dst"]]

    G = cugraph.Graph()
    G.from_cudf_edgelist(rel, source="src", destination="dst")

    return G, id_map, dict(edge_rel_map)


def induced_subgraph_view(
    graph: SGGraph,
    vertices,
    *,
    resource_handle: Optional[pylibcugraph.ResourceHandle] = None,
):

    if not isinstance(graph, pylibcugraph.SGGraph):
        raise TypeError("`graph` must be a pylibcugraph.SGGraph (single‑GPU)")

    rh = resource_handle or pylibcugraph.ResourceHandle()


    sub_offsets = cp.asarray([0, len(vertices)], dtype=vertices.dtype)

    src, dst, wgt, _ = pylibcugraph.induced_subgraph(
        rh,
        graph,
        vertices,
        sub_offsets,
        True,
    )

    props = pylibcugraph.GraphProperties()

    subgraph = pylibcugraph.SGGraph(
        rh,
        props,
        src,
        dst,
        weight_array=wgt if wgt is not None else None,
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )

    return subgraph


class GraphWrapper:
    def __init__(self,
                 graph: cugraph.Graph,
                 id_map: cudf.DataFrame,
                 set_to_indices,
                 report_list,
                 neg_report_list,
                 id_to_index,
                 set_to_id,
                 neg_set_to_id,
                 rel_dict):

        self.graph = graph
        self.id_map = id_map
        self.report_list = report_list
        self.neg_report_list = neg_report_list
        self.set_to_id = set_to_id
        self.neg_set_to_id = neg_set_to_id
        self.set_to_indices = set_to_indices
        self.id_to_index = id_to_index
        self.rel_dict = rel_dict
        self.depth_of = None
        self.pred_map = None
        self.cui_to_vid = dict(zip(id_map["CUI"].values_host, id_map["vid"].values_host))
        self.vid_to_cui = dict(zip(id_map["vid"].values_host, id_map["CUI"].values_host))




