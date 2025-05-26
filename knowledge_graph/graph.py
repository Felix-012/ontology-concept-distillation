import os
from typing import Optional, Union

import cudf, cugraph, cupy as cp
import pylibcugraph.graphs
from pylibcugraph import SGGraph


def build_graph(mrrel_path: Union[str, os.PathLike],
                keep_cuis: Optional[set[str]] = None,
                require_both: bool = True):
    #wanted_rels = ["SY", "PAR", "CHD", "RB", "RN", "RQ", "RL"]

    rel = cudf.read_csv(
        mrrel_path,
        sep="|",
        header=None,
        usecols=[0, 3, 4],  # 0=CUI1, 3=REL, 4=CUI2
        names=["CUI1", "REL", "CUI2"],
        dtype="str"
    )

    mask_not_self = rel["CUI1"] != rel["CUI2"]


    rel = (
        rel[mask_not_self]
        .drop_duplicates(["CUI1", "CUI2"])
        .drop(columns="REL")
    )

    if keep_cuis is not None:
        keep_series = cudf.Series(list(keep_cuis), dtype="str")
        mask = rel["CUI1"].isin(keep_series)
        mask &= (rel["CUI2"].isin(keep_series) if require_both
                 else rel["CUI2"].isin(keep_series))
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

    # merge to get src/dst columns
    rel = (
        rel.merge(id_map.rename(columns={"CUI": "CUI1", "vid": "src"}), on="CUI1")
           .merge(id_map.rename(columns={"CUI": "CUI2", "vid": "dst"}), on="CUI2")
           [["src", "dst"]]
    )
    if new_vids is not None and len(new_vids) > 0:
        # create a self‐loop for every missing vid
        isolated_vertices = cudf.DataFrame({
            "src": new_vids,
            "dst": new_vids
        })
        # tack them onto your real edges
        rel = cudf.concat([rel, isolated_vertices], ignore_index=True)

    # now build your undirected graph
    G = cugraph.Graph()
    G.from_cudf_edgelist(rel, source="src", destination="dst")

    return G, id_map


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
                 neg_set_to_id):

        self.graph = graph
        self.id_map = id_map
        self.report_list = report_list
        self.neg_report_list = neg_report_list
        self.set_to_id = set_to_id
        self.neg_set_to_id = neg_set_to_id
        self.set_to_indices = set_to_indices
        self.id_to_index = id_to_index
        self.cui_to_vid = dict(zip(id_map["CUI"].values_host, id_map["vid"].values_host))
        self.vid_to_cui = dict(zip(id_map["vid"].values_host, id_map["CUI"].values_host))




