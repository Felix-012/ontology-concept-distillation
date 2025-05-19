from pathlib import Path
from typing import Dict, Tuple, Union
import random
import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.graph_utils import prepare_graph, construct_gpu_graph, k_closest_reference_reports, topk_indices, \
    preprocess_mentions
from knowledge_graph.ic_metrics import construct_ic_graph_wrapper
from knowledge_graph.ner import get_mentions


def fill_low_tail_classes(unbalanced_path: Union[str, Path],
                          retrieval_path: Union[str, Path],
                          config_path: Union[str, Path]) -> pd.DataFrame:

    df_unbalanced = pd.read_csv(unbalanced_path)
    df_retrieval = pd.read_csv(retrieval_path)

    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    id_column = config["init_data_args"]["id_column"]
    report_column = config["init_data_args"]["report_column"]
    image_path_column = config["init_data_args"]["image_path_column"]
    batch_size = config["hyperparameters"]["batch_size"]

    anomalies = config["anomalies"]

    retrieval_data = initialize_data(**config["init_data_args"], df=df_retrieval)
    wrapper = prepare_graph(config["retrieval_mentions_path"], retrieval_data["reports"], config["mrrel_path"], retrieval_data["ids"])
    unbalanced_reports, _, _ = preprocess_mentions(get_mentions(config["mentions_path"], df_unbalanced[report_column]))

    retrieval_reports = wrapper.report_list.copy()
    keep_cuis = set().union(*wrapper.report_list).union(*wrapper.neg_report_list)

    ic_graph_wrapper = None
    if config["retrieval_args"]["use_ic"]:
        ic_graph_wrapper = construct_ic_graph_wrapper(config["mrrel_path"],
                                                      keep_cuis,
                                                      config["hyperparameters"]["max_expl_depth"])

    n_max = 0
    for anomaly in anomalies:
        n = len(df_unbalanced[df_unbalanced[anomaly]])
        n_max = max(n_max, n)

    add_data = {}
    for anomaly in anomalies:
        add_data[anomaly] = []
        df_anomaly = df_unbalanced[df_unbalanced[anomaly]]
        data = initialize_data(**config["init_data_args"], df=df_anomaly)
        n_to_retrieve = n_max - len(df_anomaly)
        for start in range(0, n_to_retrieve, batch_size):
            target_indices = [(start + k) % len(data) for k in range(batch_size)]
            mapped_reports = []
            mapped_targets = []
            for idx in target_indices:
                target_curr = unbalanced_reports[idx]
                mapped_reports.append((data["ids"][idx], retrieval_reports))
                mapped_targets.append((data["ids"][idx], target_curr))

            sg, handle = construct_gpu_graph(wrapper.graph)


            results = k_closest_reference_reports(
                handle,
                sg,
                wrapper.cui_to_vid,
                wrapper.vid_to_cui,
                mapped_reports,
                mapped_targets,
                wrapper.set_to_id,
                k=config["hyperparameters"]["k"],
                cost_function=config["retrieval_args"]["cost_function"],
                max_depth=config["hyperparameters"]["max_expl_depth"],
                ic_graph_wrapper=ic_graph_wrapper
            )

            mapped_indices = topk_indices(results, wrapper.id_to_index, k=config["hyperparameters"]["k"])
            for target_idx in target_indices:
                indices = mapped_indices[["ids"][target_idx]]
                add_data[anomaly].append(df_retrieval[id_column,report_column, image_path_column].iloc[indices])

        for index in add_data[anomaly]:
            df_unbalanced.loc[-1, ['path', 'Finding Labels']] = [data["image_paths"][index], anomaly]

    return df_unbalanced



if __name__ == "__main__":
    fill_low_tail_classes("/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_balanced_train.csv",
                          "/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_balanced_retrieve.csv",
                          "/vol/ideadata/ce90tate/knowledge_graph_distance/config.yml")














