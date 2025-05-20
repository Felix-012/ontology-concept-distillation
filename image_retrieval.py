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

    report_column = config["init_data_args"]["report_column"]
    image_path_column = config["init_data_args"]["image_path_column"]
    batch_size = config["hyperparameters"]["batch_size"]

    anomalies = config["anomalies"]

    retrieval_data = initialize_data(**config["init_data_args"], df=df_retrieval)
    unbalanced_data = initialize_data(**config["init_data_args"], df=df_unbalanced)
    unbalanced_reports, neg_unbalanced_reports, _ = preprocess_mentions(
        get_mentions(config["mentions_path"], df_unbalanced[report_column]))
    additional_cuis =  set().union(*unbalanced_reports).union(*neg_unbalanced_reports)
    wrapper = prepare_graph(config["retrieval_mentions_path"],
                            retrieval_data["reports"],
                            config["mrrel_path"],
                            retrieval_data["ids"],
                            additional_cuis=additional_cuis)

    retrieval_reports = wrapper.report_list.copy()


    ic_graph_wrapper = None
    if config["retrieval_args"]["use_ic"]:
        keep_cuis = (set()
                     .union(*wrapper.report_list)
                     .union(*wrapper.neg_report_list)
                     .union(unbalanced_reports)
                     .union(neg_unbalanced_reports))
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
        df_anomaly = df_unbalanced[df_unbalanced[anomaly].fillna(False)]
        data = initialize_data(**config["init_data_args"], df=df_anomaly)
        n_to_retrieve = n_max - len(df_anomaly)
        batch_size = min(batch_size, len(data["reports"]))
        for start in range(0, n_to_retrieve, batch_size):
            target_indices_anomaly = [(start + k) % len(data["reports"]) for k in range(batch_size)]
            target_indices = list(df_anomaly.iloc[target_indices_anomaly].index)
            mapped_reports = []
            mapped_targets = []
            for idx in target_indices:
                target_curr = unbalanced_reports[idx]
                mapped_reports.append((unbalanced_data["ids"][idx], retrieval_reports))
                mapped_targets.append((unbalanced_data["ids"][idx], target_curr))

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
                indices = mapped_indices[unbalanced_data["ids"][target_idx]]
                add_data[anomaly].append(df_retrieval[image_path_column].iloc[indices])
                indices = set(indices)
                retrieval_reports = [report for i, report in enumerate(retrieval_reports) if i not in indices]

        for image_path in add_data[anomaly]:
            df_unbalanced.loc[len(df_unbalanced), ['path', 'Finding Labels']] = [image_path.values[0], anomaly]
        df_unbalanced.to_csv("/vol/ideadata/ce90tate/knowledge_graph_distance/retrieved/metric=umls_consolidation_10.csv")

    return df_unbalanced



if __name__ == "__main__":
    df_filled = fill_low_tail_classes("/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_8_train_unbalanced_Consolidation.csv",
                          "/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_8_balanced_retrieve.csv",
                          "/vol/ideadata/ce90tate/knowledge_graph_distance/config.yml")














