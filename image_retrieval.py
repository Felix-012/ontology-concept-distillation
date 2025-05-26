import sys
from pathlib import Path
from typing import Dict, Tuple, Union
import random

import faiss
import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer

from data_utils import initialize_data
from knowledge_graph.graph_utils import prepare_graph, construct_gpu_graph, k_closest_reference_reports, topk_indices, \
    preprocess_mentions
from knowledge_graph.ic_metrics import construct_ic_graph_wrapper
from knowledge_graph.ner import get_mentions, ClinicalEntityLinker
from knowledge_graph.similarity_search import load_index, embed_texts, build_index, save_index, knn


def fill_low_tail_classes(unbalanced_path: Union[str, Path],
                          retrieval_path: Union[str, Path],
                          balanced_path: Union[str, Path],
                          config_path: Union[str, Path],
                          anomaly: str = "Consolidation") -> pd.DataFrame:

    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    id_column = config["init_data_args"]["id_column"]
    report_column = config["init_data_args"]["report_column"]
    image_path_column = config["init_data_args"]["image_path_column"]
    batch_size = config["hyperparameters"]["batch_size"]

    print(f"Score function: {config['retrieval_args']['cost_function']}")
    if config["retrieval_args"]["cost_function"] != "cosine_similarity":
        linker = ClinicalEntityLinker.from_default_constants()
    reports = []
    df_unbalanced = pd.read_csv(unbalanced_path)
    df_retrieval = pd.read_csv(retrieval_path)
    df_balanced = pd.read_csv(balanced_path)

    new_rows = df_balanced[
        ~df_balanced['dicom_id'].isin(df_retrieval['dicom_id'])
        & df_balanced['Finding Labels'].str.contains(anomaly, na=False)
        ]
    n_to_retrieve = len(new_rows)

    df_retrieval = pd.concat([df_retrieval, new_rows], ignore_index=True)


    retrieval_data = initialize_data(**config["init_data_args"], df=df_retrieval)
    unbalanced_data = initialize_data(**config["init_data_args"], df=df_unbalanced)
    if config["retrieval_args"]["cost_function"] == "cosine_similarity":
        if config["cosine_similarity"]["subfolder"]:
            tokenizer = CLIPTokenizer.from_pretrained(config["cosine_similarity"]["model"], subfolder="tokenizer")
            model = CLIPTextModel.from_pretrained(config["cosine_similarity"]["model"], subfolder="text_encoder")
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["cosine_similarity"]["model"], trust_remote_code=True)
            model = AutoModel.from_pretrained(config["cosine_similarity"]["model"], trust_remote_code=True)
        if Path(config["cosine_similarity"]["index_path"]).exists():
            index = load_index(config["cosine_similarity"]["index_path"])
        else:
            corpus_vecs = embed_texts(
                retrieval_data["reports"],
                tokenizer,
                model,
                config["hyperparameters"]["max_length"],
                batch_size,
                "cuda",
            )
            index = build_index(corpus_vecs)
            save_index(index, config["cosine_similarity"]["index_path"])
    else:
        unbalanced_reports, neg_unbalanced_reports, _ = preprocess_mentions(
            get_mentions(linker, config["mentions_path"], df_unbalanced[report_column]))
        additional_cuis =  set().union(*unbalanced_reports).union(*neg_unbalanced_reports)
        wrapper = prepare_graph(config["retrieval_mentions_path"],
                                retrieval_data["reports"],
                                config["mrrel_path"],
                                retrieval_data["ids"],
                                linker,
                                additional_cuis=additional_cuis)

        retrieval_reports = wrapper.report_list.copy()
        neg_retrieval_reports = wrapper.neg_report_list.copy()


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

    add_data = []
    df_anomaly = df_unbalanced[df_unbalanced[anomaly].fillna(False)]
    data = initialize_data(**config["init_data_args"], df=df_anomaly)
    batch_size = min(batch_size, len(data["reports"]))
    for start in range(0, n_to_retrieve, batch_size):
        target_indices_anomaly = [(start + k) % len(data["reports"]) for k in range(batch_size)]
        target_indices = list(df_anomaly.iloc[target_indices_anomaly].index)
        if config["retrieval_args"]["cost_function"] == "cosine_similarity":
            query_vec = embed_texts([data["reports"][i] for i in target_indices_anomaly],
                                    tokenizer,
                                    model,
                                    config["hyperparameters"]["max_length"],
                                    batch_size, "cuda")
            scores, indices = knn(index, query_vec, config["hyperparameters"]["k"])
            indices = [index for inner_indices in indices for index in inner_indices]
            indices = np.asarray(indices)
            index.remove_ids(indices)

            add_data.append(df_retrieval[image_path_column].iloc[indices])
            reports.append((indices[0], df_retrieval[report_column].iloc[indices].values[0]))
            acc = get_accuracy(df_retrieval, [report[0] for report in reports], anomaly)
            sys.stdout.write("\rAccuracy %f Length %i" % (acc, len(reports)))
            sys.stdout.flush()
        else:
            mapped_reports = []
            mapped_neg_reports = []
            mapped_targets = []
            mapped_neg_targets = []
            for idx in target_indices:
                target_curr = unbalanced_reports[idx]
                neg_target_curr = neg_unbalanced_reports[idx]
                mapped_reports.append((unbalanced_data["ids"][idx], retrieval_reports))
                mapped_neg_reports.append((unbalanced_data["ids"][idx], neg_retrieval_reports))
                mapped_targets.append((unbalanced_data["ids"][idx], target_curr))
                mapped_neg_targets.append((unbalanced_data["ids"][idx], neg_target_curr))

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
                linker=linker,
                ic_graph_wrapper=ic_graph_wrapper,
                ids_to_index=wrapper.id_to_index,
                #neg_target=mapped_neg_targets,
                #neg_reference_reports=mapped_neg_reports,
                #neg_set_to_ids = wrapper.neg_set_to_id
            )


            mapped_indices = topk_indices(results, wrapper.id_to_index, k=config["hyperparameters"]["k"])
            for target_idx in target_indices:
                indices = mapped_indices[unbalanced_data["ids"][target_idx]]
                add_data.append(df_retrieval[image_path_column].iloc[indices])
                reports.append((indices[0], df_retrieval[report_column].iloc[indices].values[0]))
                acc = get_accuracy(df_retrieval, [report[0] for report in reports], anomaly)
                sys.stdout.write("\rAccuracy %f Length %i" % (acc, len(reports)))
                sys.stdout.flush()
                used_ids = {df_retrieval.iloc[i][id_column] for i in indices}
                retrieval_reports = [report for report in retrieval_reports if len(set(wrapper.set_to_id
                [report]) & used_ids) == 0]


    for image_path in add_data:
        df_unbalanced.loc[len(df_unbalanced), ['path', 'Finding Labels']] = [image_path.values[0], anomaly]
    if config["retrieval_args"]["cost_function"] == "cosine_similarity":
        model_name = config["cosine_similarity"]["type"]
    else:
        model_name = "No"
    df_unbalanced.to_csv(f"{config['output_dir']}metric={config['retrieval_args']['cost_function']}_model={model_name}_{config['output_suffix']}")

    return df_unbalanced


def get_accuracy(df, indices, label):
    hits = 0
    for index in indices:
        if label in df.iloc[index]["Finding Labels"]:
            hits += 1
    return hits/len(indices)




if __name__ == "__main__":
    anomaly = "Consolidation"
    df_filled = fill_low_tail_classes(f"/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_8_train_unbalanced_{anomaly}.csv",
                          f"/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_8_balanced_retrieve.csv",
                                      "/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_8_balanced_train.csv",
                          f"/vol/ideadata/ce90tate/knowledge_graph_distance/config_{anomaly}.yml",
                                      anomaly=anomaly)














