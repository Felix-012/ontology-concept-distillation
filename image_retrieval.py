import argparse
from pathlib import Path
import io
from typing import Union, List
import contextlib
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

ANOMALIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "No Finding",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def fill_low_tail_classes(unbalanced_path: Union[str, Path],
                          retrieval_path: Union[str, Path],
                          balanced_path: Union[str, Path],
                          config_path: Union[str, Path],
                          anomalies: List[str]) -> pd.DataFrame:

    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    id_column = config["init_data_args"]["id_column"]
    report_column = config["init_data_args"]["report_column"]
    image_path_column = config["init_data_args"]["image_path_column"]
    batch_size = config["hyperparameters"]["batch_size"]

    print(f"Score function: {config['retrieval_args']['cost_function']}")
    print(f"alpha: {config['hyperparameters']['alpha']}")
    print(f"beta: {config['hyperparameters']['beta']}")
    print(f"Anomalies: {anomalies}")
    if config["retrieval_args"]["cost_function"] != "cosine_similarity":
        linker = ClinicalEntityLinker.from_config_path(config_path)
    reports = {anomaly: [] for anomaly in anomalies}
    df_unbalanced = pd.read_csv(unbalanced_path)
    df_retrieval = pd.read_csv(retrieval_path)
    df_balanced = pd.read_csv(balanced_path)

    new_rows_by_anom = {}
    for anom in anomalies:
        rows = df_balanced[
            ~df_balanced['dicom_id'].isin(df_retrieval['dicom_id'])
            & ~df_balanced['dicom_id'].isin(df_unbalanced['dicom_id'])
            & df_balanced['Finding Labels'].str.contains(anom, na=False)]
        new_rows_by_anom[anom] = rows
        df_retrieval = pd.concat([df_retrieval, rows], ignore_index=True)


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
        reference_ids = retrieval_data["ids"].copy()
        id_to_index_retrieval = dict(zip(reference_ids, list(range(len(reference_ids)))))
        retrieval_reports = wrapper.report_list.copy()
        neg_retrieval_reports = wrapper.neg_report_list.copy()


        ic_graph_wrapper = None
        if config["retrieval_args"]["use_ic"]:
            keep_cuis = (set()
                         .union(*wrapper.report_list)
                         .union(*wrapper.neg_report_list)
                         .union(unbalanced_reports)
                         .union(neg_unbalanced_reports))
            ic_graph_wrapper = construct_ic_graph_wrapper(config["mrhier_path"],
                                                          keep_cuis,
                                                          config["hyperparameters"]["max_expl_depth"])

    add_data = []
    # prepare per-anomaly unbalanced data
    anomaly_data = {anom: initialize_data(**config["init_data_args"], df = df_unbalanced[df_unbalanced[anom].fillna(False)])
                    for anom in anomalies}
    # round-robin over anomalies until we've pulled all n_to_retrieve
    # for each anomaly: a wrap-around pointer, and how many retrieves remain
    ptr_counts = {anom: 0 for anom in anomalies}
    remaining_counts = {anom: len(new_rows_by_anom[anom]) for anom in anomalies}
    acc_list = {anom: [] for anom in anomalies}

    # keep going until we've done exactly `remaining_counts[anom]` retrieves per anomaly
    while sum(remaining_counts.values()) > 0:
        for anom in anomalies:
            if remaining_counts[anom] == 0:
                continue
            # pick one original report in wrap-around fashion
            rpt_list = anomaly_data[anom]["reports"]
            ptr = ptr_counts[anom] % len(rpt_list)
            ptr_counts[anom] += 1
            remaining_counts[anom] -= 1

            # get the DataFrame index of that unbalanced example
            idx = df_unbalanced[df_unbalanced[anom].fillna(False)].index[ptr]
            target_indices = [idx]

            if config["retrieval_args"]["cost_function"] == "cosine_similarity":
                query_vec = embed_texts([anomaly_data[anom]["reports"][ptr]],
                                        tokenizer,
                                        model,
                                        config["hyperparameters"]["max_length"],
                                        batch_size, "cuda")

                with contextlib.redirect_stdout(io.StringIO()):
                    scores, indices = knn(index, query_vec, config["hyperparameters"]["k"])
                indices = [index for inner_indices in indices for index in inner_indices]
                indices = np.asarray(indices)
                index.remove_ids(indices)
                add_data.append((anom, df_retrieval[image_path_column].iloc[indices]))
                reports[anom].append((indices[0], df_retrieval[report_column].iloc[indices].values[0]))
                acc = get_accuracy(df_retrieval, [r[0] for r in reports[anom]], anom)
                acc_list[anom].append(acc)
                print(f"{acc} : {anom} : {len(acc_list[anom])}")

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
                    alpha=config["hyperparameters"]["alpha"],
                    beta=config["hyperparameters"]["beta"],
                    k=config["hyperparameters"]["k"],
                    cost_function=config["retrieval_args"]["cost_function"],
                    max_depth=config["hyperparameters"]["max_expl_depth"],
                    linker=linker,
                    ic_graph_wrapper=ic_graph_wrapper,
                    graph_wrapper=wrapper,
                    ids_to_index=id_to_index_retrieval,
                    neg_target=mapped_neg_targets,
                    neg_reference_reports=mapped_neg_reports,
                    neg_set_to_ids = wrapper.neg_set_to_id,
                    allow_empty_references=True
                )


                mapped_indices = topk_indices(results, wrapper.id_to_index, k=config["hyperparameters"]["k"])
                for target_idx in target_indices:
                    indices = mapped_indices[unbalanced_data["ids"][target_idx]]
                    add_data.append((anom, df_retrieval[image_path_column].iloc[indices]))
                    reports[anom].append((indices[0], df_retrieval[report_column].iloc[indices].values[0]))
                    #print(reports[-1])
                    acc = get_accuracy(df_retrieval, [report[0] for report in reports[anom]], anom)
                    acc_list[anom].append(acc)
                    print(f"{acc} : {anom} : {len(acc_list[anom])}")
                    used_ids = {df_retrieval.iloc[i][id_column] for i in indices}
                    for report in list(wrapper.set_to_id):  # snapshot of keys
                        ids = wrapper.set_to_id[report]
                        remaining = [i for i in ids if i not in used_ids]  # keep good IDs

                        if remaining:
                            wrapper.set_to_id[report] = remaining  # shrink in-place
                        else:
                            wrapper.set_to_id.pop(report)  # drop empty entry

                    retrieval_reports = [r for r in retrieval_reports
                                         if r in wrapper.set_to_id]

                    for report in list(wrapper.neg_set_to_id):  # snapshot of keys
                        ids = wrapper.neg_set_to_id[report]
                        remaining = [i for i in ids if i not in used_ids]  # keep good IDs

                        if remaining:
                            wrapper.neg_set_to_id[report] = remaining  # shrink in-place
                        else:
                            wrapper.neg_set_to_id.pop(report)  # drop empty entry

                    neg_retrieval_reports = [r for r in neg_retrieval_reports
                                         if r in wrapper.neg_set_to_id]

                    reference_ids = list(set(reference_ids) - used_ids)


                    id_to_index_retrieval = dict(zip(reference_ids, list(range(len(reference_ids)))))
                    #retrieval_reports = [report for report in retrieval_reports if len(set(wrapper.set_to_id
                    #[report]) & used_ids) == 0]


    for anom, image_path in add_data:
        df_unbalanced.loc[len(df_unbalanced), ['path', 'Finding Labels']] = [image_path.values[0], anom]
    if config["retrieval_args"]["cost_function"] == "cosine_similarity":
        model_name = config["cosine_similarity"]["type"]
    else:
        model_name = "No"
    #plt.yticks(np.arange(0, 1, 0.05))
    #plt.plot(acc_list)

    anomalies_stem = "_".join(anomalies)
    if config['retrieval_args']['cost_function'] in ["umls", "tversky", "sym_tversky"]:
       stem = (f"metric={config['retrieval_args']['cost_function']}_"
               f"a={config['hyperparameters']['alpha']}_"
               f"b={config['hyperparameters']['beta']}_"
               f"model={model_name}_{anomalies_stem}_{config['output_suffix']}")
    else:
        stem = (f"metric={config['retrieval_args']['cost_function']}_"f"model={model_name}_{anomalies_stem}_{config['output_suffix']}")

    # save per‐anomaly accuracy table
    df_acc = pd.DataFrame({anom: pd.Series(vals)
                           for anom, vals in acc_list.items()})
    df_acc.to_csv(
        f"{config['output_dir']}acc_list_{stem}.csv",
        index=False
    )

    # save the updated unbalanced set
    df_unbalanced.to_csv(f"{config['output_dir']}{stem}.csv", index=False)

    #plt.savefig(f"{stem}_plot.png")

    return df_unbalanced


def get_accuracy(df, indices, label):
    hits = 0
    for index in indices:
        if label in df.iloc[index]["Finding Labels"]:
            hits += 1
    return hits/len(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment an unbalanced split with retrieved examples "
                    "to obtain a better class distribution."
    )
    parser.add_argument("--unbalanced_csv", required=True,
                        help="Path to the original unbalanced CSV split.")
    parser.add_argument("--retrieval_csv", required=True,
                        help="Path to the retrieval set with candidate rows.")
    parser.add_argument("--train_csv", required=True,
                        help="Path to the balanced-train CSV (input baseline).")
    parser.add_argument("--config_path",
                        help="YAML config file for data initialisation.")
    parser.add_argument("--output", default=None,
                        help="Destination for the filled CSV. "
                             "If omitted, *_filled.csv is created next to --train_csv.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    df_filled = fill_low_tail_classes(
        args.unbalanced_csv,
        args.retrieval_csv,
        args.train_csv,
        args.config_path,
        anomalies=ANOMALIES,
    )

    # Resolve target path
    out_path = (
        Path(args.output)
        if args.output
        else Path(args.train_csv).with_name(
            f"{Path(args.train_csv).stem}_filled.csv"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_filled.to_csv(out_path, index=False)
    print(f"✔︎ Saved filled CSV to {out_path}")


if __name__ == "__main__":
    main()













