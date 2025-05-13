import random

from data_utils import initialize_data
from knowledge_graph.graph_utils import construct_gpu_graph, topk_indices, \
    prepare_graph, k_closest_reference_reports
from knowledge_graph.ner import get_mentions

from rich import print as rprint

CSV_PATH = "/vol/ideadata/ce90tate/data/mimic/mimic_metadata_preprocessed.csv"
IMAGE_BASE_PATH = "vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
SPLIT_VALUE = "3"
SPLIT_COLUMN = "split"
REPORT_COLUMN = "impression"
ID_COLUMN = "dicom_id"
IMAGE_PATH_COLUMN = "path"

MENTIONS_PATH = "/vol/ideadata/ce90tate/data/umls/mentions_mimic_p19.pkl"
MRREL_PATH="/vol/ideadata/ce90tate/data/umls/2024AB/META/MRREL.RRF"
K=10

data = initialize_data(CSV_PATH,
                       IMAGE_BASE_PATH,
                       SPLIT_VALUE,
                       SPLIT_COLUMN,
                       REPORT_COLUMN,
                       ID_COLUMN,
                       IMAGE_PATH_COLUMN)
impressions = data["reports"]

mentions = get_mentions(MENTIONS_PATH, impressions)

wrapper = prepare_graph(MENTIONS_PATH, MRREL_PATH, data["ids"])

target_indices = random.sample(range(0, len(mentions)), 5)

report_list = wrapper.report_list.copy()
neg_report_list = wrapper.neg_report_list.copy()

mapped_reports = []
mapped_targets = []

for idx in target_indices:
    rep_curr = report_list.copy()
    target_curr = rep_curr.pop(idx)
    mapped_reports.append((data["ids"][idx], rep_curr))
    mapped_targets.append((data["ids"][idx], target_curr))

sg, handle = construct_gpu_graph(wrapper.graph)
for cost_function in ["umls", "jac"]:
    results = k_closest_reference_reports(handle,
                                          sg,
                                          wrapper.cui_to_vid,
                                          wrapper.vid_to_cui,
                                          mapped_reports,
                                          mapped_targets,
                                          wrapper.set_to_id,
                                          k=K,
                                          cost_function=cost_function,
                                          max_depth=4)
    indices = topk_indices(results, wrapper.id_to_index, k=K)
    for i, target_idx in enumerate(target_indices):
        print("Target idx:", target_idx)
        rprint(f"[bold green]Target impression: {impressions[target_idx]}[/bold green]")
        print("Target CUIs:", ", ".join(sorted(mapped_targets[i][1])))


        for index in indices[data["ids"][target_idx]]:
            rprint(f"[bold cyan]{impressions[index]}[/bold cyan]")
            print("Reference CUIs:", ", ".join(sorted(report_list[index])))
            #rprint(f"[bold dark_cyan]{data['image_paths'][index]}[/bold dark_cyan]")
            print("")




