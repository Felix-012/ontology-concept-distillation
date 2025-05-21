import random

from data_utils import initialize_data
from knowledge_graph.graph_utils import construct_gpu_graph, topk_indices, \
    prepare_graph, k_closest_reference_reports
from knowledge_graph.ic_metrics import construct_ic_graph_wrapper
from knowledge_graph.ner import get_mentions
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich import box


CSV_PATH = "/vol/ideadata/ce90tate/knowledge_graph_distance/splits/longtail_balanced_train.csv"
IMAGE_BASE_PATH = "vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
SPLIT_VALUE = "3"
SPLIT_COLUMN = "split"
REPORT_COLUMN = "impression"
ID_COLUMN = "dicom_id"
IMAGE_PATH_COLUMN = "path"

MENTIONS_PATH = "/vol/ideadata/ce90tate/data/umls/mentions_mimic_lt_balanced_train.pkl"
MRREL_PATH="/vol/ideadata/ce90tate/data/umls/2024AB/META/MRREL.RRF"
K=5
MAX_EXPL_DEPTH=5

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

keep_cuis = set().union(*report_list).union(*neg_report_list)

ic_graph_wrapper = construct_ic_graph_wrapper(MRREL_PATH, keep_cuis, MAX_EXPL_DEPTH)

mapped_reports = []
mapped_targets = []

for idx in target_indices:
    rep_curr = report_list.copy()
    target_curr = rep_curr.pop(idx)
    mapped_reports.append((data["ids"][idx], rep_curr))
    mapped_targets.append((data["ids"][idx], target_curr))

sg, handle = construct_gpu_graph(wrapper.graph)
#cost_functions = ["umls", "jac", "dice", "overlap", "depth"]
cost_functions = ["lin"]
overview = defaultdict(dict)

for cost_function in cost_functions:
    results = k_closest_reference_reports(
        handle,
        sg,
        wrapper.cui_to_vid,
        wrapper.vid_to_cui,
        mapped_reports,
        mapped_targets,
        wrapper.set_to_id,
        k=K,
        cost_function=cost_function,
        max_depth=MAX_EXPL_DEPTH,
        ic_graph_wrapper=ic_graph_wrapper
    )

    indices = topk_indices(results, wrapper.id_to_index, k=K)

    for batch_pos, target_idx in enumerate(target_indices):
        overview[target_idx][cost_function] = indices[data["ids"][target_idx]]


console = Console(width=250)

for target_idx in target_indices:
    title = f"Target {target_idx} • Impression: {impressions[target_idx]}"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=True, expand=True)
    table.add_column("Metric", style="bold magenta")
    for rank in range(1, K + 1):
        table.add_column(f"#{rank}", overflow="fold")


    for metric in cost_functions:
        row = [metric]
        for ref_id in overview[target_idx].get(metric, []):
            imp = impressions[ref_id]
            snippet = imp
            row.append(snippet)
        table.add_row(*row)

    console.print(table)




