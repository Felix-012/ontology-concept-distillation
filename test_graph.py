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
K=1

data = initialize_data(CSV_PATH,
                       IMAGE_BASE_PATH,
                       SPLIT_VALUE,
                       SPLIT_COLUMN,
                       REPORT_COLUMN,
                       ID_COLUMN,
                       IMAGE_PATH_COLUMN)
impressions = data["reports"]

mentions = get_mentions(MENTIONS_PATH, impressions)

id_to_index = dict(zip(data["ids"], list(range(len(data["ids"])))))

wrapper = prepare_graph(MENTIONS_PATH, MRREL_PATH, id_to_index)

i= random.randint(0, len(mentions) - 1)

report_list = wrapper.report_list.copy()
neg_report_list = wrapper.neg_report_list.copy()

target_idx = i
target = report_list.pop(target_idx)

sg, handle = construct_gpu_graph(wrapper.graph)

results = k_closest_reference_reports(handle,
                                      sg,
                                      wrapper.cui_to_vid,
                                      wrapper.vid_to_cui,
                                      zip(data["ids"], report_list),
                                      [(data["ids"][i], target)],
                                      wrapper.set_to_indices,
                                      k=K)
indices = topk_indices(results, wrapper.id_to_index, k=K, shuffle=True)

print("Target idx:", target_idx)
rprint(f"[bold green]Target impression: {impressions[target_idx]}[/bold green]")
print("Target CUIs:", ", ".join(sorted(target)))

for id in indices.keys():
    rprint(f"[bold deep_pink4]closest reference reports and images for {id}:[/bold deep_pink4]")
    for index in indices[id]:
        rprint(f"[bold cyan]{impressions[index]}[/bold cyan]")
        rprint(f"[bold dark_cyan]{data['image_paths'][index]}[/bold dark_cyan]")

