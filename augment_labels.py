import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.ner import ClinicalEntityLinker, get_mentions, choose_label

CSV_FILE_PATH1 = "/vol/ideadata/ed52egek/pycharm/longtail/knowledge_graph_distance/splits/longtail_8_balanced_train.csv"
CSV_FILE_PATH2 = "/vol/ideadata/ce90tate/knowledge_graph_distance/splits/labeled_best_match.csv"
CONFIG_PATH = "/vol/ideadata/ce90tate/knowledge_graph_distance/configs/config.yml"
MENTIONS_PATH = "/vol/ideadata/ce90tate/data/umls/balanced_final_uncertain.pkl"

df1 = pd.read_csv(CSV_FILE_PATH1)
df2 = pd.read_csv(CSV_FILE_PATH2)

with open(CONFIG_PATH) as stream:
    config = yaml.safe_load(stream)
data1 = initialize_data(**config["init_data_args"], df=df1)
data2 = initialize_data(**config["init_data_args"], df=df2)
linker = ClinicalEntityLinker.from_default_constants()
mentions = get_mentions(linker, MENTIONS_PATH, data1["reports"])
labels1 = df1["Finding Labels"]
labels2 = df2["Finding Labels"]
augmented_labels = choose_label(mentions, labels1, labels2, linker)
df1.drop(columns=["Finding Labels"])
df1["Finding Labels"] = augmented_labels

df1.to_csv("/vol/ideadata/ce90tate/knowledge_graph_distance/splits/labels_aug_pop_chex_jac_set.csv")