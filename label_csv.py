from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.graph_utils import preprocess_mentions
from knowledge_graph.ner import ClinicalEntityLinker, get_mentions, create_labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=Path,
        help="Path to the *source* CSV file.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        help="Path where the output should be saved.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Path where the output should be saved.",
    )

    parser.add_argument(
        "--mentions_path",
        type=Path,
        help="Path where the output should be saved.",
    )
    return parser.parse_args()


def build_label_dataframe(label_dict, reports):
    all_labels = sorted({label for pairs in label_dict.values() for label, _ in pairs})
    to_code = {"present": 1, "absent": 0, "uncertain": -1}
    rows = []
    for dict_tuple, report in zip(label_dict.items(), reports):
        _id = dict_tuple[0]
        pairs = dict_tuple[1]
        codes = {label: to_code.get(assertion, pd.NA) for label, assertion in pairs}
        labels = []
        for label, code in codes.items():
            if code != 0:
                labels.append(label)
        if len(labels) == 0:
            labels = ["No Finding"]
        row = {
            "id": _id,
            "labels": "|".join(sorted(labels)),
            "reports": report
        }
        # fill each label column; leave NaN if label never mentioned for this id
        row.update({label: codes.get(label, pd.NA) for label in all_labels})
        rows.append(row)

    # ――― 3.  Assemble the dataframe (id, labels, LabelA, LabelB, …) ――― #
    ordered_cols = ["id", "labels", "reports"] + all_labels
    return pd.DataFrame(rows, columns=ordered_cols)

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.src)
    with open(args.config_path) as stream:
        config = yaml.safe_load(stream)
    data = initialize_data(**config["init_data_args"], df=df)
    linker = ClinicalEntityLinker.from_default_constants()
    mentions = get_mentions(linker, args.mentions_path, data["reports"])
    label_dict = create_labels(data["ids"], mentions, linker)

    label_df = build_label_dataframe(label_dict, data["reports"])

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(dst_path, index=False)

    print(f"Wrote labelled CSV to {dst_path}")




