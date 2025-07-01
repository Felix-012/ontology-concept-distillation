from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.ner import ClinicalEntityLinker, get_mentions, create_labels, create_labels_sim, CUIS


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

    parser.add_argument(
        "--use_uncertain",
        action="store_true",
        default=False,
        help="Path where the output should be saved.",
    )
    return parser.parse_args()


def build_label_dataframe(label_dict, reports, use_uncertain):
    all_labels = sorted({label for pairs in label_dict.values() for label, _ in pairs})
    to_code = {"present": 1, "absent": 0, "uncertain": -1}
    rows = []
    for dict_tuple, report in zip(label_dict.items(), reports):
        _id = dict_tuple[0]
        pairs = dict_tuple[1]
        codes = {label: to_code.get(assertion, pd.NA) for label, assertion in pairs}
        labels = []
        for label, code in codes.items():
            if use_uncertain:
                if code != 0:
                    labels.append(label)
            else:
                if code == 1:
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
    # drop the empty frozenset
    #non_empty = [fs for fs in CUIS if fs]

    # 3) Build every non-empty combination (the power-set minus ∅)
    #all_combos = [
    #    frozenset().union(*subset)  # merge the chosen singletons
    #    for r in range(1, len(non_empty) + 1)  # subset sizes 1 … 9
    #    for subset in combinations(non_empty, r)  # all those subsets
    #]


    args = parse_args()
    df = pd.read_csv(args.src)
    #df = df[df["dicom_id"].isin(["f523bedf-f3f156fb-084a055e-0d047cfb-d199314a",
    #                             "46d98844-09036afb-42ec58ac-562cfbc5-903aba88",
    #                             "113dfb1b-a9f85d25-b5bd442f-3ac4eb91-0fddb8b4",
    #                             "5f336568-31432e87-3525f462-f206bf51-11c80f82",
    #                             "20fa17c8-b818a39a-ad6aaffe-ba124cd6-ef8a3985"])]
    with open(args.config_path) as stream:
        config = yaml.safe_load(stream)
    data = initialize_data(**config["init_data_args"], df=df)
    linker = ClinicalEntityLinker.from_default_constants()
    mentions = get_mentions(linker, args.mentions_path, data["reports"])
    #label_dict = create_labels(data["ids"], mentions, linker, data["reports"])
    label_dict = create_labels(data["ids"], mentions, linker, data["reports"])

    label_df = build_label_dataframe(label_dict, data["reports"], args.use_uncertain)

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(dst_path, index=False)

    print(f"Wrote labelled CSV to {dst_path}")




