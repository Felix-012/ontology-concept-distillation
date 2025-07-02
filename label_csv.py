#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.ner import ClinicalEntityLinker, get_mentions, create_labels

_RENAME_MAP: Dict[str, str] = {
    "Pleural": "Pleural Other",
    "Device": "Support Devices",
    "Lesion": "Lung Lesion",
    "Opacity": "Lung Opacity",
    "Support Device": "Support Devices",
}


def replace_effusion_token(label_string: str | float) -> str | float:
    if pd.isna(label_string):
        return label_string

    tokens = [tok.strip() for tok in str(label_string).split("|")]
    tokens = ["Pleural Effusion" if tok == "Effusion" else tok for tok in tokens]
    if "Pleural Effusion" in tokens:
        tokens = [tok for tok in tokens if tok != "Pleural"]
    tokens = [_RENAME_MAP.get(tok, tok) for tok in tokens]

    return "|".join(tokens)


def build_label_dataframe(
    label_dict: Dict[str, List[Tuple[str, str]]],
    reports: pd.Series,
    use_uncertain: bool,
) -> pd.DataFrame:

    all_labels = sorted({label for pairs in label_dict.values() for label, _ in pairs})
    to_code = {"present": 1, "absent": 0, "uncertain": -1}
    rows = []

    for (_id, pairs), report in zip(label_dict.items(), reports):
        codes = {label: to_code.get(assertion, pd.NA) for label, assertion in pairs}

        # Build the pipe‑separated label string
        labels: List[str] = []
        for label, code in codes.items():
            if use_uncertain:
                if code != 0:
                    labels.append(label)
            else:
                if code == 1:
                    labels.append(label)
        if not labels:
            labels = ["No Finding"]

        row = {
            "id": _id,
            "labels": "|".join(sorted(labels)),
            "reports": report,
        }
        row.update({label: codes.get(label, pd.NA) for label in all_labels})
        rows.append(row)

    ordered_cols = ["id", "labels", "reports"] + all_labels
    return pd.DataFrame(rows, columns=ordered_cols)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label chest‑x‑ray studies **and** post‑process in a single run.")

    parser.add_argument("--src", required=True, type=Path, help="Input metadata CSV (one row per study).")
    parser.add_argument("--dst", required=True, type=Path, help="Destination path for the processed CSV.")
    parser.add_argument("--config_path", required=True, type=Path, help="YAML config consumed by initialize_data.")
    parser.add_argument("--mentions_path", required=True, type=Path, help="Cache path for mention extraction.")
    parser.add_argument("--use_uncertain", action="store_true", help="Treat 'uncertain' as positive when set.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df_src = pd.read_csv(args.src)
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    data = initialize_data(**config["init_data_args"], df=df_src)

    linker = ClinicalEntityLinker.from_config_path(args.config_path)
    mentions = get_mentions(linker, args.mentions_path, data["reports"])
    label_dict = create_labels(data["ids"], mentions, linker, data["reports"])
    label_df = build_label_dataframe(label_dict, data["reports"], args.use_uncertain)

    label_df = label_df.rename(
        columns={"id": "dicom_id", "labels": "Finding Labels", "reports": "impression"}
    )

    label_df["Finding Labels"] = label_df["Finding Labels"].apply(replace_effusion_token)

    label_df = label_df.merge(df_src[["dicom_id", "path"]], on="dicom_id", how="left")

    dst_path = args.dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(dst_path, index=False)

    print(f"Wrote labelled CSV with {len(label_df):,} rows → {dst_path.resolve()}")
