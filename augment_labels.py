#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import yaml

from data_utils import initialize_data
from knowledge_graph.ner import (
    ClinicalEntityLinker,
    get_mentions,
    choose_label,
)


def parse_args() -> argparse.Namespace:
    """Define and parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Augment radiology label set with knowledge-graph linking."
    )
    parser.add_argument(
        "--csv1",
        required=True,
        help="Path to the primary CSV (e.g., balanced_train.csv).",
    )
    parser.add_argument(
        "--csv2",
        required=True,
        help="Path to the secondary CSV providing additional labels.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file consumed by initialize_data().",
    )
    parser.add_argument(
        "--mentions",
        required=True,
        help="Pickle file containing cached UMLS mentions.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination for the augmented CSV. If omitted, a file called "
            "'labels_augmented.csv' will be created next to --csv1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # I/O
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    with open(args.config) as fp:
        config = yaml.safe_load(fp)

    # Prep data structures
    data1 = initialize_data(**config["init_data_args"], df=df1)
    initialize_data(**config["init_data_args"], df=df2)  # not used downstream

    linker = ClinicalEntityLinker.from_config_path(args.config)

    mentions = get_mentions(linker, args.mentions, data1["reports"])

    # Combine / choose labels
    labels1 = df1["Finding Labels"]
    labels2 = df2["Finding Labels"]
    augmented_labels = choose_label(mentions, labels1, labels2, linker)

    df1 = df1.copy()
    df1["Finding Labels"] = augmented_labels

    # Resolve output path
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.csv1).with_name("labels_augmented.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df1.to_csv(output_path, index=False)
    print(f"✔︎ Saved augmented CSV to {output_path}")


if __name__ == "__main__":
    main()
