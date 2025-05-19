import os
from typing import Union, Dict, Optional

import pandas as pd

# Global array defining all classes in order
CLASSES = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion',
    'Pneumonia',
    'Pneumothorax',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
]


def initialize_data(csv_path: Union[str, os.PathLike],
                    image_base_path: Union[str, os.PathLike],
                    report_column:  Union[str, int],
                    id_column: Union[str, int],
                    image_path_column: Union[str, int],
                    suffix: str =".jpg",
                    csv_path: Union[str, os.PathLike] = None,
                    df: Optional[pd.DataFrame] = None,
                    split_value: Optional[str] = None,
                    split_column: Union[str, int] = None,
                    ) -> Dict[str, list]:

    if df is None and csv_path is None:
        raise ValueError("Either dataframe or csv_path must be specified")

    dataset = {}
    path_f = lambda x: image_base_path + x.rsplit('.', 1)[0] + suffix
    if df is None:
        df: pd.DataFrame = pd.read_csv(csv_path)
    if split_value is not None:
        df = df.loc[df[split_column].astype(str) == split_value]
    if split_column is None:
        df = df.dropna(subset=[id_column, report_column, image_path_column])
    else:
        df = df.dropna(subset=[id_column, report_column, image_path_column, split_column])
    dataset["ids"] = df.loc[:, id_column].values.tolist() \
        if isinstance(id_column, str) else df.iloc[:, id_column].values.tolist()
    dataset["reports"] = df.loc[:, report_column].values.tolist() \
        if isinstance(report_column, str) else df.iloc[:, report_column].values.tolist()
    dataset["image_paths"] =df.loc[:, image_path_column].apply(path_f).values.tolist() \
        if isinstance(image_path_column, str) \
        else df.iloc[:, image_path_column].apply(path_f).values.tolist()

    return dataset


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




def map_to_label_vector(findings: str, num_classes: int = None) -> list:
    """
    Maps a string of findings to a binary label vector.
    Uses only the first num_classes classes from CLASSES list.

    Args:
        findings (str): Comma-separated string of findings
        num_classes (int, optional): Number of classes to use. If None, uses all classes.

    Returns:
        list: Binary vector where 1 indicates presence of finding
    """
    # Use all classes if num_classes not specified
    if num_classes is None:
        num_classes = len(CLASSES)

    # Initialize zero vector
    label_vector = [0] * num_classes

    # Split findings string and clean each finding
    if pd.isna(findings):
        # Handle NaN/empty case
        label_vector[0] = 1  # Set No Finding to 1
        return label_vector

    finding_list = [f.strip() for f in findings.split('|')]

    # Map findings to vector
    for finding in finding_list:
        if finding in CLASSES[:num_classes]:
            label_vector[CLASSES.index(finding)] = 1

    return label_vector

def label_vector_to_findings(label_vector: list) -> str:
    """
    Maps a binary label vector back to a comma-separated string of findings.
    First 8 positions are the main classes, remaining positions are additional findings.

    Args:
        label_vector (list): Binary vector where 1 indicates presence of finding

    Returns:
        str: Comma-separated string of findings
    """
    # Get findings where label is 1
    findings = [CLASSES[i] for i, label in enumerate(label_vector) if label == 1]

    # Handle empty case
    if not findings:
        return 'No Finding'

    # Join findings with commas
    return ', '.join(findings)
