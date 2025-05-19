import os
from typing import Union, Dict, Optional

import pandas as pd

def initialize_data(
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




