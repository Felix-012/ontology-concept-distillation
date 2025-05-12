import os
from typing import Union, Dict

import pandas as pd

def initialize_data(csv_path: Union[str, os.PathLike],
                    image_base_path: Union[str, os.PathLike],
                    split_value: str,
                    split_column: Union[str, int],
                    report_column:  Union[str, int],
                    id_column: Union[str, int],
                    image_path_column: Union[str, int],
                    suffix: str =".jpg"
                    ) -> Dict[str, list]:

    dataset = {}
    path_f = lambda x: image_base_path + x.rsplit('.', 1)[0] + suffix
    df: pd.DataFrame = pd.read_csv(csv_path)
    df = df.loc[df[split_column].astype(str) == split_value]
    df.dropna(subset=[id_column, report_column, image_path_column, split_column], inplace=True)
    dataset["ids"] = df.loc[:, id_column].values.tolist() \
        if isinstance(id_column, str) else df.iloc[:, id_column].values.tolist()
    dataset["reports"] = df.loc[:, report_column].values.tolist() \
        if isinstance(report_column, str) else df.iloc[:, report_column].values.tolist()
    dataset["image_paths"] =df.loc[:, image_path_column].apply(path_f).values.tolist() \
        if isinstance(image_path_column, str) \
        else df.iloc[:, image_path_column].apply(path_f).values.tolist()

    return dataset




