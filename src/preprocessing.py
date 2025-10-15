from typing import Optional, Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Function to load the medical insurance dataset from a CSV file
    and return it as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The medical insurance dataset as a pandas DataFrame,
        including an additional column 'target' (same as 'charges').

    Notes
    -----
    The dataset is expected to be located at './src/medical_data/insurance.csv'.
    It contains columns such as:
        - age
        - sex
        - bmi
        - children
        - smoker
        - region
        - charges
    """
    # Загружаем CSV
    df = pd.read_csv('./src/medical_data/insurance.csv')

    # Проверим, что нужные колонки на месте
    expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset is missing expected columns: {expected_cols - set(df.columns)}")

    # Добавляем целевую переменную
    df["target"] = df["charges"]

    return df



def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        The random state to use, by default 42.
    stratify : Optional, optional
        The variable to stratify the data on, by default None.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and test DataFrames.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train, test
