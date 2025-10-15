import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# def train_model(
#     train: pd.DataFrame,
#     model: BaseEstimator=RandomForestClassifier,
#     model_params: dict = {"n_estimators": 100},
# ) -> RandomForestClassifier:
#     """
#     Function to train a model on the training data.

#     Parameters
#     ----------
#     model : BaseEstimator, optional
#         The model to train, by default RandomForestClassifier.
#     train : pd.DataFrame
#         The training data to train the model on.

#     Returns
#     -------
#     RandomForestClassifier
#         The trained model.
#     """
#     X = train.drop("target", axis=1)
#     y = train["target"]
#     clf = model(**model_params)
#     clf.fit(X, y)
#     return clf



def load_data() -> pd.DataFrame:
    """
    Loads the medical insurance dataset and adds a 'target' column (same as 'charges').

    Returns
    -------
    pd.DataFrame
        DataFrame with all features and 'target' column.
    """
    df = pd.read_csv('./src/medical_data/insurance.csv')
    df = df.drop_duplicates().dropna()

    df["target"] = df["charges"]
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train and test sets.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train, test


def train_model(
    train: pd.DataFrame,
    model: RegressorMixin = RandomForestRegressor,
    model_params: dict = {"n_estimators": 100, "random_state": 42},
) -> Pipeline:
    """
    Trains a regression model (default: RandomForestRegressor) with preprocessing pipeline.

    Parameters
    ----------
    train : pd.DataFrame
        Training DataFrame containing features and target.
    model : RegressorMixin, optional
        The regression model class to train, by default RandomForestRegressor.
    model_params : dict, optional
        Parameters for the model, by default {"n_estimators": 100, "random_state": 42}.

    Returns
    -------
    Pipeline
        A trained sklearn Pipeline with preprocessing + model.
    """
    X = train.drop(["target", "charges"], axis=1)
    y = train["target"]

    # Определяем типы признаков
    num_features = ["age", "bmi", "children"]
    cat_features = ["sex", "smoker", "region"]

    # Предобработка
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # Конструируем пайплайн: предобработка → модель
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model(**model_params))
    ])

    # Обучаем
    pipeline.fit(X, y)

    return pipeline



# def evaluate_model(
#     model: RandomForestClassifier, test: pd.DataFrame
# ) -> float:
#     """
#     Function to evaluate the model on the test data.

#     Parameters
#     ----------
#     model : RandomForestClassifier
#         The trained model to evaluate.
#     test : pd.DataFrame
#         The test data to evaluate the model on.

#     Returns
#     -------
#     float
#         The accuracy of the model on the test data.
#     """
#     X = test.drop("target", axis=1)
#     y = test["target"]
#     y_pred = model.predict(X)
#     return accuracy_score(y, y_pred)

def evaluate_model(model: Pipeline, test: pd.DataFrame) -> dict:
    """
    Evaluates a regression model on the test data using MAE, RMSE, and R².

    Parameters
    ----------
    model : Pipeline
        The trained regression model (with preprocessing).
    test : pd.DataFrame
        The test data.

    Returns
    -------
    dict
        Dictionary containing 'MAE', 'RMSE', and 'R2' metrics.
    """
    X = test.drop(["target", "charges"], axis=1)
    y = test["target"]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
