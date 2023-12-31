import os
from typing import Any, Dict, Tuple

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from config import paths
from logger import get_logger
from schema.data_schema import ClassificationSchema
from sklearn.decomposition import PCA

logger = get_logger(task_name="preprocess")


def impute_numeric(
    input_data: pd.DataFrame, column: str, value="median"
) -> Tuple[pd.DataFrame, Any]:
    """
    Imputes the missing numeric values in the given dataframe column based on the parameter 'value'.

    Args:
        input_data (pd.DataFrame): The data to be imputed.
        column (str): The name of the column.
        value (str): The value to use when imputing the column. Can only be one of ['mean', 'median', 'mode']

    Returns:
        Tuple[pd.DataFrame, Any]:  Dataframe after imputation and the value used for imputation
    """

    if column not in input_data.columns:
        return input_data, None
    if value == "mean":
        value = input_data[column].mean()
    elif value == "median":
        value = input_data[column].median()
    elif value == "mode":
        value = input_data[column].mode().iloc[0]
    input_data[column].fillna(value=value, inplace=True)
    return input_data, value


def impute_categorical(
    input_data: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, Any]:
    """
    Imputes the missing categorical values in the given dataframe column.
    If the percentage of missing values in the column is greater than 0.1, imputation is done using the word "Missing".
    Otherwise, the mode is used.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        column (str): The name of the column to be imputed.

    Returns:
        Tuple[pd.DataFrame, Any]:  Dataframe after imputation and the value used for imputation
    """
    if column not in input_data.columns:
        return input_data, None
    perc = percentage_of_missing_values(input_data)
    if column in perc and perc[column] > 10:
        value = "Missing"
    else:
        value = input_data[column].mode().iloc[0]
    input_data[column].fillna(value=value, inplace=True)
    return input_data, value


def percentage_of_missing_values(input_data: pd.DataFrame) -> Dict:
    """
    Calculates the percentage of missing values in each column of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to calculate the percentage of missing values on.

    Returns:
        A dictionary of column names as keys and the percentage of missing values as values.
    """
    columns_with_missing_values = input_data.columns[input_data.isna().any()]
    return (
        input_data[columns_with_missing_values]
        .isna()
        .mean()
        .sort_values(ascending=False)
        * 100
    ).to_dict()


def encode(
    input_data: pd.DataFrame, schema: ClassificationSchema, encoder=None
) -> pd.DataFrame:
    """
    Performs one-hot encoding for the top 10 categories on categorical features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        schema (ClassificationSchema): The schema of the given data.
        encoder: Indicates if instantiating a new encoder is required or not.

    Returns:
        A dataframe after performing one-hot encoding
    """
    cat_features = schema.categorical_features
    if not cat_features:
        return input_data

    if encoder is not None and os.path.exists(paths.ENCODER_FILE):
        encoder = load(paths.ENCODER_FILE)
        input_data = encoder.transform(input_data)
        return input_data

    encoder = OneHotEncoder(top_categories=10)
    encoder.fit(input_data)
    input_data = encoder.transform(input_data)
    dump(encoder, paths.ENCODER_FILE)

    return input_data


def normalize(
    input_data: pd.DataFrame, schema: ClassificationSchema, scaler=None
) -> pd.DataFrame:
    """
    Performs MinMax normalization on numeric features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The data to be normalized.
        schema (ClassificationSchema): The schema of the given data.
        scaler: Indicated if a new scaler needs to be instantiated.

    Returns:
        A dataframe after z-score normalization
    """

    input_data = input_data.copy()
    numeric_features = schema.numeric_features
    if not numeric_features:
        return input_data
    numeric_features = [f for f in numeric_features if f in input_data.columns]
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(input_data[numeric_features])
        dump(scaler, paths.SCALER_FILE)
    else:
        scaler = load(paths.SCALER_FILE)
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    return input_data



def encode_target(target: pd.Series, save_path: str = paths.TARGET_ENCODER_FILE) -> Tuple[pd.Series, LabelEncoder]:

    """
    Performs label encoding for the target classes and saves the encoder to the provided path.

    Args:
        target (pd.Series): The target labels.
        save_path (str) Directory to save the encoder.

    Returns:
        Tuple[pd.Series, LabelEncoder]: The encoded targets and the used encoder.
    """

    encoder = LabelEncoder()
    results = encoder.fit_transform(target)
    dump(encoder, save_path)
    return results, encoder


def reduce_dimensions(input: pd.DataFrame, training:bool = True) -> pd.DataFrame:
    """
    Performs dimensionality reduction on the give input.
    Args:
        input (pd.DataFrame): The input data.
        training (bool): Determines the phase (training or testing).

    Returns:
        pd.DataFrame: The data after reducing the dimension.
    """
    if training:
        pca = PCA(n_components=0.95)
        result = pca.fit_transform(input)
        dump(pca, paths.PCA_FILE)
    else:
        pca = load(paths.PCA_FILE)
        result = pca.transform(input)
    return pd.DataFrame(result)
