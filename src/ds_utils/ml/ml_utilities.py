import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import seaborn as sns
import re


# region AUX Class -------------------------------------------------------------

# endregion AUX Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def bold(text: str) -> str:
    return rf"$\bf{{{text}}}$"

def normalize_model_name(model_name: str) -> str:
    """
    Convierte un nombre de modelo a un identificador en snake_case.

    Ejemplos:
        "Isolation Forest" -> "isolation_forest"
        "Local Outlier Factor" -> "local_outlier_factor"
        "One-Class SVM" -> "one_class_svm"
        "Elliptic Envelope" -> "elliptic_envelope"
    """
    model_name = model_name.lower()
    model_name = re.sub(r"[^a-z0-9]+", "_", model_name)
    model_name = re.sub(r"_+", "_", model_name)
    return model_name.strip("_")

# region Aux Functions ---------------------------------------------------------


# region Machine Learning Functions --------------------------------------------

def format_max_features(max_features: str | float | int | None, n_features: int) -> str:
    """
    Formats the `max_features` parameter from tree-based models into a readable string.

    Converts typical sklearn values ('sqrt', 'log2', float, None)
    into a more interpretable representation, including the percentage
    of features used relative to the total.

    Args:
        max_features (str | float | int | None):
            Value of the max_features parameter. Can be:
            - 'sqrt'  -> square root of total features
            - 'log2'  -> base-2 logarithm of total features
            - float   -> fraction (0.0 - 1.0)
            - None    -> use all features

        n_features (int):
            Total number of features in the dataset.

    Returns:
        str:
            Formatted representation, for example:
            - "sqrt (0.32)"
            - "log2 (0.21)"
            - "fraction (0.50)"
            - "all (1.00)"
    """

    if max_features == "sqrt":
        val = np.sqrt(n_features)
        perc = val / n_features
        return f"sqrt ({perc:.2f})"

    if max_features == "log2":
        val = np.log2(n_features)
        perc = val / n_features
        return f"log2 ({perc:.2f})"

    if max_features is None:
        return "all (1.00)"

    return f"fraction ({max_features})"


def extract_used_params(all_params: dict, used_params: dict):
    """
    Extracts only the parameters defined by the user from the full parameter set.

    Args:
        all_params (dict): Dictionary with all model parameters (e.g. model.get_params()).
        used_params (dict): Dictionary with only the parameters defined by the user.

    Returns:
        dict: Dictionary containing only the used parameters with their actual values.
    """

    return {k: all_params[k] for k in used_params.keys() if k in all_params}








def explain_anomalies(
    df: pd.DataFrame,
    col_time: str,
    anomaly_scores: ArrayLike,
    anomaly_mask: ArrayLike,
    top_features: int = 5,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Generate human-readable explanations for detected anomalies.

    For each detected anomaly, the function compares every numeric feature
    against its historical behaviour and returns the variables with the
    largest deviation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables used to explain anomalies.

    col_time : str
        Name of the timestamp column.

    anomaly_scores : ArrayLike
        Anomaly scores returned by the anomaly detector.

    anomaly_mask : ArrayLike
        Boolean mask identifying the detected anomalies.
        True indicates an anomaly and False a normal observation.

    top_features : int, default=5
        Number of variables to include in the explanation.

    ascending : bool, default=True
        Sort order applied to the anomaly scores in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per detected anomaly containing:

        - timestamp
        - score
        - summary
        - details
    """

    # Normalizacion para convertir en array de numpy por si el formato es distinto
    anomaly_scores = np.asarray(anomaly_scores)
    anomaly_mask = np.asarray(anomaly_mask)

    numeric_columns = df.select_dtypes(include="number").columns

    # Historical statistics used as reference.
    medians = df[numeric_columns].median()
    p99 = df[numeric_columns].quantile(0.99)

    explanations = []

    # anomaly_indices = np.where(anomaly_mask == -1)[0]
    anomaly_indices = np.where(anomaly_mask)[0]

    for idx in anomaly_indices:

        row = df.iloc[idx]

        feature_details = []

        for col in numeric_columns:

            current = float(row[col])
            median = float(medians[col])
            percentile99 = float(p99[col])

            # Heuristic used only to rank variables.
            deviation = abs(current - median)

            feature_details.append({
                "feature": col,
                "current": current,
                "median": median,
                "p99": percentile99,
                "deviation": deviation,
            })

        feature_details = sorted(
            feature_details,
            key=lambda x: x["deviation"],
            reverse=True,
        )[:top_features]

        summary = "; ".join(
            (
                f"{item['feature']} "
                f"(actual={item['current']:.3f}, "
                f"mediana={item['median']:.3f}, "
                f"P99={item['p99']:.3f})"
            )
            for item in feature_details
        )

        explanations.append({
            "timestamp": row[col_time],
            "score": float(anomaly_scores[idx]),
            "summary": summary,
            "details": feature_details,
        })

    return (
        pd.DataFrame(explanations)
        .sort_values(by="score", ascending=ascending)
        .reset_index(drop=True)
    )


# def explain_anomalies_OLD(
#     X: pd.DataFrame,
#     anomaly_scores: np.ndarray,
#     anomaly_predictions: np.ndarray,
#     timestamp: pd.Series | None = None,
#     top_features: int = 5,
# ) -> pd.DataFrame:
#     """
#     Generate human-readable explanations for detected anomalies.

#     For each anomalous sample, the function compares every numeric feature
#     against its historical median and identifies the variables with the
#     largest relative deviation.

#     Parameters
#     ----------
#     X : pd.DataFrame
#         Dataset used for anomaly detection.

#     anomaly_scores : np.ndarray
#         Scores returned by the anomaly detector.

#     anomaly_predictions : np.ndarray
#         Predictions returned by the anomaly detector.
#         Expected values are 1 (normal) and -1 (anomaly).

#     timestamp : pd.Series, optional
#         Timestamp associated with each sample.

#     top_features : int, default=5
#         Number of most abnormal variables to include.

#     Returns
#     -------
#     pd.DataFrame
#         One row per detected anomaly containing:

#         - timestamp
#         - score
#         - summary (formatted string)
#         - details (list of dictionaries)
#     """

#     numeric_columns = X.select_dtypes(include="number").columns

#     # Use the median as the reference because it is more robust to outliers.
#     medians = X[numeric_columns].median()

#     EPSILON = 1e-12

#     explanations = []

#     anomaly_indices = np.where(anomaly_predictions == -1)[0]

#     for idx in anomaly_indices:

#         row = X.iloc[idx]

#         feature_details = []

#         for col in numeric_columns:

#             current = float(row[col])
#             reference = float(medians[col])

#             # Avoid division by zero.
#             ratio = current / max(abs(reference), EPSILON)

#             # Symmetric deviation score.
#             deviation = abs(np.log(max(abs(ratio), EPSILON)))

#             feature_details.append({
#                 "feature": col,
#                 "current": current,
#                 "reference": reference,
#                 "ratio": ratio,
#                 "deviation": deviation,
#             })

#         feature_details = sorted(
#             feature_details,
#             key=lambda x: x["deviation"],
#             reverse=True,
#         )[:top_features]

#         summary = ", ".join(
#             f"{item['feature']} ({'↑' if item['ratio'] >= 1 else '↓'}×{item['ratio']:.1f})"
#             for item in feature_details
#         )

#         explanations.append({
#             "timestamp": None if timestamp is None else timestamp.iloc[idx],
#             "score": float(anomaly_scores[idx]),
#             "summary": summary,
#             "details": feature_details,
#         })

#     return (
#         pd.DataFrame(explanations)
#         .sort_values("score")
#         .reset_index(drop=True)
#     )



# def normalize_anomaly_scores_min_max(
#     scores: pd.Series | pd.DataFrame,
#     lower_percentile: float = 0.01,
#     upper_percentile: float = 0.99,
# ) -> pd.Series | pd.DataFrame:
#     """
#     Normalize anomaly scores using robust inverted Min-Max scaling.

#     Each score column is normalized independently to the range [0, 1], where:

#     - 0 represents the least anomalous observations.
#     - 1 represents the most anomalous observations.

#     Instead of using the absolute minimum and maximum values, the normalization
#     is performed between configurable percentiles to reduce the influence of
#     extreme outliers. Values outside this interval are clipped to [0, 1].

#     Parameters
#     ----------
#     scores : pd.DataFrame
#         DataFrame whose columns contain anomaly scores from one or more models.

#     lower_percentile : float, default=0.01
#         Lower percentile used as the normalization minimum.

#     upper_percentile : float, default=0.99
#         Upper percentile used as the normalization maximum.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with the normalized scores.
#     """

#     def _normalize(col: pd.Series) -> pd.Series:

#         lower = col.quantile(lower_percentile)
#         upper = col.quantile(upper_percentile)

#         if upper == lower:
#             return pd.Series(0.0, index=col.index)

#         normalized = 1 - (col - lower) / (upper - lower)

#         return normalized.clip(0, 1)

#     return scores.apply(_normalize)


def normalize_anomaly_scores_min_max(
    scores: pd.Series | pd.DataFrame,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.Series | pd.DataFrame:
    """
    Normalize anomaly scores using robust inverted Min-Max scaling.

    Each score column is normalized independently to the range [0, 1], where:

    - 0 represents the least anomalous observations.
    - 1 represents the most anomalous observations.

    Instead of using the absolute minimum and maximum values, the normalization
    is performed between configurable percentiles to reduce the influence of
    extreme outliers. Values outside this interval are clipped to [0, 1].

    This function assumes that lower scores correspond to more anomalous
    observations.

    Parameters
    ----------
    scores : pd.Series or pd.DataFrame
        Anomaly scores to normalize. When a DataFrame is provided, each column
        is normalized independently.

    lower_percentile : float, default=0.01
        Lower percentile used as the normalization minimum.

    upper_percentile : float, default=0.99
        Upper percentile used as the normalization maximum.

    Returns
    -------
    pd.Series or pd.DataFrame
        Normalized anomaly scores with the same type and shape as the input.

    Raises
    ------
    ValueError
        If `lower_percentile` and `upper_percentile` do not satisfy
        `0 <= lower_percentile < upper_percentile <= 1`.
    """

    if not 0 <= lower_percentile < upper_percentile <= 1:
        raise ValueError(
            "Expected 0 <= lower_percentile < upper_percentile <= 1."
        )

    is_series = isinstance(scores, pd.Series)

    if is_series:
        scores = scores.to_frame()

    lower = scores.quantile(lower_percentile)
    upper = scores.quantile(upper_percentile)

    normalized = 1 - (scores - lower) / (upper - lower)
    normalized = normalized.clip(0, 1)

    equal_limits = upper == lower
    if equal_limits.any():
        normalized.loc[:, equal_limits] = 0.0

    if is_series:
        return normalized.iloc[:, 0]

    return normalized



# def normalize_anomaly_scores_rank(
#     scores: pd.Series | pd.DataFrame
# ) -> pd.Series | pd.DataFrame:
#     """
#     Normalize anomaly scores using percentile ranking.

#     Each model is transformed independently so that:
#         1 -> most anomalous observations
#         0 -> least anomalous observations

#     This normalization makes anomaly scores comparable across models while
#     preserving their relative ordering.
#     """

#     return scores.apply(
#         lambda col: 1 - col.rank(pct=True)
#     )

def normalize_anomaly_scores_rank(
    scores: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Normalize anomaly scores using percentile ranking.

    Scores are transformed independently so that:

    - 0 represents the least anomalous observations.
    - 1 represents the most anomalous observations.

    The transformation preserves the relative ordering of the scores while
    making the outputs of different anomaly detection models directly
    comparable. This function assumes that lower scores correspond to more
    anomalous observations.

    Parameters
    ----------
    scores : pd.Series or pd.DataFrame
        Anomaly scores to normalize. When a DataFrame is provided, each column
        is normalized independently.

    Returns
    -------
    pd.Series or pd.DataFrame
        Rank-normalized anomaly scores with the same type and shape as the
        input.
    """

    is_series = isinstance(scores, pd.Series)

    if is_series:
        scores = scores.to_frame()

    normalized = 1 - scores.rank(pct=True)

    if is_series:
        return normalized.iloc[:, 0]

    return normalized


# endregion Machine Learning Functions -----------------------------------------


# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------