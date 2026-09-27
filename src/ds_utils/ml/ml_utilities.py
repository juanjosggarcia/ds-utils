import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import re


# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def normalize_model_name(model_name: str) -> str:
    """
    Convert a model name to a ``snake_case`` identifier.

    Parameters
    ----------
    model_name : str
        Model name to normalize.

    Returns
    -------
    str
        Normalized model name in ``snake_case``.

    Examples
    --------
    >>> normalize_model_name("Isolation Forest")
    'isolation_forest'
    >>> normalize_model_name("Local Outlier Factor")
    'local_outlier_factor'
    >>> normalize_model_name("One-Class SVM")
    'one_class_svm'
    >>> normalize_model_name("Elliptic Envelope")
    'elliptic_envelope'
    """

    model_name = model_name.lower()
    model_name = re.sub(r"[^a-z0-9]+", "_", model_name)
    model_name = re.sub(r"_+", "_", model_name)

    return model_name.strip("_")

# region Aux Functions ---------------------------------------------------------


# region Machine Learning Functions --------------------------------------------

def format_max_features(
    max_features: str | float | int | None, 
    n_features: int
) -> str:
    """
    Format the ``max_features`` parameter from tree-based models.

    Converts typical scikit-learn ``max_features`` values into a readable
    string representation, including the fraction of features used relative
    to the total number of features.

    Parameters
    ----------
    max_features : str, float, int, or None
        Value of the ``max_features`` parameter. Supported values are:

        - ``"sqrt"``: square root of the total number of features.
        - ``"log2"``: base-2 logarithm of the total number of features.
        - ``float``: fraction of features to use.
        - ``int``: number of features to use.
        - ``None``: use all features.

    n_features : int
        Total number of features in the dataset.

    Returns
    -------
    str
        Formatted representation of the ``max_features`` value. Examples
        include ``"sqrt (0.32)"``, ``"log2 (0.21)"``,
        ``"fraction (0.50)"``, and ``"all (1.00)"``.
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
    Extract only the parameters defined by the user from the full parameter set.

    Parameters
    ----------
    all_params : dict
        Dictionary containing all model parameters, such as those returned by
        ``model.get_params()``.

    used_params : dict
        Dictionary containing only the parameters explicitly defined by the
        user.

    Returns
    -------
    dict
        Dictionary containing only the parameters defined by the user that
        are present in ``all_params``.
    """

    extracted_params = {
        parameter: all_params[parameter]
        for parameter in used_params
        if parameter in all_params
    }

    return extracted_params


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
        Anomaly scores returned by the anomaly detector. Must contain one
        score for each row in ``df``.

    anomaly_mask : ArrayLike
        Boolean mask identifying the detected anomalies. ``True`` indicates
        an anomaly and ``False`` indicates a normal observation. Must contain
        one value for each row in ``df``.

    top_features : int, default=5
        Maximum number of variables to include in the explanation for each
        anomaly.

    ascending : bool, default=True
        Whether to sort the returned DataFrame by anomaly score in ascending
        order.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per detected anomaly with the following
        columns:

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


def normalize_anomaly_scores_min_max(
    scores: pd.Series | pd.DataFrame,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.Series | pd.DataFrame:
    """
    Normalize anomaly scores using robust inverted Min-Max scaling.

    Each score column is normalized independently to the range ``[0, 1]``,
    where ``0`` represents the least anomalous observations and ``1``
    represents the most anomalous observations.

    Instead of using the absolute minimum and maximum values, the
    normalization is performed between configurable percentiles to reduce
    the influence of extreme outliers. Values outside this interval are
    clipped to ``[0, 1]``.

    This function assumes that lower scores correspond to more anomalous
    observations.

    Parameters
    ----------
    scores : pd.Series or pd.DataFrame
        Anomaly scores to normalize. If a DataFrame is provided, each
        column is normalized independently.

    lower_percentile : float, default=0.01
        Lower percentile used as the normalization minimum. Must satisfy
        ``0 <= lower_percentile < upper_percentile <= 1``.

    upper_percentile : float, default=0.99
        Upper percentile used as the normalization maximum. Must satisfy
        ``lower_percentile < upper_percentile <= 1``.

    Returns
    -------
    pd.Series or pd.DataFrame
        Normalized anomaly scores with the same type and shape as the input.

    Raises
    ------
    ValueError
        If ``lower_percentile`` and ``upper_percentile`` do not satisfy
        ``0 <= lower_percentile < upper_percentile <= 1``.
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


def normalize_anomaly_scores_rank(
    scores: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Normalize anomaly scores using percentile ranking.

    Scores are transformed independently so that ``0`` represents the least
    anomalous observations and ``1`` represents the most anomalous
    observations.

    The transformation preserves the relative ordering of the scores while
    making the outputs of different anomaly detection models directly
    comparable. This function assumes that lower scores correspond to more
    anomalous observations.

    Parameters
    ----------
    scores : pd.Series or pd.DataFrame
        Anomaly scores to normalize. If a DataFrame is provided, each column
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