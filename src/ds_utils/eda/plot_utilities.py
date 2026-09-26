from ds_utils.preprocessing.preprocessing_config import (
    FEATURES_CONFIG,
    FeatureNamingConfig,
)

import numpy as np
import pandas as pd
from typing import Literal, Protocol
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# region Aux Class -------------------------------------------------------------

class Transformer(Protocol):
    """Protocol for objects implementing a fit-transform operation."""

    def fit_transform(self, X):
        """Fit the transformer and transform the input data."""
        ...

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def bold(text: str) -> str:
    return rf"$\bf{{{text}}}$"

# endregion Aux Functions ------------------------------------------------------

# region Plots Functions -------------------------------------------------------

def plot_circular_feature(
    df: pd.DataFrame,
    col_sin: str,
    col_cos: str,
    title: str = "Representación Cíclica"
) -> None:
    """
    Plot a circular feature representation using sine and cosine components.

    This visualization helps verify that a cyclic variable has been correctly
    transformed into sine and cosine components.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sine and cosine feature columns.

    col_sin : str
        Name of the column containing the sine component.

    col_cos : str
        Name of the column containing the cosine component.

    title : str, default="Representación Cíclica"
        Figure title.
    """
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df[col_cos], df[col_sin], alpha=0.3, color='teal')
    
    # Añadimos líneas de referencia para marcar los cuadrantes
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    
    plt.title(title)
    plt.xlabel(col_cos)
    plt.ylabel(col_sin)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_sensor_signals(
    df: pd.DataFrame,
    col_time: str,
    tolerances: dict[str, float],
    config: FeatureNamingConfig = FEATURES_CONFIG,
) -> None:
    """
    Plot raw and cleaned sensor signals, highlighting detected noise samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sensor data.

    col_time : str
        Name of the column containing the time values.

    tolerances : dict[str, float]
        Mapping of sensor names to their corresponding noise thresholds.

    config : FeatureNamingConfig, default=FEATURES_CONFIG
        Naming configuration used to identify the cleaned-signal and
        noise-indicator columns.
    """

    if col_time not in df.columns:
        raise ValueError(f"Column '{col_time}' not found in DataFrame.")

    for sensor, threshold in tolerances.items():

        if sensor not in df.columns:
            print(f"Sensor '{sensor}' not found. Skipping.")
            continue

        plt.figure(figsize=(15, 5))

        # Raw signal
        plt.plot(
            df[col_time],
            df[sensor],
            color="darkgray",
            alpha=0.9,
            linewidth=1.2,
            label="Raw",
        )

        # Clean signal
        clean_col = f"{sensor}{config.CLEAN_SUFFIX}"
        if clean_col in df.columns:
            plt.plot(
                df[col_time],
                df[clean_col],
                color="green",
                linewidth=2,
                label="Limpio",
            )

        # Detected noise
        noise_col = f"{sensor}{config.IS_NOISE_SUFFIX}"
        if noise_col in df.columns:
            noise_df = df[df[noise_col]]
            plt.scatter(
                noise_df[col_time],
                noise_df[sensor],
                color="red",
                s=20,
                label="Ruido Detectado",
            )

        plt.title(
            rf"Análisis de Ruido en '{sensor}' "
            rf"$\bf{{(Umbral: >\ {threshold})}}$"
        )
        plt.xlabel("Tiempo")
        plt.ylabel(sensor)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()



# TODO deprecated
# def plot_sensor_signals_old(
#     df: pd.DataFrame,
#     col_time: str,
#     tolerances: dict[str, float],
# ) -> None:
#     """
#     Plot raw and cleaned sensor signals, highlighting detected noise samples.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing the sensor data.

#     col_time : str
#         Name of the column containing the time values.

#     tolerances : dict[str, float]
#         Mapping of sensor names to their corresponding noise thresholds.

#     Notes
#     -----
#     The names of the cleaned-signal and noise-indicator columns are derived from
#     the suffixes defined in ``FEATURES_CONFIG``.
#     """

#     if col_time not in df.columns:
#         raise ValueError(f"Column '{col_time}' not found in DataFrame.")

#     for sensor, threshold in tolerances.items():

#         if sensor not in df.columns:
#             print(f"Sensor '{sensor}' not found. Skipping.")
#             continue

#         plt.figure(figsize=(15, 5))

#         # Raw signal
#         plt.plot(
#             df[col_time],
#             df[sensor],
#             color="darkgray",
#             alpha=0.9,
#             linewidth=1.2,
#             label="Raw",
#         )

#         # Clean signal
#         clean_col = f"{sensor}{FEATURES_CONFIG.CLEAN_SUFFIX}"
#         if clean_col in df.columns:
#             plt.plot(
#                 df[col_time],
#                 df[clean_col],
#                 color="green",
#                 linewidth=2,
#                 label="Limpio",
#             )

#         # Detected noise
#         noise_col = f"{sensor}{FEATURES_CONFIG.IS_NOISE_SUFFIX}"
#         if noise_col in df.columns:
#             noise_df = df[df[noise_col]]
#             plt.scatter(
#                 noise_df[col_time],
#                 noise_df[sensor],
#                 color="red",
#                 s=20,
#                 label="Ruido Detectado",
#             )

#         plt.title(
#             rf"Análisis de Ruido en '{sensor}' $\bf{{(Umbral: >\ {threshold})}}$"
#         )
#         plt.xlabel("Tiempo")
#         plt.ylabel(sensor)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


def plot_model_anomaly_scores(
    df: pd.DataFrame,
    col_time: str,
    col_score: str,
    anomaly_threshold: float = 0.0,
    title: str | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Plot anomaly scores over time highlighting detected anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing timestamps and anomaly scores.

    col_time : str
        Timestamp column.

    col_score : str
        Column containing anomaly scores.

    anomaly_threshold : float, default=0.0
        Scores below or equal to this threshold are considered anomalies.

    title : str, optional
        Figure title.

    figsize : tuple[int, int], default=(15, 5)
        Figure size.
    """

    anomaly_mask = df[col_score] <= anomaly_threshold

    fig, ax = plt.subplots(figsize=figsize)

    # Complete score signal
    ax.plot(
        df[col_time],
        df[col_score],
        linewidth=0.8,
        label="Score",
    )

    # Detected anomalies
    ax.scatter(
        df.loc[anomaly_mask, col_time],
        df.loc[anomaly_mask, col_score],
        color="red",
        s=15,
        label="Anomalies",
    )

    # Decision threshold
    ax.axhline(
        anomaly_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({anomaly_threshold})",
    )

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title or col_score)

    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()


def plot_model_anomaly_score_distribution(
    df: pd.DataFrame,
    col_score: str,
    anomaly_threshold: float = 0.0,
    bins: int = 30,
    title: str | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Plot the anomaly score distribution separating normal and anomalous
    observations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing anomaly scores.

    col_score : str
        Column containing anomaly scores.

    anomaly_threshold : float, default=0.0
        Scores below or equal to this threshold are considered anomalies.

    bins : int, default=30
        Number of histogram bins.

    title : str, optional
        Figure title.

    figsize : tuple[int, int], default=(15, 5)
        Figure size.
    """

    normal_scores = df.loc[
        df[col_score] > anomaly_threshold,
        col_score,
    ]

    anomaly_scores = df.loc[
        df[col_score] <= anomaly_threshold,
        col_score,
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label="Normal",
    )

    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        label="Anomaly",
    )

    ax.axvline(
        anomaly_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({anomaly_threshold})",
    )

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(title or "Anomaly score distribution")

    ax.legend()

    fig.tight_layout()

    plt.show()


def plot_model_predictions(
    y_real: pd.Series,
    y_pred: ArrayLike,
    mae: float | None = None,
    title: str = "Rendimiento del modelo",
    steps: int | None = None,
    color_real: str = "black",
    color_pred: str = "green",
) -> None:
    """
    Plot real and predicted values over the observation sequence.

    Parameters
    ----------
    y_real : pd.Series
        Observed target values. A DatetimeIndex is used for the x-axis when
        available.

    y_pred : ArrayLike
        Predicted target values.

    mae : float, optional
        Mean absolute error displayed in the figure title. If ``None``, the MAE
        is computed from ``y_real`` and ``y_pred``.

    title : str, default="Rendimiento del modelo"
        Figure title.

    steps : int, optional
        Number of initial observations to plot. If ``None``, all observations
        are plotted.

    color_real : str, default="black"
        Color used for the observed values.

    color_pred : str, default="green"
        Color used for the predicted values.
    """

    # Slice if needed
    if steps is not None:
        y_real = y_real.iloc[:steps]
        y_pred = y_pred[:steps]

    # X axis
    x_axis = (
        y_real.index if isinstance(y_real.index, pd.DatetimeIndex)
        else range(len(y_real))
    )

    # MAE fallback
    if mae is None:
        # from sklearn.metrics import mean_absolute_error
        # mae = mean_absolute_error(y_real, y_pred)

        mae = np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred)))

    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, y_real, label="Realidad", color=color_real, alpha=0.7)
    plt.plot(x_axis, y_pred, label="Predicción", color=color_pred, linestyle="--", alpha=0.8)

    plt.title(f"{title} — MAE: {mae:.4f}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_model_feature_importance(
    modelo,
    X: pd.DataFrame | ArrayLike,
    top_n: int = 15,
    title: str | None = None,
) -> pd.Series:
    """
    Plot the native feature importance of a fitted model.

    The function extracts feature importance from the final estimator of a
    pipeline when applicable. It supports tree-based models through
    ``feature_importances_``, models exposing ``get_feature_importance()``,
    LightGBM models using gain-based importance, and linear models through
    ``coef_``.

    No additional importance method, such as permutation importance, is
    calculated.

    Parameters
    ----------
    modelo : object
        Fitted model or pipeline from which to extract feature importance.

    X : pd.DataFrame or array-like
        Feature matrix used to determine the feature names. If ``X`` is a
        DataFrame, its column names are used. Otherwise, generic names in the
        form ``"feature_0"``, ``"feature_1"``, etc. are generated.

    top_n : int, default=15
        Number of most important features to display.

    title : str, optional
        Figure title. If ``None``, the title ``"Top {top_n} Features"`` is
        used.

    Returns
    -------
    pd.Series
        Feature importances for the selected top features, sorted in ascending
        order for horizontal bar plotting.

    Raises
    ------
    NotImplementedError
        If the fitted model does not expose a supported native feature
        importance interface.

    ValueError
        If the number of extracted feature importances does not match the
        number of features in ``X``.

    Notes
    -----
    For linear models with multiple outputs or classes, the importance of
    each feature is calculated as the mean absolute coefficient across
    outputs or classes.

    For LightGBM models, gain-based feature importance is used instead of the
    default split-based importance.

    Examples
    --------
    Tree-based model:

    >>> plot_model_feature_importance(modelo_rf, X_train)

    Linear model:

    >>> plot_model_feature_importance(modelo_lr, X_train)

    Both cases use the native feature importance interface exposed by the
    fitted model.
    """

    # ---------------------------
    # 1. Feature names
    # ---------------------------
    if hasattr(X, "columns"):
        feature_names = X.columns
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # ---------------------------
    # 2. Pipeline support
    # ---------------------------
    if hasattr(modelo, "named_steps"):
        modelo_final = list(modelo.named_steps.values())[-1]
    else:
        modelo_final = modelo

    # ---------------------------
    # 3. Importancias
    # ---------------------------

    # Árboles
    if hasattr(modelo_final, "feature_importances_"):
        importancias = modelo_final.feature_importances_

    # CatBoost
    elif hasattr(modelo_final, "get_feature_importance"):
        importancias = modelo_final.get_feature_importance()

    elif modelo_final.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
        # importancias = modelo_final.feature_importances_ # por defecto es 'split' en lugar de gain
        importancias = modelo_final.booster_.feature_importance(importance_type="gain")

    # Lineales
    elif hasattr(modelo_final, "coef_"):
        coef = modelo_final.coef_

        if coef.ndim == 1:
            importancias = np.abs(coef)
        else:
            importancias = np.mean(np.abs(coef), axis=0)

    else:
        raise NotImplementedError(
            f"El modelo {type(modelo_final).__name__} no soporta "
            f"importancia de features nativa"
        )

    # ---------------------------
    # 4. Validación
    # ---------------------------
    importancias = np.array(importancias)

    if len(importancias) != len(feature_names):
        raise ValueError(
            f"Mismatch: {len(importancias)} importancias vs {len(feature_names)} features"
        )

    feat_importances = pd.Series(importancias, index=feature_names)

    top_feats = feat_importances.nlargest(top_n).sort_values()

    # ---------------------------
    # 5. Plot
    # ---------------------------
    plt.figure(figsize=(10, 6))
    top_feats.plot(kind="barh")

    plt.title(title if title else f"Top {top_n} Features")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return top_feats


# endregion Plots --------------------------------------------------------------


# region Plots Functions with "lazy import" ------------------------------------

def plot_feature_space_projection(
    features: pd.DataFrame,
    projection: Literal["pca", "tsne", "umap"] = "pca",
    scaler: Transformer | None = None,
    anomaly_scores: ArrayLike | None = None,
    anomaly_threshold: float | None = None,
    random_state: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Visualize the feature space using a two-dimensional projection.

    The feature matrix is projected onto two dimensions using PCA, t-SNE or
    UMAP. This visualization is intended for exploratory data analysis to
    inspect the overall geometry of the feature space, such as clusters,
    isolated observations or variations in local density.

    Optionally, anomaly scores can be displayed either as a continuous colour
    gradient or as a binary normal/anomaly classification using a decision
    threshold.

    The projection does not represent the decision boundary or behaviour of
    any anomaly detection model.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.

    projection : {"pca", "tsne", "umap"}, default="pca"
        Dimensionality reduction technique used for the projection.

    scaler : Transformer, optional
        Transformer implementing ``fit_transform()`` such as
        ``StandardScaler`` or ``RobustScaler``.

    anomaly_scores : ArrayLike, optional
        Anomaly scores associated with each observation. When provided
        without an anomaly threshold, the scores are displayed as a
        continuous colour gradient. When an anomaly threshold is also
        provided, observations are classified as normal or anomalous.

    anomaly_threshold : float, optional
        Threshold used to classify observations as anomalies.
        Observations with scores below the threshold are highlighted as
        anomalies.

    random_state : int, optional
        Random state used by the projection algorithm.

    title : str, optional
        Figure title.

    figsize : tuple[int, int], default=(10, 8)
        Figure size.

    Notes
    -----
    The ``"pca"`` and ``"tsne"`` projections require ``scikit-learn``.
    The ``"umap"`` projection requires ``umap-learn``.
    """

    if projection not in ("pca", "tsne", "umap"):
        raise ValueError(
            "projection must be one of {'pca', 'tsne', 'umap'}."
        )

    POINT_SIZE = 25
    ALPHA = 0.7

    X = features

    if scaler is not None:
        X = scaler.fit_transform(X)

    match projection:

        case "pca":

            try:
                from sklearn.decomposition import PCA
            except ImportError as exc:
                raise ImportError(
                    "PCA projection requires the 'scikit-learn' package."
                ) from exc

            reducer = PCA(
                n_components=2,
                random_state=random_state,
            )

        case "tsne":

            try:
                from sklearn.manifold import TSNE
            except ImportError as exc:
                raise ImportError(
                    "t-SNE projection requires the 'scikit-learn' package."
                ) from exc

            reducer = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            )

        case "umap":

            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP projection requires the 'umap-learn' package."
                ) from exc

            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
            )

    X_proj = reducer.fit_transform(X)

    # ------------------------------------
    # VISUALIZACIÓN
    # ------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    if anomaly_scores is None:

        ax.scatter(
            X_proj[:, 0],
            X_proj[:, 1],
            s=POINT_SIZE,
            alpha=ALPHA,
        )

    elif anomaly_threshold is None:

        scatter = ax.scatter(
            X_proj[:, 0],
            X_proj[:, 1],
            c=anomaly_scores,
            cmap="inferno", # "viridis", "inferno", "plasma"
            s=POINT_SIZE,
            alpha=ALPHA,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label="Anomaly score",
        )

    else:

        anomaly_mask = np.asarray(anomaly_scores) <= anomaly_threshold

        ax.scatter(
            X_proj[~anomaly_mask, 0],
            X_proj[~anomaly_mask, 1],
            s=POINT_SIZE,
            alpha=ALPHA,
            label="Normal",
        )

        ax.scatter(
            X_proj[anomaly_mask, 0],
            X_proj[anomaly_mask, 1],
            s=POINT_SIZE,
            alpha=ALPHA,
            color="red",
            label="Anomaly",
        )

        ax.legend()


    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Oculta los valores de los ejes que no representan una variable real
    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:

        title = f"{projection.upper()} projection"

    if projection == "pca":

        explained = reducer.explained_variance_ratio_.sum()

        title += f" (2D projection retains {explained:.1%} of the original information)"


    ax.set_title(title)

    ax.grid(alpha=0.3)

    fig.tight_layout()

    plt.show()

# endregion Plots Functions with "lazy import" ---------------------------------