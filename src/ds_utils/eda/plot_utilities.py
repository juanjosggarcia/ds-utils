from typing import Literal
from numpy.typing import ArrayLike
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
from enum import Enum
from IPython.display import display, HTML
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import TransformerMixin


# region AUX Class -------------------------------------------------------------

# class Alignment(Enum):
#     LEFT = "left"
#     CENTER = "center"
#     RIGHT = "right"

# class Color(Enum):
#     BLACK = "#000000"
#     WHITE = "#ffffff"
#     ORANGE = "#e64a19"
#     BLUE = "#1976d2"
#     PURPLE = "#7b1fa2"
#     YELLOW = "#fbc02d"
#     GREEN = "#2e7d32"
#     RED = "#d32f2f"
#     GRAY = "#4e4e4e"

# endregion --------------------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# DEFAULT_COLOR = Color.WHITE
# """
# the default color, which is important for ensuring consistency with the editor's light and dark themes
# """

# endregion --------------------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def bold(text: str) -> str:
    return rf"$\bf{{{text}}}$"

# endregion --------------------------------------------------------------------

# region Plots Functions -------------------------------------------------------

def plot_circular_feature(df, col_sin, col_cos, title="Representación Cíclica"):
    """
    Generate a scatter plot to verify the correct transformation 
    of variables in the circular coordinate system (Sine/Cosine).
    """
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df[col_cos], df[col_sin], alpha=0.3, color='teal')
    
    # Añadimos líneas de referencia para marcar los cuadrantes
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    
    plt.title(title)
    plt.xlabel(f"cos({col_cos})")
    plt.ylabel(f"sin({col_sin})")
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_sensor_signals(
    df,
    col_time: str,
    tolerances: dict[str, float],
    clean_suffix: str = "_clean",
    noise_suffix: str = "_is_noise",
) -> None:
    """
    Plot the raw and cleaned signals for each sensor, highlighting detected
    noise samples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the sensor data.
    col_time : str
        Name of the column containing the time values.
    tolerances : dict[str, float]
        Mapping of sensor names to their corresponding noise thresholds.
    clean_suffix : str, default="_clean"
        Suffix used to identify each sensor's cleaned signal.
    noise_suffix : str, default="_is_noise"
        Suffix used to identify each sensor's noise flag column.
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
        clean_col = f"{sensor}{clean_suffix}"
        if clean_col in df.columns:
            plt.plot(
                df[col_time],
                df[clean_col],
                color="green",
                linewidth=2,
                label="Limpio",
            )

        # Detected noise
        noise_col = f"{sensor}{noise_suffix}"
        if noise_col in df.columns:
            noise_df = df[df[noise_col]]
            plt.scatter(
                noise_df[col_time],
                noise_df[sensor],
                color="red",
                s=20,
                label="Ruido Detectado",
            )

        # plt.title(f"Análisis de Ruido en '{sensor}' (Umbral: > {threshold})")
        plt.title(
            rf"Análisis de Ruido en '{sensor}' $\bf{{(Umbral: >\ {threshold})}}$"
        )
        plt.xlabel("Tiempo")
        # plt.xlabel(col_time)
        plt.ylabel(sensor)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# def plot_sensor_signals(
#     df: pd.DataFrame,
#     time_column: str,
#     sensors: list[str],
#     clean_suffix: str = "_clean",
#     noise_suffix: str = "_is_noise",
# ) -> None:
#     """
#     Plot the raw and cleaned signals for each sensor, highlighting detected
#     noise samples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the sensor data.
#     time_column : str
#         Name of the column containing the time values.
#     sensors : list[str]
#         List of sensor names to visualize.
#     clean_suffix : str, default="_clean"
#         Suffix used to identify each sensor's cleaned signal.
#     noise_suffix : str, default="_is_noise"
#         Suffix used to identify each sensor's noise flag column.
#     """

#     if time_column not in df.columns:
#         raise ValueError(f"Column '{time_column}' not found in DataFrame.")

#     for sensor in sensors:

#         if sensor not in df.columns:
#             print(f"Sensor '{sensor}' not found. Skipping.")
#             continue

#         plt.figure(figsize=(15, 5))

#         # Raw signal
#         plt.plot(
#             df[time_column],
#             df[sensor],
#             color="darkgray",
#             alpha=0.9,
#             linewidth=1.2,
#             label="Raw",
#         )

#         # Clean signal
#         clean_col = f"{sensor}{clean_suffix}"
#         if clean_col in df.columns:
#             plt.plot(
#                 df[time_column],
#                 df[clean_col],
#                 color="green",
#                 linewidth=2,
#                 label="Limpio",
#             )

#         # Detected noise
#         noise_col = f"{sensor}{noise_suffix}"
#         if noise_col in df.columns:
#             noise_df = df[df[noise_col]]
#             plt.scatter(
#                 noise_df[time_column],
#                 noise_df[sensor],
#                 color="red",
#                 s=20,
#                 label="Ruido Detectado",
#             )

#         plt.title(f"Análisis de Ruido en el sensor '{sensor}'")
#         # plt.xlabel(time_column)
#         plt.xlabel("Tiempo")
#         plt.ylabel(sensor)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()





# def plot_feature_space_projection(
#     features: pd.DataFrame,
#     projection: Literal["pca", "tsne", "umap"] = "pca",
#     standardize: bool = False,
#     random_state: int | None = None,
#     title: str | None = None,
#     figsize: tuple[int, int] = (8, 8),
# ) -> None:
#     """
#     Visualize the feature space using a two-dimensional projection.

#     The feature matrix is projected onto two dimensions using either PCA or
#     UMAP. This visualization is intended for exploratory data analysis to
#     inspect the overall geometry of the feature space, such as clusters,
#     isolated observations or variations in local density.

#     The projection does not represent the decision boundary or behaviour of
#     any anomaly detection model.

#     Parameters
#     ----------
#     features : pd.DataFrame
#         Feature matrix.

#     projection : {"pca", "tsne", "umap"}, default="pca"
#         Dimensionality reduction technique used for the projection.

#     standardize : bool, default=False
#         Whether to standardize the features before computing the projection.

#     random_state : int, optional
#         Random state used by the projection algorithm.

#     title : str, optional
#         Figure title.

#     figsize : tuple[int, int], default=(8, 8)
#         Figure size.
#     """

#     if projection not in ("pca", "tsne", "umap"):
#         raise ValueError("projection must be 'pca' or 'umap'")

#     X = features

#     if standardize:
#         from sklearn.preprocessing import StandardScaler

#         X = StandardScaler().fit_transform(X)

#     if projection == "pca":
#         from sklearn.decomposition import PCA

#         reducer = PCA(
#             n_components=2,
#             random_state=random_state,
#         )

#     elif projection == "tsne":
#         from sklearn.manifold import TSNE

#         reducer = TSNE(
#             n_components=2,
#             random_state=random_state,
#             init="pca",
#             learning_rate="auto",
#         )

#     elif projection == "umap":

#         try:
#             import umap
#         except ImportError as exc:
#             raise ImportError(
#                 "UMAP projection requires the 'umap-learn' package."
#             ) from exc

#         reducer = umap.UMAP(
#             n_components=2,
#             random_state=random_state,
#         )

#     else:
#         raise ValueError(
#             f"Unknown projection method: {projection!r}."
#         )

#     X_proj = reducer.fit_transform(X)

#     fig, ax = plt.subplots(figsize=figsize)

#     ax.scatter(
#         X_proj[:, 0],
#         X_proj[:, 1],
#         s=15,
#         alpha=0.7,
#     )

#     ax.set_xlabel("Component 1")
#     ax.set_ylabel("Component 2")

#     if title is not None:
#         ax.set_title(title)

#     elif projection == "pca":

#         explained_variance = reducer.explained_variance_ratio_.sum()

#         ax.set_title(
#             f"PCA projection (2D projection retains {explained_variance:.1%} of the original information)"
#         )

#     else:

#         ax.set_title(f"{projection.upper()} projection")

#     ax.grid(alpha=0.3)

#     fig.tight_layout()

#     plt.show()






def plot_feature_space_projection(
    features: pd.DataFrame,
    projection: Literal["pca", "tsne", "umap"] = "pca",
    scaler: TransformerMixin | None = None,
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

    scaler : TransformerMixin, optional
        Scikit-learn compatible transformer implementing
        ``fit_transform()``, such as ``StandardScaler`` or
        ``RobustScaler``. If ``None``, the features are projected without
        preprocessing.

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

    figsize : tuple[int, int], default=(8, 8)
        Figure size.
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

            from sklearn.decomposition import PCA

            reducer = PCA(
                n_components=2,
                random_state=random_state,
            )

        case "tsne":

            from sklearn.manifold import TSNE

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

        anomaly_mask = np.asarray(anomaly_scores) < anomaly_threshold

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

    anomaly_mask = df[col_score] < anomaly_threshold

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

    figsize : tuple[int, int], default=(10, 5)
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
    y_real,
    y_pred,
    mae=None,
    title="Rendimiento del modelo",
    steps=None,
    color_real="black",
    color_pred="green"
):
    """
    Plots real vs predicted values.

    Parameters
    ----------
    y_real : pd.Series
    y_pred : array-like
    mae : float, optional
        If None, it will not be displayed.
    title : str
    steps : int, optional
        If provided, limits plot to first N samples.
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
        mae = mean_absolute_error(y_real, y_pred)

    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, y_real, label="Realidad", color=color_real, alpha=0.7)
    plt.plot(x_axis, y_pred, label="Predicción", color=color_pred, linestyle="--", alpha=0.8)

    plt.title(f"{title} — MAE: {mae:.4f}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



# def show_model_old(y_test_real, y_test_preds, mae_test=None, title="Rendimiento Modelo"):
#     """
#     Grafica la realidad y la predicción.
    
#     Y_test: pd.Series con valores reales
#     test_preds: np.array o lista con predicciones
#     mae_test: float, MAE del test
#     title: str, título opcional
#     """
#     # Si Y_test tiene índice temporal, lo usamos; si no, creamos rango numérico
#     x_axis = y_test_real.index if isinstance(y_test_real.index, pd.DatetimeIndex) else range(len(y_test_real))

#     if mae_test is not None:
#         mae_test = mean_absolute_error(y_test_real, y_test_preds)
    
#     plt.figure(figsize=(15, 6))
#     plt.plot(x_axis, y_test_real, label='Realidad', color='black', alpha=0.7)
#     plt.plot(x_axis, y_test_preds, label='Predicción', color='green', linestyle='--', alpha=0.8)
    
#     plt.title(f"{title} — MAE: {mae_test:.4f}")
#     plt.xlabel("Tiempo")
#     plt.ylabel("Valor")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()


# def show_model_2(y_test, y_test_preds, mae_test, pasos):
#     # pasos = 12 * 48 # 48 horas
#     plt.figure(figsize=(15, 7))
#     plt.plot(y_test.index[:pasos], y_test.iloc[:pasos], label='Realidad', color='black', alpha=0.7)
#     plt.plot(y_test.index[:pasos], y_test_preds[:pasos], label='IA Optimizada (Optuna)', color='green', linestyle='--')
#     plt.title(f'Rendimiento Final: MAE {mae_test:.4f}')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()


def plot_model_feature_importance(
    modelo,
    X,
    top_n=15,
    title=None
):
    """
    Muestra importancia de variables SOLO si el modelo lo soporta nativamente.

    Soporta:
    - Árboles / ensembles (feature_importances_)
    - CatBoost / similares (get_feature_importance)
    - Modelos lineales (coef_)

    NO calcula nada extra (no permutation importance).

    Example:
    '''
    Regresión lineal
    plot_feature_importance_general(modelo_lr, X_train)

    → usa coef_

    Random Forest
    plot_feature_importance_general(modelo_rf, X_train)

    → usa feature_importances_

    '''
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


# endregion --------------------------------------------------------------------
