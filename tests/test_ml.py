"""Unit tests for the ds_utils.ml module and plot_confusion_matrix."""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from ds_utils.ml.ml_utilities import (
    explain_anomalies,
    extract_used_params,
    format_max_features,
    normalize_anomaly_scores_min_max,
    normalize_anomaly_scores_rank,
    normalize_model_name,
)
from ds_utils.eda.plot_utilities import plot_confusion_matrix


# ==============================================================================
# Tests for normalize_model_name
# ==============================================================================
class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_standard_model_names(self):
        assert normalize_model_name("Isolation Forest") == "isolation_forest"
        assert normalize_model_name("Local Outlier Factor") == "local_outlier_factor"
        assert normalize_model_name("One-Class SVM") == "one_class_svm"
        assert normalize_model_name("Elliptic Envelope") == "elliptic_envelope"

    def test_already_normalized(self):
        assert normalize_model_name("isolation_forest") == "isolation_forest"

    def test_special_characters_and_consecutive_separators(self):
        assert normalize_model_name("Random---Forest___Classifier!!") == "random_forest_classifier"
        assert normalize_model_name("  Gradient  Boosting  ") == "gradient_boosting"

    def test_leading_and_trailing_underscores(self):
        assert normalize_model_name("__Model_Name__") == "model_name"

    def test_with_numbers(self):
        assert normalize_model_name("Model 2 v1.0") == "model_2_v1_0"

    def test_empty_or_symbols_only(self):
        assert normalize_model_name("---") == ""
        assert normalize_model_name("") == ""


# ==============================================================================
# Tests for format_max_features
# ==============================================================================
class TestFormatMaxFeatures:
    """Tests for format_max_features function."""

    def test_sqrt(self):
        result = format_max_features("sqrt", 16)
        # sqrt(16) / 16 = 4 / 16 = 0.25
        assert result == "sqrt (0.25)"

    def test_log2(self):
        result = format_max_features("log2", 16)
        # log2(16) / 16 = 4 / 16 = 0.25
        assert result == "log2 (0.25)"

    def test_none_uses_all_features(self):
        assert format_max_features(None, 20) == "all (1.00)"

    def test_float_fraction(self):
        assert format_max_features(0.5, 10) == "fraction (0.5)"
        assert format_max_features(0.75, 10) == "fraction (0.75)"

    def test_int_value(self):
        assert format_max_features(5, 10) == "fraction (5)"


# ==============================================================================
# Tests for extract_used_params
# ==============================================================================
class TestExtractUsedParams:
    """Tests for extract_used_params function."""

    def test_extract_present_parameters(self):
        all_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        used_params = {"n_estimators": 50, "random_state": 0}
        result = extract_used_params(all_params, used_params)
        assert result == {"n_estimators": 100, "random_state": 42}

    def test_ignore_params_not_in_all_params(self):
        all_params = {"n_estimators": 100}
        used_params = {"n_estimators": 100, "unknown_param": 999}
        result = extract_used_params(all_params, used_params)
        assert result == {"n_estimators": 100}

    def test_empty_inputs(self):
        assert extract_used_params({}, {"param": 1}) == {}
        assert extract_used_params({"param": 1}, {}) == {}
        assert extract_used_params({}, {}) == {}


# ==============================================================================
# Tests for explain_anomalies
# ==============================================================================
class TestExplainAnomalies:
    """Tests for explain_anomalies function."""

    @pytest.fixture
    def sample_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="h"),
            "temp": [20.0, 21.0, 20.5, 100.0, 20.2, 20.8],
            "pressure": [1.0, 1.1, 1.0, 1.2, 1.05, 1.0],
            "humidity": [50.0, 52.0, 51.0, 95.0, 50.5, 51.2],
        })
        scores = np.array([0.1, 0.2, 0.15, -0.9, 0.12, 0.18])
        # Anomaly mask: row 3 is anomalous
        mask = np.array([False, False, False, True, False, False])
        return df, scores, mask

    def test_normal_explanation(self, sample_data):
        df, scores, mask = sample_data
        result = explain_anomalies(
            df=df,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=mask,
            top_features=2,
            ascending=True,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["timestamp", "score", "summary", "details"]
        assert result.iloc[0]["score"] == -0.9
        assert result.iloc[0]["timestamp"] == df.iloc[3]["timestamp"]

        details = result.iloc[0]["details"]
        assert len(details) <= 2
        # Temp had the biggest deviation
        assert details[0]["feature"] == "temp"
        assert "temp" in result.iloc[0]["summary"]

    def test_multiple_anomalies_sorted(self, sample_data):
        df, scores, mask = sample_data
        # Mark row 1 and row 3 as anomalies
        mask[1] = True
        scores[1] = -0.5

        # Ascending sort: -0.9 first, then -0.5
        result_asc = explain_anomalies(
            df=df,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=mask,
            ascending=True,
        )
        assert len(result_asc) == 2
        assert result_asc.iloc[0]["score"] == -0.9
        assert result_asc.iloc[1]["score"] == -0.5

        # Descending sort: -0.5 first, then -0.9
        result_desc = explain_anomalies(
            df=df,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=mask,
            ascending=False,
        )
        assert result_desc.iloc[0]["score"] == -0.5
        assert result_desc.iloc[1]["score"] == -0.9

    def test_no_anomalies_detected_returns_empty_dataframe_with_columns(self, sample_data):
        df, scores, _ = sample_data
        empty_mask = np.zeros(len(df), dtype=bool)

        # Test ascending=True
        result_asc = explain_anomalies(
            df=df,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=empty_mask,
            ascending=True,
        )
        assert isinstance(result_asc, pd.DataFrame)
        assert len(result_asc) == 0
        assert list(result_asc.columns) == ["timestamp", "score", "summary", "details"]

        # Test ascending=False
        result_desc = explain_anomalies(
            df=df,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=empty_mask,
            ascending=False,
        )
        assert isinstance(result_desc, pd.DataFrame)
        assert len(result_desc) == 0
        assert list(result_desc.columns) == ["timestamp", "score", "summary", "details"]

    def test_empty_dataframe_input_robustness(self):
        df_empty = pd.DataFrame(columns=["timestamp", "temp", "pressure"])
        scores = np.array([])
        mask = np.array([], dtype=bool)

        result = explain_anomalies(
            df=df_empty,
            col_time="timestamp",
            anomaly_scores=scores,
            anomaly_mask=mask,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "score", "summary", "details"]


# ==============================================================================
# Tests for normalize_anomaly_scores_min_max
# ==============================================================================
class TestNormalizeAnomalyScoresMinMax:
    """Tests for normalize_anomaly_scores_min_max function."""

    def test_series_normalization(self):
        # 100 values ranging from -10 to 10
        scores = pd.Series(np.linspace(-10, 10, 101))
        norm = normalize_anomaly_scores_min_max(scores, 0.05, 0.95)

        assert isinstance(norm, pd.Series)
        assert len(norm) == len(scores)
        # Values should be clipped between 0 and 1
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        # Inverted logic: lowest score gets the highest normalized anomaly score (~1.0)
        assert norm.iloc[0] == 1.0
        assert norm.iloc[-1] == 0.0

    def test_dataframe_normalization(self):
        df = pd.DataFrame({
            "model_a": np.linspace(-5, 5, 50),
            "model_b": np.linspace(100, 200, 50),
        })
        norm = normalize_anomaly_scores_min_max(df)

        assert isinstance(norm, pd.DataFrame)
        assert list(norm.columns) == ["model_a", "model_b"]
        assert norm["model_a"].min() >= 0.0
        assert norm["model_a"].max() <= 1.0
        assert norm["model_b"].min() >= 0.0
        assert norm["model_b"].max() <= 1.0

    def test_invalid_percentiles_raise_value_error(self):
        scores = pd.Series([1.0, 2.0, 3.0])
        # lower >= upper
        with pytest.raises(ValueError, match="Expected 0 <= lower_percentile < upper_percentile <= 1"):
            normalize_anomaly_scores_min_max(scores, lower_percentile=0.5, upper_percentile=0.5)

        # lower < 0
        with pytest.raises(ValueError):
            normalize_anomaly_scores_min_max(scores, lower_percentile=-0.1, upper_percentile=0.9)

        # upper > 1
        with pytest.raises(ValueError):
            normalize_anomaly_scores_min_max(scores, lower_percentile=0.1, upper_percentile=1.2)

    def test_equal_limits_constant_values(self):
        # When all values are equal, upper == lower, so output should be 0.0 without division by zero error
        scores = pd.Series([5.0, 5.0, 5.0, 5.0])
        norm = normalize_anomaly_scores_min_max(scores)
        assert isinstance(norm, pd.Series)
        assert (norm == 0.0).all()


# ==============================================================================
# Tests for normalize_anomaly_scores_rank
# ==============================================================================
class TestNormalizeAnomalyScoresRank:
    """Tests for normalize_anomaly_scores_rank function."""

    def test_series_rank_normalization(self):
        scores = pd.Series([10.0, 20.0, 30.0, 40.0])
        norm = normalize_anomaly_scores_rank(scores)

        assert isinstance(norm, pd.Series)
        # Lowest raw score (10.0) represents most anomalous -> rank pct is 0.25 -> 1 - 0.25 = 0.75
        # Highest raw score (40.0) -> rank pct is 1.0 -> 1 - 1.0 = 0.0
        assert norm.iloc[0] > norm.iloc[-1]
        assert norm.iloc[-1] == 0.0

    def test_dataframe_rank_normalization(self):
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0],
            "col2": [10.0, 20.0, 30.0],
        })
        norm = normalize_anomaly_scores_rank(df)

        assert isinstance(norm, pd.DataFrame)
        assert list(norm.columns) == ["col1", "col2"]
        assert (norm["col1"] == norm["col2"]).all()




# ==============================================================================
# Tests for plot_confusion_matrix
# ==============================================================================
class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    @patch("ds_utils.eda.plot_utilities.plt.show")
    @patch("ds_utils.eda.plot_utilities.sns.heatmap")
    @patch("ds_utils.eda.plot_utilities.plt.subplots")
    def test_standard_2x2_confusion_matrix(self, mock_subplots, mock_heatmap, mock_show):
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        cm = np.array([[50, 10], [5, 35]])
        plot_confusion_matrix(cm)

        mock_subplots.assert_called_once()
        mock_heatmap.assert_called_once()
        mock_show.assert_called_once()

        # Check heatmap arguments
        call_kwargs = mock_heatmap.call_args.kwargs
        assert call_kwargs["xticklabels"] == ["Class 0", "Class 1"]
        assert call_kwargs["yticklabels"] == ["Class 0", "Class 1"]
        assert call_kwargs["fmt"] == "g"

    @patch("ds_utils.eda.plot_utilities.plt.show")
    @patch("ds_utils.eda.plot_utilities.sns.heatmap")
    @patch("ds_utils.eda.plot_utilities.plt.subplots")
    def test_custom_labels_and_title(self, mock_subplots, mock_heatmap, mock_show):
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        cm = np.array([[40, 2], [3, 45]])
        plot_confusion_matrix(cm, labels=["Negative", "Positive"], title="My Model CM")

        call_kwargs = mock_heatmap.call_args.kwargs
        assert call_kwargs["xticklabels"] == ["Negative", "Positive"]
        ax_mock.set_title.assert_called_with("My Model CM")

    @patch("ds_utils.eda.plot_utilities.plt.show")
    @patch("ds_utils.eda.plot_utilities.sns.heatmap")
    @patch("ds_utils.eda.plot_utilities.plt.subplots")
    def test_dataframe_input_extracts_labels(self, mock_subplots, mock_heatmap, mock_show):
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        df_cm = pd.DataFrame([[10, 2], [1, 15]], columns=["Cat", "Dog"], index=["Cat", "Dog"])
        plot_confusion_matrix(df_cm)

        call_kwargs = mock_heatmap.call_args.kwargs
        assert call_kwargs["xticklabels"] == ["Cat", "Dog"]

    @patch("ds_utils.eda.plot_utilities.plt.show")
    @patch("ds_utils.eda.plot_utilities.sns.heatmap")
    @patch("ds_utils.eda.plot_utilities.plt.subplots")
    def test_show_binary_labels(self, mock_subplots, mock_heatmap, mock_show):
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        cm = np.array([[100, 10], [5, 85]])
        plot_confusion_matrix(cm, show_binary_labels=True)

        call_kwargs = mock_heatmap.call_args.kwargs
        # Annot should be a 2x2 array of custom strings containing TN, FP, FN, TP
        annot = call_kwargs["annot"]
        assert annot.shape == (2, 2)
        assert "TN" in annot[0, 0]
        assert "FP" in annot[0, 1]
        assert "FN" in annot[1, 0]
        assert "TP" in annot[1, 1]
        assert call_kwargs["fmt"] == ""

    def test_invalid_dimensions_raises_value_error(self):
        # 1D array
        with pytest.raises(ValueError, match="The confusion matrix must be a non-empty square matrix"):
            plot_confusion_matrix([1, 2, 3])

        # Non-square matrix (2x3)
        with pytest.raises(ValueError, match="The confusion matrix must be a non-empty square matrix"):
            plot_confusion_matrix([[1, 2, 3], [4, 5, 6]])

        # Empty matrix
        with pytest.raises(ValueError, match="The confusion matrix must be a non-empty square matrix"):
            plot_confusion_matrix(np.zeros((0, 0)))

    def test_mismatched_labels_raises_value_error(self):
        cm = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="The number of labels must match the matrix dimensions"):
            plot_confusion_matrix(cm, labels=["A", "B", "C"])

    def test_show_binary_labels_on_non_2x2_raises_value_error(self):
        cm = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError, match="Binary labels can only be displayed for a 2x2 matrix"):
            plot_confusion_matrix(cm, show_binary_labels=True)
