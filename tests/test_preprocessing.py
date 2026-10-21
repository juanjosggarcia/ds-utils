"""Unit tests for the ds_utils.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from ds_utils.preprocessing.preprocessing_utilities import (
    convert_timestamp,
    convert_timestamps_vectorized,
    count_cells_with_missing,
    count_columns_with_missing,
    count_constant_columns,
    count_edge_null_rows,
    count_missing_cells_by_column,
    count_missing_timestamps,
    count_rows_by_time,
    count_rows_with_missing,
    detect_timestamp,
    find_column_pairs,
    find_duplicates,
    find_large_gaps,
    get_columns_summary,
    handle_duplicates,
    handle_missing_values,
    handle_sensor_noise,
    handle_time_bursts,
    handle_time_split,
    prepare_dataframe_for_ml,
    prepare_time_features,
    regression_metrics,
    remove_constant_columns,
    remove_redundant_sensor_noise_columns,
    remove_replaceable_columns,
    resample_time,
    reset_index_safe,
    set_index_safe,
    validate_required_columns,
)


# ==============================================================================
# 1. Tests for Handlers
# ==============================================================================
class TestPreprocessingHandlers:
    """Tests for handler functions in preprocessing."""

    def test_handle_duplicates_first_and_last(self):
        df = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "value": [10, 20, 30],
        })

        clean_first, dups_first = handle_duplicates(df, col_time="timestamp", keep="first")
        assert len(clean_first) == 2
        assert len(dups_first) == 1
        assert clean_first["value"].tolist() == [10, 30]
        assert dups_first["value"].tolist() == [20]

        clean_last, dups_last = handle_duplicates(df, col_time="timestamp", keep="last")
        assert len(clean_last) == 2
        assert len(dups_last) == 1
        assert clean_last["value"].tolist() == [20, 30]
        assert dups_last["value"].tolist() == [10]

    def test_handle_duplicates_invalid_keep_raises_value_error(self):
        df = pd.DataFrame({"time": ["2023-01-01"], "val": [1]})
        with pytest.raises(ValueError, match="keep must be 'first' or 'last'"):
            handle_duplicates(df, col_time="time", keep="invalid")  # type: ignore

    def test_handle_time_split_tail_and_head(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01 10:00:00", periods=5, freq="h"),
            "val": [1, 2, 3, 4, 5],
        })
        cutoff = "2023-01-01 12:00:00"

        # mode="tail": main <= cutoff, removed > cutoff
        main_tail, rem_tail = handle_time_split(df, col_time="timestamp", cutoff_date=cutoff, mode="tail")
        assert len(main_tail) == 3
        assert len(rem_tail) == 2
        assert main_tail["val"].tolist() == [1, 2, 3]
        assert rem_tail["val"].tolist() == [4, 5]

        # mode="head": main > cutoff, removed <= cutoff
        main_head, rem_head = handle_time_split(df, col_time="timestamp", cutoff_date=cutoff, mode="head")
        assert len(main_head) == 2
        assert len(rem_head) == 3
        assert main_head["val"].tolist() == [4, 5]
        assert rem_head["val"].tolist() == [1, 2, 3]

    def test_handle_time_split_date_only_cutoff(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01 00:00:00", periods=3, freq="D"),
            "val": [1, 2, 3],
        })
        # Date only: entire day 2023-01-02 is included in tail
        main, rem = handle_time_split(df, col_time="timestamp", cutoff_date="2023-01-02", mode="tail")
        assert len(main) == 2
        assert len(rem) == 1
        assert main["val"].tolist() == [1, 2]
        assert rem["val"].tolist() == [3]

    def test_handle_time_split_invalid_mode_raises(self):
        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=2, freq="D")})
        with pytest.raises(ValueError, match="Invalid mode"):
            handle_time_split(df, "time", "2023-01-01", mode="middle")  # type: ignore

    def test_handle_time_bursts(self):
        df = pd.DataFrame({
            "timestamp": [
                "2023-01-01 10:00:00",
                "2023-01-01 10:01:00",  # burst (< 5min)
                "2023-01-01 10:05:00",  # retained (>= 5min from 10:00)
                "2023-01-01 10:07:00",  # burst (< 5min from 10:05)
                "2023-01-01 10:10:00",  # retained (>= 5min from 10:05)
            ],
            "val": [1, 2, 3, 4, 5],
        })
        clean, removed = handle_time_bursts(df, col_time="timestamp", freq="5min")
        assert len(clean) == 3
        assert len(removed) == 2
        assert clean["val"].tolist() == [1, 3, 5]
        assert removed["val"].tolist() == [2, 4]

    def test_handle_time_bursts_invalid_freq_raises(self):
        df = pd.DataFrame({"time": ["2023-01-01 10:00:00"]})
        with pytest.raises(ValueError, match="Invalid freq"):
            handle_time_bursts(df, col_time="time", freq="not_a_freq")

    def test_handle_sensor_noise_isolated_spike(self):
        df = pd.DataFrame({
            "temperature": [20.0, 20.2, 50.0, 20.1, 20.3],
        })
        clean_df, noise_df = handle_sensor_noise(
            df=df,
            tolerances={"temperature": 2.0},
            window=2,
            add_extra_features=True,
        )

        # Spike at index 2 (50.0) should be corrected to 20.2
        assert clean_df["temperature_clean"].iloc[2] == 20.2
        assert clean_df["temperature_is_noise"].iloc[2] is True or clean_df["temperature_is_noise"].iloc[2] == 1
        assert len(noise_df) == 1
        assert "temperature_diff" in clean_df.columns
        assert "temperature_noise_magnitude" in clean_df.columns

    def test_handle_sensor_noise_persistent_change_not_corrected(self):
        # Sustained shift: 20 -> 50 -> 50 -> 50 (real change, not isolated spike)
        df = pd.DataFrame({
            "temperature": [20.0, 50.0, 50.0, 50.0],
        })
        clean_df, noise_df = handle_sensor_noise(
            df=df,
            tolerances={"temperature": 2.0},
            window=2,
        )
        # Should NOT be flagged as noise because window stayed at 50
        assert not clean_df["temperature_is_noise"].any()
        assert len(noise_df) == 0

    def test_handle_sensor_noise_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Sensor column 'missing' not found"):
            handle_sensor_noise(df, tolerances={"missing": 1.0})

    def test_handle_missing_values_audit_only(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", "y", None],
        })
        clean_df, audit_df = handle_missing_values(df, fill_value=None)
        assert clean_df.isna().sum().sum() == 2
        assert len(audit_df) == 2
        assert set(audit_df["column"]) == {"a", "b"}

    def test_handle_missing_values_scalar_fill(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        clean_df, audit_df = handle_missing_values(df, fill_value=0.0)
        assert clean_df["a"].isna().sum() == 0
        assert clean_df["a"].iloc[1] == 0.0

    def test_handle_missing_values_dict_fill(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["cat", None, "dog"],
        })
        clean_df, _ = handle_missing_values(df, fill_value={"a": -1.0})
        assert clean_df["a"].iloc[1] == -1.0
        # b was not in fill_value dict, so it stays None
        assert pd.isna(clean_df["b"].iloc[1])

    def test_handle_missing_values_dict_with_missing_col_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            handle_missing_values(df, fill_value={"non_existent": 0})


# ==============================================================================
# 2. Tests for Finders
# ==============================================================================
class TestPreprocessingFinders:
    """Tests for finder functions in preprocessing."""

    def test_find_duplicates(self):
        df = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "val": [1, 2, 3],
        })
        n_dups, dups_df = find_duplicates(df, col_time="timestamp")
        assert n_dups == 1
        assert len(dups_df) == 2
        assert dups_df["val"].tolist() == [1, 2]

    def test_find_duplicates_no_duplicates(self):
        df = pd.DataFrame({"timestamp": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]})
        n_dups, dups_df = find_duplicates(df, col_time="timestamp")
        assert n_dups == 0
        assert len(dups_df) == 0

    def test_find_large_gaps(self):
        df = pd.DataFrame({
            "timestamp": [
                "2023-01-01 10:00:00",
                "2023-01-01 10:05:00",
                "2023-01-01 10:20:00",  # gap of 15 min > 5 min
            ]
        })
        gaps = find_large_gaps(df, col_time="timestamp", freq="5min", limit=1)
        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_periods"] == 2
        assert gaps.iloc[0]["duration"] == pd.Timedelta("15min")

    def test_find_large_gaps_invalid_freq_raises(self):
        df = pd.DataFrame({"timestamp": ["2023-01-01 10:00:00"]})
        with pytest.raises(ValueError, match="Invalid freq"):
            find_large_gaps(df, col_time="timestamp", freq="bad_freq")

    def test_find_column_pairs_suffix(self):
        df = pd.DataFrame({
            "temp": [1],
            "temp_clean": [1],
            "press": [2],
            "press_clean": [2],
            "humidity": [3],
        })
        orig, repl = find_column_pairs(df, suffix="_clean")
        assert orig == ["press", "temp"]
        assert repl == ["press_clean", "temp_clean"]

    def test_find_column_pairs_prefix(self):
        df = pd.DataFrame({
            "temp": [1],
            "clean_temp": [1],
            "noise": [2],
        })
        orig, repl = find_column_pairs(df, prefix="clean_")
        assert orig == ["temp"]
        assert repl == ["clean_temp"]

    def test_find_column_pairs_validation_errors(self):
        df = pd.DataFrame({"a": [1]})
        # Neither prefix nor suffix
        with pytest.raises(ValueError, match="At least one of 'prefix' or 'suffix'"):
            find_column_pairs(df)

        # Both prefix and suffix
        with pytest.raises(ValueError, match="'prefix' and 'suffix' cannot be used together"):
            find_column_pairs(df, prefix="p_", suffix="_s")


# ==============================================================================
# 3. Tests for Counters
# ==============================================================================
class TestPreprocessingCounters:
    """Tests for counter functions in preprocessing."""

    def test_count_missing_timestamps(self):
        df = pd.DataFrame({
            "time": ["2023-01-01 10:00:00", "2023-01-01 10:15:00"],
        })
        # 10:05 and 10:10 are missing
        n_missing, missing_idx = count_missing_timestamps(df, col_time="time", freq="5min")
        assert n_missing == 2
        assert len(missing_idx) == 2

    def test_count_missing_timestamps_empty(self):
        df = pd.DataFrame({"time": []})
        n_missing, missing_idx = count_missing_timestamps(df, col_time="time", freq="5min")
        assert n_missing == 0
        assert len(missing_idx) == 0

    def test_count_missing_timestamps_invalid_freq(self):
        df = pd.DataFrame({"time": ["2023-01-01"]})
        with pytest.raises(ValueError, match="Invalid freq"):
            count_missing_timestamps(df, col_time="time", freq="invalid")

    def test_count_edge_null_rows(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 10.0, 20.0, np.nan],
            "b": [np.nan, 5.0, 10.0, 20.0, np.nan],
        })
        # mode="any": start row 0 (both null) and row 1 (a is null) -> 2 rows. end row 4 -> 1 row
        n_start, n_end = count_edge_null_rows(df, mode="any")
        assert n_start == 2
        assert n_end == 1

        # mode="all": start row 0 (both null) -> 1 row. end row 4 (both null) -> 1 row
        n_start_all, n_end_all = count_edge_null_rows(df, mode="all")
        assert n_start_all == 1
        assert n_end_all == 1

    def test_count_edge_null_rows_invalid_mode(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Invalid mode"):
            count_edge_null_rows(df, mode="other")  # type: ignore

    def test_count_rows_by_time(self):
        df = pd.DataFrame({
            "time": [
                "2023-01-01 10:00:00",
                "2023-01-01 11:00:00",
                "2023-01-02 08:00:00",
            ]
        })
        result = count_rows_by_time(df, col_time="time", freq="D")
        assert len(result) == 2
        assert result["n_rows"].tolist() == [2, 1]

    def test_count_rows_by_time_invalid_freq(self):
        df = pd.DataFrame({"time": ["2023-01-01"]})
        with pytest.raises(ValueError):
            count_rows_by_time(df, col_time="time", freq="unknown")

    def test_count_constant_columns(self):
        df = pd.DataFrame({
            "const1": [1, 1, 1],
            "const_with_nan": [1, np.nan, 1],
            "var": [1, 2, 3],
        })
        # dropna=True: const1 (1 unique), const_with_nan (1 unique ignoring nan) -> 2
        assert count_constant_columns(df, dropna=True) == 2
        # dropna=False: const_with_nan has 2 unique (1 and nan) -> only const1 is constant
        assert count_constant_columns(df, dropna=False) == 1

    def test_count_cells_with_missing(self):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0],
            "text": ["hello", "   ", None],
        })
        # include_empty_strings=True: nan (1) + "   " (1) + None (1) = 3
        assert count_cells_with_missing(df, include_empty_strings=True) == 3
        # include_empty_strings=False: nan (1) + None (1) = 2
        assert count_cells_with_missing(df, include_empty_strings=False) == 2

    def test_count_rows_with_missing(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3, 4],
            "b": ["a", "b", " ", "d"],
        })
        # row 1 has nan, row 2 has empty string
        assert count_rows_with_missing(df, include_empty_strings=True) == 2
        assert count_rows_with_missing(df, include_empty_strings=False) == 1

    def test_count_columns_with_missing(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1, np.nan, 3],
            "c": ["a", "  ", "c"],
        })
        assert count_columns_with_missing(df, include_empty_strings=True) == 2
        assert count_columns_with_missing(df, include_empty_strings=False) == 1

    def test_count_missing_cells_by_column(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan],
            "b": ["a", " ", None],
            "c": [10, 20, 30],
        })
        res = count_missing_cells_by_column(df, include_empty_strings=True)
        assert res["a"] == 2
        assert res["b"] == 2
        assert res["c"] == 0


# ==============================================================================
# 4. Tests for Removes
# ==============================================================================
class TestPreprocessingRemoves:
    """Tests for removal functions in preprocessing."""

    def test_remove_redundant_sensor_noise_columns(self):
        # sensor 'temp' clean signal is identical to original (no noise was found)
        # sensor 'press' clean signal is different (noise was found and corrected)
        df = pd.DataFrame({
            "temp": [20.0, 21.0, 22.0],
            "temp_clean": [20.0, 21.0, 22.0],
            "temp_is_noise": [False, False, False],
            "press": [1.0, 50.0, 1.2],
            "press_clean": [1.0, 1.0, 1.2],
            "press_is_noise": [False, True, False],
        })
        tolerances = {"temp": 2.0, "press": 2.0}
        df_clean, removed = remove_redundant_sensor_noise_columns(df, tolerances)

        # temp_clean and temp_is_noise should be removed
        assert "temp_clean" not in df_clean.columns
        assert "temp_is_noise" not in df_clean.columns
        assert "temp_clean" in removed
        # press_clean and press_is_noise should be retained
        assert "press_clean" in df_clean.columns
        assert "press_is_noise" in df_clean.columns

    def test_remove_constant_columns(self):
        df = pd.DataFrame({
            "a": [1, 1, 1],
            "b": [1, 2, 3],
            "c": ["x", "x", "x"],
        })
        df_clean, removed = remove_constant_columns(df)
        assert removed == ["a", "c"]
        assert list(df_clean.columns) == ["b"]

    def test_remove_constant_columns_with_exclusions(self):
        df = pd.DataFrame({
            "const_id": [99, 99, 99],
            "const_drop": [1, 1, 1],
            "var": [1, 2, 3],
        })
        df_clean, removed = remove_constant_columns(df, excluded_columns=["const_id"])
        assert removed == ["const_drop"]
        assert "const_id" in df_clean.columns

    def test_remove_replaceable_columns(self):
        df = pd.DataFrame({
            "temp": [1, 2],
            "temp_clean": [1, 2],
            "press": [10, 20],
        })
        df_clean, removed = remove_replaceable_columns(df, replacement_suffix="_clean")
        assert removed == ["temp"]
        assert list(df_clean.columns) == ["temp_clean", "press"]

    def test_remove_replaceable_columns_validation(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="At least one of 'replacement_prefix'"):
            remove_replaceable_columns(df)

        with pytest.raises(ValueError, match="'replacement_prefix' and 'replacement_suffix' cannot be used together"):
            remove_replaceable_columns(df, replacement_prefix="p_", replacement_suffix="_s")


# ==============================================================================
# 5. Tests for Features
# ==============================================================================
class TestPreprocessingFeatures:
    """Tests for feature generation and resampling in preprocessing."""

    def test_prepare_time_features(self):
        df = pd.DataFrame({
            "time": ["2023-01-01 14:30:00", "2023-01-02 08:15:00"],
            "val": [10, 20],
        })
        res = prepare_time_features(
            df=df,
            col_time="time",
            hour_linear=True,
            hour_cyclic=True,
            weekday_linear="both",
            weekday_cyclic=True,
        )

        assert "hour" in res.columns
        assert "hour_sin" in res.columns
        assert "hour_cos" in res.columns
        assert "weekday" in res.columns
        assert "weekday_text" in res.columns
        assert "week_sin" in res.columns
        assert "week_cos" in res.columns
        assert res.columns[0] == "time"

    def test_prepare_time_features_invalid_weekday_linear(self):
        df = pd.DataFrame({"time": ["2023-01-01"]})
        with pytest.raises(ValueError, match="Invalid week_linear"):
            prepare_time_features(df, col_time="time", weekday_linear="invalid")  # type: ignore

    def test_resample_time_round(self):
        # 10:01 and 10:06 round to 10:00 and 10:05
        df = pd.DataFrame({
            "time": ["2023-01-01 10:01:00", "2023-01-01 10:06:00", "2023-01-01 10:16:00"],
            "val": [1.0, 2.0, 4.0],
        })
        resampled = resample_time(df, col_time="time", freq="5min", alignment="round", fill_method="ffill")
        # Timestamps: 10:00, 10:05, 10:10 (filled from 10:05), 10:15
        assert len(resampled) == 4
        assert resampled["val"].iloc[2] == 2.0  # forward-filled

    def test_resample_time_preserve(self):
        df = pd.DataFrame({
            "time": ["2023-01-01 10:01:00", "2023-01-01 10:11:00"],
            "val": [1.0, 2.0],
        })
        resampled = resample_time(df, col_time="time", freq="5min", alignment="preserve")
        assert len(resampled) == 3
        assert resampled["time"].iloc[0] == pd.Timestamp("2023-01-01 10:01:00")

    def test_resample_time_duplicate_timestamps_raises(self):
        df = pd.DataFrame({
            "time": ["2023-01-01 10:00:00", "2023-01-01 10:00:00"],
            "val": [1, 2],
        })
        with pytest.raises(ValueError, match="contains duplicate timestamps"):
            resample_time(df, col_time="time", freq="5min")


# ==============================================================================
# 6. Tests for Others (Utilities & Conversions)
# ==============================================================================
class TestPreprocessingOthers:
    """Tests for index, timestamp and summary utilities in preprocessing."""

    def test_validate_required_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        # Passing valid columns
        validate_required_columns(df, ["a", "b"])
        validate_required_columns(df, "a")

        # Missing column raises KeyError
        with pytest.raises(KeyError, match="Missing columns in the DataFrame: \\['c'\\]"):
            validate_required_columns(df, ["a", "c"])

    def test_set_index_safe(self):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df_indexed = set_index_safe(df, "id")
        assert df_indexed.index.name == "id"

        # Safe if already index
        df_same = set_index_safe(df_indexed, "id")
        assert df_same.index.name == "id"

        # Missing column raises KeyError
        with pytest.raises(KeyError):
            set_index_safe(df, "missing")

    def test_reset_index_safe(self):
        # Default RangeIndex: drops safely
        df = pd.DataFrame({"a": [1, 2]})
        res = reset_index_safe(df)
        assert isinstance(res.index, pd.RangeIndex)
        assert list(res.columns) == ["a"]

        # Custom index: resets into column
        df_custom = pd.DataFrame({"val": [10, 20]}, index=["row1", "row2"])
        res_custom = reset_index_safe(df_custom)
        assert "index" in res_custom.columns

    def test_detect_timestamp(self):
        # Seconds
        sec = detect_timestamp("1672531199")
        assert "Epoch in seconds" in sec

        # Milliseconds (13+ digits)
        ms = detect_timestamp("1672531199000")
        assert "Epoch in milliseconds" in ms

        # ISO 8601
        iso = detect_timestamp("2023-01-01T12:00:00Z")
        assert "ISO 8601" in iso

        # Unknown
        unknown = detect_timestamp("invalid_date_string")
        assert unknown == "Unknown timestamp format"

    def test_convert_timestamp(self):
        # Seconds
        ts_s = convert_timestamp(1672531200)
        assert isinstance(ts_s, pd.Timestamp)

        # Milliseconds
        ts_ms = convert_timestamp(1672531200000)
        assert isinstance(ts_ms, pd.Timestamp)

        # ISO string
        ts_iso = convert_timestamp("2023-01-01T00:00:00Z")
        assert isinstance(ts_iso, pd.Timestamp)

        # Invalid format returns NaT
        assert pd.isna(convert_timestamp("invalid_ts"))

    def test_convert_timestamps_vectorized(self):
        s = pd.Series([
            1672531200,             # seconds
            1672531200000,          # ms
            "2023-01-01T00:00:00Z", # ISO
            None,                   # null
        ])
        converted = convert_timestamps_vectorized(s, utc=True)
        assert len(converted) == 4
        assert pd.isna(converted.iloc[3])
        assert str(converted.dtype).startswith("datetime64[ns")

    def test_convert_timestamps_vectorized_empty(self):
        empty_s = pd.Series([], dtype="object")
        res = convert_timestamps_vectorized(empty_s, utc=True)
        assert len(res) == 0
        assert str(res.dtype).startswith("datetime64[ns")

    def test_get_columns_summary(self):
        df = pd.DataFrame({
            "col_a": [1, 2, np.nan],
            "col_b": ["x", " ", "z"],
        })
        summary = get_columns_summary(df, sort_by="nombre_columna", ascending=True)
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == [
            "nombre_columna",
            "tipo_dato",
            "num_valores_distintos",
            "num_valores_no_validos",
            "porcentaje_no_validos",
        ]
        assert summary.iloc[0]["nombre_columna"] == "col_a"

    def test_get_columns_summary_invalid_sort_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Invalid sort_by value"):
            get_columns_summary(df, sort_by="invalid_sort_key")


# ==============================================================================
# 7. Tests for ML functions in Preprocessing
# ==============================================================================
class TestPreprocessingML:
    """Tests for ML-related utilities in preprocessing."""

    def test_prepare_dataframe_for_ml_boolean_and_numeric_strings(self):
        df = pd.DataFrame({
            "bool_col": [True, False, True],
            "str_bool": ["true", "false", "1"],
            "str_num": ["10", "20.5", "30"],
            "dt_col": pd.date_range("2023-01-01", periods=3, freq="D"),
        })

        processed = prepare_dataframe_for_ml(df, bool_as_int=True)

        assert processed["bool_col"].dtype in [np.int64, int]
        assert processed["str_bool"].dtype in [np.int64, int]
        assert pd.api.types.is_numeric_dtype(processed["str_num"])
        assert pd.api.types.is_numeric_dtype(processed["dt_col"])

    def test_prepare_dataframe_for_ml_categorical_strategies(self):
        df = pd.DataFrame({
            "category_col": ["apple", "banana", "apple"],
        })

        # "category" strategy
        res_cat = prepare_dataframe_for_ml(df, categorical_strategy="category")
        assert isinstance(res_cat["category_col"].dtype, pd.CategoricalDtype)

        # "label" strategy
        res_label = prepare_dataframe_for_ml(df, categorical_strategy="label")
        assert pd.api.types.is_integer_dtype(res_label["category_col"])

        # "onehot" strategy
        res_onehot = prepare_dataframe_for_ml(df, categorical_strategy="onehot")
        assert "category_col_apple" in res_onehot.columns
        assert "category_col_banana" in res_onehot.columns

        # "drop" strategy
        res_drop = prepare_dataframe_for_ml(df, categorical_strategy="drop")
        assert "category_col" not in res_drop.columns

    def test_prepare_dataframe_for_ml_fillna_and_excluded(self):
        df = pd.DataFrame({
            "feat_with_nan": [1.0, np.nan, 3.0],
            "excluded": ["keep_me", "as_is", "raw"],
        })

        processed = prepare_dataframe_for_ml(
            df,
            fillna_cols=["feat_with_nan"],
            excluded_cols=["excluded"],
        )

        assert processed["feat_with_nan"].iloc[1] == 0.0
        assert processed["excluded"].iloc[0] == "keep_me"

    def test_prepare_dataframe_for_ml_invalid_strategy_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Invalid categorical_strategy"):
            prepare_dataframe_for_ml(df, categorical_strategy="invalid_strat")  # type: ignore

    def test_regression_metrics(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        metrics = regression_metrics(y_true, y_pred, decimals=4)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert metrics["mae"] == round(float(expected_mae), 4)

    def test_regression_metrics_different_lengths_raises(self):
        with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
            regression_metrics([1, 2], [1])

    def test_regression_metrics_constant_y_true_raises(self):
        with pytest.raises(ValueError, match="R² is undefined when y_true is constant"):
            regression_metrics([5.0, 5.0, 5.0], [4.9, 5.1, 5.0])

