from ds_utils.preprocessing.feature_config import FEATURES_CONFIG

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Collection, Literal, TypeAlias
from numpy.typing import ArrayLike
from collections.abc import Sequence
from pandas.tseries.frequencies import to_offset
from enum import Enum
from datetime import datetime

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region Typing Class ----------------------------------------------------------

SummaryColumn: TypeAlias = Literal[
    "nombre_columna",
    "tipo_dato",
    "num_valores_distintos",
    "num_valores_no_validos",
    "porcentaje_no_validos",
]

# VALID_SUMMARY_COLUMNS = [*SummaryColumn]

VALID_SUMMARY_COLUMNS = (
    "nombre_columna",
    "tipo_dato",
    "num_valores_distintos",
    "num_valores_no_validos",
    "porcentaje_no_validos",
)

# SummaryColumn: TypeAlias = Literal[*VALID_SUMMARY_COLUMNS]

# endregion Typing Class -------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# endregion Aux Functions ------------------------------------------------------


# region Handlers Functions ----------------------------------------------------

from ._features_handlers_utilities import (
    handle_duplicates,
    handle_time_split,
    handle_time_bursts,
    handle_sensor_noise,
    handle_missing_values,
)

# endregion Handlers Functions -------------------------------------------------

# region Finders Functions -----------------------------------------------------

from ._features_finders_utilities import (
    find_duplicates,
    find_large_gaps,
    find_column_pairs,
)

# endregion Finders Functions --------------------------------------------------

# region Counters Functions ----------------------------------------------------

from ._features_counters_utilities import (
    count_missing_timestamps,
    count_edge_null_rows,
    count_rows_by_time,
    count_constant_columns,
    count_cells_with_missing,
    count_rows_with_missing,
    count_columns_with_missing,
    count_missing_cells_by_column
)

# endregion Counters Functions -------------------------------------------------

# region Remove Functions ------------------------------------------------------

from ._features_removes_utilities import (
    remove_redundant_sensor_noise_columns,
    remove_constant_columns,
    remove_replaceable_columns
)

# endregion Remove Functions ---------------------------------------------------

# region Features Functions ----------------------------------------------------

from ._features_features_utilities import (
    resample_time,
    prepare_time_features
)

# endregion Features Functions -------------------------------------------------

# region Others Functions ------------------------------------------------------

from ._features_others_utilities import (
    validate_required_columns,
    set_index_safe,
    reset_index_safe,
    detect_timestamp,
    convert_timestamp,
    get_columns_summary
)

# endregion Others Functions ---------------------------------------------------

# region ML Functions ----------------------------------------------------------

from ._features_ml_utilities import (
    prepare_dataframe_for_ml
)

# endregion ML Functions -------------------------------------------------------

# region Functions with "lazy import" ------------------------------------------

# endregion Functions with "lazy import" ---------------------------------------