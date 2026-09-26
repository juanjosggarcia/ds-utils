from ds_utils.eda.preview_config import (
    Alignment,
    Color,
    DEFAULT_COLOR,
)
from ds_utils.preprocessing.preprocessing_config import (
    FEATURES_CONFIG,
    FeatureNamingConfig,
)

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Week, Day, Hour, Minute, Second
from IPython.display import display, HTML

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def format_time_gap(freq: str, periods: int) -> str:
    """
    Format a time interval as a human-readable Spanish duration.

    The function converts a pandas frequency and a number of periods into
    a duration expressed in weeks, days, hours, minutes, or seconds.

    Parameters
    ----------
    freq : str
        Pandas frequency string. Supported frequencies are weeks (``"W"``),
        days (``"D"``), hours (``"h"``), minutes (``"min"``), and seconds
        (``"s"``).

    periods : int
        Number of periods to include in the duration.

    Returns
    -------
    str
        Human-readable duration in Spanish, using the largest applicable
        time unit.

    Raises
    ------
    ValueError
        If ``freq`` is invalid or represents an unsupported frequency.

    Examples
    --------
    >>> format_time_gap("s", 10)
    '10 segundos'
    >>> format_time_gap("min", 5)
    '5 minutos'
    >>> format_time_gap("h", 2)
    '2 horas'
    >>> format_time_gap("D", 1)
    '1 día'
    >>> format_time_gap("W", 3)
    '3 semanas'
    """

    # Validar freq
    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid freq '{freq}'") from exc


    match offset:
        case Week():
            value = offset.n * periods
            unit = "semana" if value == 1 else "semanas"
        case Day():
            value = offset.n * periods
            unit = "día" if value == 1 else "días"
        case Hour():
            value = offset.n * periods
            unit = "hora" if value == 1 else "horas"
        case Minute():
            value = offset.n * periods
            unit = "minuto" if value == 1 else "minutos"
        case Second():
            value = offset.n * periods
            unit = "segundo" if value == 1 else "segundos"
        case _:
            raise ValueError(
                "Only seconds, minutes, hours, days, and weeks are supported."
            )


    return f"{value} {unit}"

# endregion Aux Functions ------------------------------------------------------

# region Preview Functions -----------------------------------------------------

def print_table_section(
    title: str, 
    content: pd.DataFrame | dict | list, 
    show_table_headers: bool = True,
    show_table_index: bool = True,
    header_color: Color = DEFAULT_COLOR, 
    header_align: Alignment = Alignment.LEFT, 
    footer: None | str = None
) -> None:
    """
    Display a formatted HTML table section.

    The function accepts a pandas DataFrame, a dictionary, or a list of
    records and converts the input into an HTML table. Dictionaries with
    list-like values are converted directly into a DataFrame, while other
    dictionaries are displayed as key-value pairs.

    Parameters
    ----------
    title : str
        Section title displayed above the table.

    content : pd.DataFrame, dict, or list
        Tabular content to display.

        - ``pd.DataFrame``: displayed directly as an HTML table.
        - ``dict`` with list values: each key is treated as a column name.
        - ``dict`` with scalar values: displayed as key-value pairs.
        - ``list`` containing lists or dictionaries: converted to a
          DataFrame.

    show_table_headers : bool, default=True
        Whether to display the table column headers.

    show_table_index : bool, default=True
        Whether to display the DataFrame index.

    header_color : Color, default=Color.BLACK
        Color used for the section header.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str, optional
        Text displayed below the table.

    Raises
    ------
    TypeError
        If ``content`` is not a supported type or a list contains elements
        that cannot be converted into a table.

    Notes
    -----
    Empty content is displayed as a message instead of an empty HTML table.

    The function displays the generated HTML directly and returns ``None``.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "18px"
    text_font_size = "14px"
    content_margin_top = "8px"

    # Generar tabla HTML según tipo de contenido
    if len(content) == 0:
            table_html = "La tabla no tiene ningun elemento"

    elif isinstance(content, pd.DataFrame):
        # Mostrar índice si tiene nombre, pero dejar ocultar cabeceras horizontal
        table_html = content.to_html(index=show_table_index, header=show_table_headers)
    
    elif isinstance(content, dict):
        is_dict_of_lists = all(
            isinstance(value, list)
            for value in content.values()
        )
        
        if is_dict_of_lists:
            # Para clave como cabecera y arrays del MISMO tamaño como valor, que seran las filas
            df = pd.DataFrame(content)
        else:
            # Para clave valor simple asignamos las cabeceras de las columnas
            df = pd.DataFrame(list(content.items()), columns=["Key", "Value"])

        table_html = df.to_html(index=False, header=show_table_headers)

    elif isinstance(content, list):
        is_list_of_list = all(
            isinstance(item, (list, dict))
            for item in content
        )

        if is_list_of_list:
            df = pd.DataFrame(content)
            table_html = df.to_html(index=False, header=show_table_headers)
        else:
            raise TypeError("Content list should be a list of lists or list of dict")

    else:
        raise TypeError("Content must be a pandas DataFrame, dict or list of (lists or dict)")

    # Bloque del titulo
    if title:
        html_title = f"""
            <h2 style='text-align:{header_align.value}; color:{header_color.value}; 
                    font-size:{header_font_size}; margin:0;'>{title}</h2>
            <hr style='border:1px solid {header_color.value}; margin:5px 10px 10px 0;'>
        """
    else:
        html_title = ""

    # Bloque del pie de pagina
    if footer:
        html_footer= f"""
            <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{footer}</div>
        """
    else:
        html_footer = ""

    # Bloque completo
    html = f"""
    <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
        {html_title}
        <div style="margin-top:{content_margin_top}; font-size:{text_font_size};">
            {table_html}
        </div>
        {html_footer}
    </div>
    """
    display(HTML(html))


def print_list_section(
    title: str, 
    items: list[str], 
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    numbered: bool = False,
    footer: None | str = None
) -> None:
    """
    Display a formatted HTML list section.

    The function displays a list of items inside a styled HTML block.
    Items can be rendered as either an ordered or unordered list.

    Parameters
    ----------
    title : str
        Section title displayed above the list.

    items : list[str]
        Items to display in the list.

    header_color : Color, default=Color.BLACK
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    numbered : bool, default=False
        Whether to display the items as a numbered list. If ``False``,
        an unordered list with bullet points is displayed.

    footer : str, optional
        Text displayed below the list.

    Notes
    -----
    The function displays the generated HTML directly and returns ``None``.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "18px"
    text_font_size = "14px"
    content_margin_top = "14px"

    # Convertir lista a HTML
    list_html = "".join(f"<li>{item}</li>" for item in items)

    if numbered:
        list_tag = (
            f"<ol style='font-family:monospace; "
            f"white-space:pre-wrap; margin:8px 0 0 0; "
            f"padding-left:30px; font-size:{text_font_size};'>"
            f"{list_html}</ol>"
        )
    else:
        list_tag = (
            f"<ul style='font-family:monospace; "
            f"white-space:pre-wrap; margin:{content_margin_top} 0 0 0; "
            f"padding-left:30px; font-size:{text_font_size};'>"
            f"{list_html}</ul>"
        )

    # Bloque del título
    if title:
        html_title = f"""
            <h3 style="text-align:{header_align.value}; font-size:{header_font_size}; margin:0; color:{header_color.value};">{title}</h3>
            <hr style="border:1px solid {header_color.value}; margin:5px 0;">
        """
    else:
        html_title = ""

    # Bloque del pie de pagina
    if footer:
        html_footer= f"""
            <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{footer}</div>
        """
    else:
        html_footer = ""

    # Bloque completo
    html = f"""
    <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
        {html_title}
        {list_tag}
        {html_footer}
    </div>
    """
    display(HTML(html))


def print_dict_section(
    title: str, 
    content: dict, 
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    footer: None | str = None
) -> None:
    """
    Display a dictionary as a formatted section.

    The dictionary is displayed as two aligned columns, with keys on the
    left and values on the right.

    Parameters
    ----------
    title : str
        Section title displayed above the content.

    content : dict
        Dictionary containing the key-value pairs to display.

    header_color : Color, default=DEFAULT_COLOR
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str or None, default=None
        Optional text displayed below the content.
    """

    if not isinstance(content, dict):
        raise TypeError("Content must be a dict")
    
    # Convertimos todos los valores a string
    content_str = {k: str(v) if not isinstance(v, str) else v for k, v in content.items()}

    content_text = pd.Series(content_str).to_string()

    print_section(
        title = title, 
        content = content_text,
        header_color = header_color,
        header_align = header_align,
        footer = footer
    )


def print_section(
    title: str, 
    content: str, 
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    footer: None | str = None
) -> None:
    """
    Display a formatted HTML section.

    The function displays a styled section containing a title, text content,
    and an optional footer. The content is inserted directly into the
    generated HTML.

    Parameters
    ----------
    title : str
        Section title displayed above the content.

    content : str
        Text or HTML content displayed inside the section.

    header_color : Color, default=Color.BLACK
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str, optional
        Text displayed below the content.

    Notes
    -----
    The value of ``content`` is inserted directly into the generated HTML.
    HTML markup contained in ``content`` is therefore rendered as HTML.

    The function displays the generated HTML directly and returns ``None``.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "18px"
    text_font_size = "14px"
    content_margin_top = "14px"

    # Bloque del título
    if title:
        html_title = f"""
            <h2 style='text-align:{header_align.value}; color:{header_color.value}; 
                    font-size:{header_font_size}; margin:0;'>{title}</h2>
            <hr style='border:1px solid {header_color.value}; margin:5px 10px 10px 0;'>
        """
    else:
        html_title = ""

    # Bloque del pie de pagina
    if footer:
        html_footer= f"""
            <div style="font-family:monospace; white-space:pre-wrap; 
                    font-size:{text_font_size}; margin-top:{content_margin_top};">{footer}</div>
        """
    else:
        html_footer = ""

    # Bloque completo
    html = f"""
    <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
        {html_title}
        <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{content}</div>
        {html_footer}
    </div>
    """

    display(HTML(html))


def print_console_section(
    title: str, 
    content: str,
    separator_length: int = 80
) -> None:
    """
    Display a formatted section in the console using plain text.

    The section consists of a title surrounded by separator lines,
    followed by the provided content.

    Parameters
    ----------
    title : str
        Section title displayed in uppercase.

    content : str
        Text or results to display below the title.

    separator_length : int, default=80
        Number of characters used for the separator lines.
    """

    separator = "=" * separator_length
    print(separator)
    print(str(title).upper())
    print(separator)
    print(content)


def preview_dataframe(
    title: str, 
    df: pd.DataFrame, 
    rows: int = 5,
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
) -> None:
    """
    Display a styled preview of a DataFrame with a formatted title.

    Parameters
    ----------
    title : str
        Title displayed above the DataFrame.

    df : pd.DataFrame
        DataFrame to preview.

    rows : int, default=5
        Number of rows to display.

    header_color : Color, default=Color.BLACK
        Color used for the title and bottom border.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the title.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "24px"

    caption_style = (
        f"<h2 style='text-align:{header_align.value}; color:{header_color.value}; font-size:{header_font_size}; "
        f"border-bottom: 2px solid {header_color.value}; padding-bottom: 5px;'>"
        f"{title}</h2>"
    )

    styled_df = df.head(rows).style.set_caption(caption_style) # Necesario para generar/renderizar HTML en pandas
    display(styled_df)


def show_df_details(
    df: pd.DataFrame, 
    col_time: str, 
    full_info: bool = False, 
    resume_info: bool = False, 
    freq_missing: None | str = None, 
    missing_gap_limit: None | int =None
) -> None:
    """
    Display a summary of a DataFrame with optional detailed information.

    The summary includes the number of invalid cells, invalid rows, and
    constant columns. Optional timestamp analysis can report missing
    periods and large gaps, while additional descriptive information can
    be displayed when requested.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.

    col_time : str
        Name of the timestamp column used for missing-period and gap analysis.

    full_info : bool, default=False
        Whether to display the column summary. If ``freq_missing`` and
        ``missing_gap_limit`` are provided, large timestamp gaps are also
        displayed.

    resume_info : bool, default=False
        Whether to display descriptive statistics and raw DataFrame
        information.

    freq_missing : str, optional
        Pandas offset alias used to detect missing timestamps, such as
        ``"30s"``, ``"5min"``, ``"1h"``, or ``"1D"``.

    missing_gap_limit : int, optional
        Minimum number of consecutive missing periods required for a gap
        to be reported.

    Raises
    ------
    ValueError
        If ``freq_missing`` is not a valid pandas offset alias.

    Notes
    -----
    The preprocessing utilities are imported lazily in this function.
    """

    from ds_utils.preprocessing.preprocessing_utilities import (
        count_cells_with_missing, 
        count_rows_with_missing,
        count_constant_columns
    )

    # Tipos no validos: None, NaN, NaT, "", " "
    n_invalid_cells = count_cells_with_missing(df)

    # Tipos no validos: None, NaN, NaT, "", " "
    n_invalid_rows = count_rows_with_missing(df)

    n_constans_col = count_constant_columns(df)


    # Se muestran el numero de filas totales y el numero de filas con valores invalidos en ellas
    content = f"""DataFrame analizado:
- Celdas: {df.size}
- Filas: {df.shape[0]}
- Columnas: {df.shape[1]}
- Total de celdas con valores no válidos: {n_invalid_cells}
- Total de filas con algún valor no válido: {n_invalid_rows}
- Total de columnas con valores constantes: {n_constans_col}"""
    
    if freq_missing:
        # Validar freq
        try:
            to_offset(freq_missing)
        except ValueError as exc:
            raise ValueError(f"Invalid freq '{freq_missing}'") from exc
    
        n_missing, _ = features_utilities.count_missing_timestamps(df, col_time, freq=freq_missing, realign=False)

        content += f"\n- Total Filas con frecuencia de {freq_missing} que faltan: {n_missing}"

    print_section(
        "Resumen de dataframe".upper(),
        content
    )


    if full_info:

        df_columns_summary = features_utilities.get_columns_summary(
            df,
            sort_by=[
                "num_valores_no_validos",
                "num_valores_distintos",
            ],
            ascending=[
                False,
                True,
            ],
        )

        print_table_section(
            "Resumen de Columnas".upper(),
            df_columns_summary,
        )


        if freq_missing and missing_gap_limit:

            gap_duration_str = format_time_gap(freq_missing, missing_gap_limit)

            # Ver gaps que hay
            df_gaps = features_utilities.find_large_gaps(df, col_time, freq=freq_missing, limit=missing_gap_limit) # 30 min
            print_table_section(
                f"Gaps mayores a {gap_duration_str}".upper(), 
                df_gaps, 
                show_table_index=False
            )

    if resume_info:

        print_table_section(
            "Descripción del dataframe".upper(), 
            df.describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        )

        print_console_section("Resumen Raw del dataframe".upper(), "")
        
        print(df.info())



def show_df_differences(
    df_original: pd.DataFrame,
    df_modified: pd.DataFrame
) -> None:
    """
    Display a summary of the structural differences between two DataFrames.

    The function compares the dimensions and column names of the input
    DataFrames and reports the columns that were added or removed.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original DataFrame.

    df_modified : pd.DataFrame
        Modified DataFrame.
    """

    added_columns = sorted(set(df_modified.columns) - set(df_original.columns))
    removed_columns = sorted(set(df_original.columns) - set(df_modified.columns))

    content = f"""DataFrame original:
- Celdas: {df_original.size}
- Filas: {df_original.shape[0]}
- Columnas: {df_original.shape[1]}

DataFrame modificado:
- Celdas: {df_modified.size}
- Filas: {df_modified.shape[0]}
- Columnas: {df_modified.shape[1]}

Cambios entre original y modificado:
- Columnas añadidas ({len(added_columns)}): {added_columns if added_columns else "Ninguna"}
- Columnas eliminadas ({len(removed_columns)}): {removed_columns if removed_columns else "Ninguna"}
"""

    print_section("Diferencias entre dataframes".upper(), content)



def show_noise_summary(
    df: pd.DataFrame,
    tolerances: dict[str, float],
    config: FeatureNamingConfig = FEATURES_CONFIG,
) -> None:
    """
    Display a summary of sensor statistics and detected noise.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sensor data and noise flags.

    tolerances : dict[str, float]
        Mapping of sensor names to their corresponding noise thresholds.

    config : FeatureNamingConfig, default=FEATURES_CONFIG
        Naming configuration used to identify generated noise columns.
    """
    sections = []
    num_analyzed_sensors = 0

    for sensor, threshold in tolerances.items():
        noise_col = f"{sensor}{config.IS_NOISE_SUFFIX}"

        if sensor not in df.columns or noise_col not in df.columns:
            continue

        num_analyzed_sensors += 1

        sections.append(
            f"""- {sensor}
    Media: {df[sensor].mean():.2f}
    Mediana: {df[sensor].median():.2f}
    Mínimo: {df[sensor].min():.2f}
    Máximo: {df[sensor].max():.2f}
    Umbral: > {threshold}
    Picos detectados: {df[noise_col].sum()}
"""
        )

    total_noise = (
        df[config.GLOBAL_NOISE_COLUMN].sum()
        if config.GLOBAL_NOISE_COLUMN in df.columns
        else "N/A"
    )

    content = f"""SENSORES ANALIZADOS:
{''.join(sections)}
RESUMEN:
Total de sensores analizados: {num_analyzed_sensors}
Total de registros analizados: {len(df)}
Total de registros con ruido encontrados: {total_noise}
"""

    print_section(
        "Resultados de ruido analizados".upper(),
        content,
    )


# TODO deprecated
# def show_noise_summary_old(
#     df: pd.DataFrame,
#     tolerances: dict[str, float],
#     noise_suffix: str = "_is_noise",
#     global_noise_column: str = "any_sensor_noise",
# ) -> None:
#     """
#     Display a summary of sensor statistics and detected noise.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the sensor data and noise flags.
#     tolerances : dict[str, float]
#         Mapping of sensor names to their corresponding noise thresholds.
#     noise_suffix : str, default="_is_noise"
#         Suffix used to identify each sensor's noise flag column.
#     global_noise_column : str, default="any_sensor_noise"
#         Name of the column indicating whether any sensor is marked as noise.
#     """

#     sections = []
#     num_added_sensor = 0

#     for sensor, threshold in tolerances.items():
#         noise_col = f"{sensor}{noise_suffix}"

#         if sensor not in df.columns or noise_col not in df.columns:
#             continue

#         num_added_sensor += 1

#         sections.append(
#             f'''- {sensor}
#     Media: {df[sensor].mean():.2f}
#     Mediana: {df[sensor].median():.2f}
#     Mínimo: {df[sensor].min():.2f}
#     Máximo: {df[sensor].max():.2f}
#     Umbral: > {threshold}
#     Picos detectados: {df[noise_col].sum()}
# '''
#         )

#     total_noise = (
#         df[global_noise_column].sum()
#         if global_noise_column in df.columns
#         else "N/A"
#     )

#     content = f'''SENSORES ANALIZADOS:
# {''.join(sections)}
# RESUMEN:
# Total de sensores analizados: {num_added_sensor}
# Total de registros analizados: {len(df)}
# Total de registros con ruido encontrados: {total_noise}
# '''

#     print_section(
#         "Resultados de ruido analizados".upper(),
#         content,
#     )


# endregion Preview Functions --------------------------------------------------

# region Preview Functions with "lazy import" ----------------------------------

# endregion Preview Functions with "lazy import" -------------------------------