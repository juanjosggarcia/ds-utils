from ds_utils.eda.types import Alignment, Color

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from enum import Enum
from IPython.display import display, HTML
from datetime import datetime
# import matplotlib.pyplot as plt

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

# endregion AUX Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

DEFAULT_COLOR = Color.WHITE
"""
the default color, which is important for ensuring consistency with the editor's light and dark themes
"""

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def format_time_gap(freq: str, periods: int) -> str:
    """
    Returns a human-readable representation of a temporal gap.

    Args:
        freq:
            Pandas offset alias representing the sampling frequency
            (e.g., ``"30s"``, ``"5min"``, ``"1h"``, ``"1D"``).
        periods:
            Number of consecutive sampling periods composing the gap.

    Returns:
        Human-readable string representing the total gap duration
        (e.g., ``"30 minutos"``, ``"2 horas"``, ``"1 día"``).

    Raises:
        ValueError:
            If ``freq`` is not a valid pandas fixed-frequency offset alias
            or represents a frequency greater than days (e.g., ``"2W"``, ``"1ME"``).
    """

    # Validar freq
    try:
        offset = to_offset(freq)

        if offset.name not in {"s", "min", "h", "D"}:
            raise ValueError(
                "Only fixed frequencies up to one day are supported ('s', 'min', 'h', 'D')."
            )

        delta = pd.Timedelta(offset)

    except ValueError:
        raise ValueError(f"Invalid freq '{freq}'")

    # total_seconds = int(offset.delta.total_seconds() * periods)
    total_seconds = int(delta.total_seconds() * periods)

    match total_seconds:
        case s if s % 86400 == 0:
            value = s // 86400
            unit = "día" if value == 1 else "días"

        case s if s % 3600 == 0:
            value = s // 3600
            unit = "hora" if value == 1 else "horas"

        case s if s % 60 == 0:
            value = s // 60
            unit = "minuto" if value == 1 else "minutos"

        case _:
            value = total_seconds
            unit = "segundo" if value == 1 else "segundos"

    return f"{value} {unit}"

# endregion Aux Functions ------------------------------------------------------

# region Preview Functions -----------------------------------------------------


def print_table_section(
        title: str, 
        content: pd.DataFrame | dict | list, 
        show_table_headers=True,
        show_table_index=True,
        header_color=DEFAULT_COLOR, 
        header_align=Alignment.LEFT, 
        footer: None | str =None
):
    """
    Displays a styled block with a DataFrame, dictionary or list of (lists or dict) as a table.

    Args:
        title (str): Block title.
        content (pd.DataFrame or dict or list of (lists or dict)): Dictionary (Key/Value) or DataFrame to display.
        show_table_headers (bool): If False and content is dict, hides Key/Value column headers.
        header_color (Color): Title color (Enum).
        header_align (Alignment): Title alignment (Enum).
        footer (None | str): Footer of the block
    """

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
        if isinstance(next(iter(content.values())), list):
            # Para clave como cabecera y arrays del MISMO tamaño como valor, que seran las filas
            df = pd.DataFrame(content)
        else:
            # Para clave valor simple asignamos las cabeceras de las columnas
            df = pd.DataFrame(list(content.items()), columns=["Key", "Value"])

        table_html = df.to_html(index=False, header=show_table_headers)

    elif isinstance(content, list):
        is_list_of_list = all(isinstance(x, (list, dict)) for x in content)
        if is_list_of_list:
            df = pd.DataFrame(content)
            table_html = df.to_html(index=False, header=show_table_headers)
        else:
            raise TypeError("Content list should be a list of lists or list of dict")

    else:
        raise TypeError("Content must be a pandas DataFrame, dict or list of (lists or dict)")

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


# def print_table_section_old(title, data_dict, header_color=DEFAULT_COLOR, header_align=Alignment.LEFT):
#     """
#     Displays a styled block with a title and a dictionary as a 2-column table (Key / Value).

#     Args:
#         title (str): Block title.
#         data_dict (dict): Dictionary to display as table.
#         header_color (Color): Title color (Enum).
#         header_align (Alignment): Title alignment (Enum).
#         title_font_size (str): CSS font size of the title.
#     """

#     header_font_size = "18px"
#     text_font_size = "14px"

#     content_margin_top = "8px"

#     # Convertir diccionario a DataFrame
#     df = pd.DataFrame(list(data_dict.items()), columns=["Key", "Value"])
#     table_html = df.to_html(index=False)

#     # Construir el bloque
#     html = f"""
#     <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
#         <h2 style='text-align:{header_align.value}; color:{header_color.value}; font-size:{header_font_size}; margin:0;'>{title}</h2>
#         <hr style='border:1px solid {header_color.value}; margin:5px 10px 10px 0;'>
#         {table_html}
#     </div>
#     """
#     display(HTML(html))


def print_list_section(title: str, items: list[str] | str, header_color=DEFAULT_COLOR, 
                       header_align=Alignment.LEFT, enumerate=False, footer: None | str = None):
    """
    Displays a styled block with a heading and an enumerated list.

    Args:
        title (str): Heading of the block.
        items (list | str): List of points to enumerate.
        header_color (str): CSS color of the title and separator line.
        header_align (str): Alignment for the title ('left', 'center', 'right').
        enumerate (bool): If True, uses numbered list; if False, uses bullets.
        footer (None | str): Footer of the block
    """

    header_font_size = "18px"
    text_font_size = "14px"

    content_margin_top = "14px"

    # Convertir lista a HTML
    if enumerate:
        list_html = "".join(f"<li>{item}</li>" for item in items)
        list_tag = f"<ol style='font-family:monospace; white-space:pre-wrap; margin:8 0 0 0; padding-left:30px; font-size:{text_font_size};'>{list_html}</ol>"
    else:
        list_html = "".join(f"<li>{item}</li>" for item in items)
        list_tag = f"<ul style='font-family:monospace; white-space:pre-wrap; margin:{content_margin_top} 0 0 0; padding-left:30px; font-size:{text_font_size};'>{list_html}</ul>"

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


def print_simple_table_section(title: str, content: dict, header_color=DEFAULT_COLOR, header_align=Alignment.LEFT):
    """
    Displays a styled block with a dictionary as a simple table.
    Converts all values to strings, including lists.

    Args:
        title (str): Block title.
        content (dict): Dict to be converted into a key-value table row for display.
        header_color (Color): Title color (Enum).
        header_align (Alignment): Title alignment (Enum).
    """

    if not isinstance(content, dict):
        raise TypeError("Content must be a dict")
    
    # Convertimos todos los valores a string
    content_str = {k: str(v) if not isinstance(v, str) else v for k, v in content.items()}

    # Creamos un DataFrame con una sola fila
    df = pd.DataFrame([content_str])

    fila = df.loc[0]  # o df.iloc[0]

    print_section(
        title = title, 
        content = fila.to_string(),
        header_color = header_color,
        header_align = header_align
    )


def print_section(title: str, content: str, header_color=DEFAULT_COLOR, header_align=Alignment.LEFT, footer: None | str = None):
    """
    Displays a styled diagnostic block with a heading and full-width separator line.

    Args:
        title (str): Heading of the diagnostic step.
        content (str): Text or HTML content to display.
        header_color (str): CSS color for the title.
        header_align (str): Alignment for the title ('left', 'center', 'right').
        footer (None | str): Footer of the block
    """

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
            <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{footer}</div>
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

    # if title:
    #     html = f"""
    #     <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
    #         <h3 style="text-align:{header_align.value}; font-size:{header_font_size}; margin:0; color:{header_color.value};">{title}</h3>
    #         <hr style="border:1px solid {header_color.value}; margin:5px 0;">
    #         <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{content}</div>
    #     </div>
    #     """
    # else:
    #     html = f"""
    #     <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
    #         <div style="font-family:monospace; white-space:pre-wrap; font-size:{text_font_size}; margin-top:{content_margin_top};">{content}</div>
    #     </div>
    #     """

    display(HTML(html))


def print_simple_section(title: str, content: str):
    """
    Prints a formatted diagnostic block with separators (Does not use HTML styles).
    
    Args:
        title (str): The heading of the diagnostic step.
        content (str): The data or results to be displayed.
    """

    separator = "=" * 80
    print(separator)
    print(str(title).upper())
    print(separator)
    print(content)
    # print(separator + "\n")


def preview_dataframe(title: str, df: pd.DataFrame, rows=5, header_color=DEFAULT_COLOR, header_align=Alignment.LEFT):
    """
    Displays a styled preview of a DataFrame with a formatted title.

    Args:
        title (str): Title displayed above the table.
        df (pd.DataFrame): Input dataframe.
        rows (int): Number of rows to show (default 5).
        header_color (Color): Title color (Enum).
        header_align (Alignment): Title alignment (Enum).
    """

    header_font_size = "24px"

    caption_style = (
        f"<h2 style='text-align:{header_align.value}; color:{header_color.value}; font-size:{header_font_size}; "
        f"border-bottom: 2px solid {header_color.value}; padding-bottom: 5px;'>"
        f"{title}</h2>"
    )
    styled_df = df.head(rows).style.set_caption(caption_style)
    display(styled_df)

# def preview_dataframe(df, title, rows=5, header_color=DEFAULT_COLOR, header_align=Alignment.LEFT):
#     """
#     Displays a styled preview of a DataFrame inside a formatted block with a title.
#     """

#     # Convertimos solo las primeras 'rows' filas a HTML
#     table_html = df.head(rows).to_html(index=False)

#     # Creamos el bloque completo
#     html = f"""
#     <div style="border:2px solid #444; padding:10px; margin:10px 0; border-radius:5px;">
#         <h2 style='text-align:{header_align.value}; color:{header_color.value}; 
#                    margin:0; font-size:20px;'>{title}</h2>
#         <hr style='border:1px solid {header_color.value}; margin:5px 10px 10px 0;'>
#         {table_html}
#     </div>
#     """
#     display(HTML(html))

# def preview_dataframe_old(df, title, rows=5, header_color="black"):
#     """
#     Displays a styled preview of the dataframe with an HTML caption.
#     Useful for clean documentation within notebooks.
#     """

#     COLOR_PALETTE = {
#         "black": "#000000",     # Black
#         "white": "#ffffff",     # White
#         "orange": "#e64a19",    # Deep Orange
#         "blue": "#1976d2",      # Blue
#         "purple": "#7b1fa2",    # Purple
#         "yellow": "#fbc02d",    # Amber/Yellow
#         "green": "#2e7d32",     # Forest Green (Default)
#         "red": "#d32f2f",       # Red
#         "pink": "#CA5CC1",      # Pink
#         "gray": "#4e4e4e"       # Gray       
#     }

#     header_color = COLOR_PALETTE.get(header_color, "black")

#     ALIGNMENT = ["left", "center", "right"]

#     # Using <h2> instead of <h1> for better integration with notebook UI
#     caption_style = f"<h2 style='text-align:{ALIGNMENT[1]}; color:{header_color};'>{title}</h2>"
#     display(df.head(rows).style.set_caption(caption_style))


def show_df_details(
    df: pd.DataFrame, 
    col_time: str, 
    full_info: bool = False, 
    resume_info: bool = False, 
    freq_missing: None|str = None, 
    missing_gap_limit: None|int =None
) -> None:
    """
    Displays a summary of a DataFrame including missing values,
    constant columns and optional descriptive information.

    Args:
        df:
            DataFrame to analyse.
        col_time:
            Name of the timestamp column.
        full_info:
            Whether to display unique values per column and detected gaps.
        resume_info:
            Whether to display ``describe()`` and ``info()``.
        freq_missing:
            Pandas offset alias for resampling frequency, (e.g., ``"30s"``, ``"5min"``, ``"1h"``, ``"1D"``, ``"1W"``).
            used to detect missing timestamps.
        missing_gap_limit:
            Minimum number of consecutive missing periods required for a gap
            to be reported.

    Raises:
        ValueError:
            If ``freq_missing`` is not a valid pandas offset alias.
    """

    # from data_analysis.utils import features_utilities
    from ds_utils.preprocessing import features_utilities

    # Tipos no validos: None, NaN, NaT, "", " "
    n_invalid_cells = features_utilities.count_cells_with_missing(df)

    # Tipos no validos: None, NaN, NaT, "", " "
    n_invalid_rows = features_utilities.count_rows_with_missing(df)

    n_constans_col = features_utilities.count_constant_columns(df)


    # Se muestran el numero de filas totales y el numero de filas con valores invalidos en ellas
#     content = f'''Dataframe analizado: celdas({df.size}), filas{df.shape}columnas
# - Total de Celdas con valores no validos en ellas: {n_invalid_cells}
# - Total de Filas con algun valor no valido en ellas: {n_invalid_rows}
# - Total de Columnas con valores unicos en ellas: {n_constans_col}
# '''

    content = f"""DataFrame analizado:
- Celdas: {df.size}
- Filas: {df.shape[0]}
- Columnas: {df.shape[1]}
- Total de celdas con valores no válidos: {n_invalid_cells}
- Total de filas con algún valor no válido: {n_invalid_rows}
- Total de columnas con valores constantes: {n_constans_col}"""

#     content = f"""DataFrame analizado: Celdas: {df.size}; Columnas: {df.shape[1]}; Filas: {df.shape[0]}
#  - Total de celdas con valores no válidos: {n_invalid_cells}
#  - Total de filas con algún valor no válido: {n_invalid_rows}
#  - Total de columnas con valores constantes: {n_constans_col}"""
    
    if freq_missing:
        # Validar freq
        try:
            offset = to_offset(freq_missing)
        except ValueError:
            raise ValueError(f"Invalid freq '{freq_missing}'")
    
        n_missing, _ = features_utilities.count_missing_timestamps(df, col_time, freq=freq_missing, realign=False)

        content += f"\n- Total Filas con frecuencia de {freq_missing} que faltan: {n_missing}"

    print_section(
        "Resumen de dataframe".upper(),
        content
    )


    if full_info:

        # # Ver numero de valores distintos por columna
        # df_nunique = (
        #     df.nunique(dropna=True)
        #     .sort_values()
        #     .reset_index(name="num_valores_distintos") # reset_index ya fuerza que la salida sea un dataframe no necesito añadir .to_frame()
        #     .rename(columns={"index": "nombre_columna"})
        # )
        # print_table_section(
        #     f"Total valores distintos por columnas".upper(), 
        #     df_nunique, 
        # )

        # Resumen de columnas

        # invalid_values = (
        #     df.isna().sum()
        #     + (
        #         df.select_dtypes(include="object")
        #         .apply(lambda col: col.str.strip().eq("").sum())
        #     )
        # )

        # invalid_values = features_utilities.count_missing_cells_by_column(df)

        # df_columns_summary = pd.DataFrame({
        #     "nombre_columna": df.columns,
        #     "tipo_dato": df.dtypes.astype(str).values,
        #     "num_valores_distintos": df.nunique(dropna=True).values,
        #     "num_valores_no_validos": invalid_values,
        # })

        # df_columns_summary["porcentaje_no_validos"] = (
        #     df_columns_summary["num_valores_no_validos"] / len(df) * 100
        # ).round(2)

        # df_columns_summary = (
        #     df_columns_summary
        #     .sort_values(
        #         by=[
        #             "num_valores_no_validos",
        #             "num_valores_distintos",
        #         ],
        #         ascending=[
        #             False,
        #             True,
        #         ],
        #     )
        #     .reset_index(drop=True)
        # )


        # invalid_values = features_utilities.count_missing_cells_by_column(df)

        # df_columns_summary = pd.DataFrame(index=df.columns)

        # df_columns_summary["tipo_dato"] = df.dtypes.astype(str)
        # df_columns_summary["num_valores_distintos"] = df.nunique(dropna=True)
        # df_columns_summary["num_valores_no_validos"] = invalid_values

        # df_columns_summary["porcentaje_no_validos"] = (
        #     df_columns_summary["num_valores_no_validos"] / len(df) * 100
        # ).round(2)

        # df_columns_summary = (
        #     df_columns_summary
        #     .reset_index(names="nombre_columna")
        #     .sort_values(
        #         by=[
        #             "num_valores_no_validos",
        #             "num_valores_distintos",
        #         ],
        #         ascending=[
        #             False,
        #             True,
        #         ],
        #     )
        #     .reset_index(drop=True)
        # )

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

        # print_table_section("Descripción del dataframe".upper(), df.describe())
        print_table_section(
            "Descripción del dataframe".upper(), 
            df.describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        )

        print_simple_section("Resumen Raw del dataframe".upper(), "")
        
        print(df.info())



def show_df_differences(
    df_original: pd.DataFrame,
    df_modified: pd.DataFrame
) -> None:
    """
    Displays a summary of the structural differences between two DataFrames.

    The function compares the input DataFrames and reports their dimensions,
    as well as the columns that have been added or removed.

    Args:
        df_original:
            Original DataFrame.

        df_modified:
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
    
#     content = f'''- Dataset original: celdas({df_before.size}), filas{df_before.shape}columnas
# - Dataset Despues del procesamiento: celdas({df_after.size}), filas{df_after.shape}columnas
# - Columnas Añadidas: {added_columns}
# - Columnas Eliminadas: {removed_columns}
# '''

    print_section("Diferencias entre dataframes".upper(), content)




def show_noise_summary(
    df: pd.DataFrame,
    tolerances: dict[str, float],
    noise_suffix: str = "_is_noise",
    global_noise_column: str = "any_sensor_noise",
) -> None:
    """
    Display a summary of sensor statistics and detected noise.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the sensor data and noise flags.
    tolerances : dict[str, float]
        Mapping of sensor names to their corresponding noise thresholds.
    noise_suffix : str, default="_is_noise"
        Suffix used to identify each sensor's noise flag column.
    global_noise_column : str, default="any_sensor_noise"
        Name of the column indicating whether any sensor is marked as noise.
    """

    sections = []
    num_added_sensor = 0

    for sensor, threshold in tolerances.items():
        noise_col = f"{sensor}{noise_suffix}"

        if sensor not in df.columns or noise_col not in df.columns:
            continue

        num_added_sensor += 1

        sections.append(
            f'''- {sensor}
    Media: {df[sensor].mean():.2f}
    Mediana: {df[sensor].median():.2f}
    Mínimo: {df[sensor].min():.2f}
    Máximo: {df[sensor].max():.2f}
    Umbral: > {threshold}
    Picos detectados: {df[noise_col].sum()}
'''
        )

    total_noise = (
        df[global_noise_column].sum()
        if global_noise_column in df.columns
        else "N/A"
    )

    content = f'''SENSORES ANALIZADOS:
{''.join(sections)}
RESUMEN:
Total de sensores analizados: {num_added_sensor}
Total de registros analizados: {len(df)}
Total de registros con ruido encontrados: {total_noise}
'''

    print_section(
        "Resultados de ruido analizados".upper(),
        content,
    )



# endregion Preview Functions --------------------------------------------------

# region Preview Functions with "lazy import" ----------------------------------

# endregion Preview Functions with "lazy import" -------------------------------