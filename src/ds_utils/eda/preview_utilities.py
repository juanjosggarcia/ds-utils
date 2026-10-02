from ds_utils.eda.preview_config import (
    Alignment,
    Color,
    DEFAULT_COLOR,
)
from ds_utils.preprocessing.preprocessing_config import (
    FEATURES_CONFIG,
    FeatureNamingConfig,
)
from ds_utils.preprocessing import preprocessing_utilities

import numpy as np
import pandas as pd
from typing import Literal
from collections.abc import Iterable
from html import escape as html_escape
from tabulate import tabulate
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Week, Day, Hour, Minute, Second
from IPython.display import display, HTML, Math, Markdown, Latex

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

# region HTML Functions --------------------------------------------------------

def print_html_table(
    content: pd.DataFrame | dict | list,
    header: str | None = None,
    show_table_headers: bool = True,
    show_table_index: bool = True,
    header_color: Color = DEFAULT_COLOR, 
    header_align: Alignment = Alignment.LEFT, 
    footer: str | None = None,
    escape_html: bool = True
) -> None:
    """
    Display a formatted HTML table section.

    The function accepts a pandas DataFrame, a dictionary, or a list of
    records and converts the input into an HTML table. Dictionaries with
    list-like values are converted directly into a DataFrame, while other
    dictionaries are displayed as key-value pairs.

    Parameters
    ----------
    content : pd.DataFrame, dict, or list
        Tabular content to display.

        - ``pd.DataFrame``: displayed directly as an HTML table.
        - ``dict`` with list values: each key is treated as a column name.
        - ``dict`` with scalar values: displayed as key-value pairs.
        - ``list`` containing lists or dictionaries: converted to a
          DataFrame.
    
    header : str or None, default=None
        Optional section title displayed above the table.

    show_table_headers : bool, default=True
        Whether to display the table column headers.

    show_table_index : bool, default=True
        Whether to display the DataFrame index.

    header_color : Color, default=Color.BLACK
        Color used for the section header.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str or None, default=None
        Optional text displayed below the content.

    escape_html : bool, default=True
        Whether to escape HTML markup in ``header`` and ``footer``.
        If ``True``, HTML characters are escaped and rendered as plain text.
        If ``False``, the values are inserted directly into the generated
        HTML and any valid HTML markup is rendered. Disabling this option
        should only be done for trusted content.

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

    if escape_html:
        header = html_escape(header) if header else header
        footer = html_escape(footer) if footer else footer

    # Generar tabla HTML según tipo de contenido
    if len(content) == 0:
            table_html = "La tabla no tiene ningún elemento"

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
    if header:
        html_title = f"""
            <h2 style='text-align:{header_align.value}; color:{header_color.value}; 
                    font-size:{header_font_size}; margin:0;'>{header}</h2>
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


def print_html_list(
    items: Iterable[str],
    numbered: bool = False,
    header: str | None = None,
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    footer: str | None = None,
    escape_html: bool = True
) -> None:
    """
    Display a formatted HTML list section.

    The function displays a list of items inside a styled HTML block.
    Items can be rendered as either an ordered or unordered list.

    Parameters
    ----------
    items : Iterable[str]
        Items to display in the list.

    numbered : bool, default=False
        Whether to display the items as a numbered list. If ``False``,
        an unordered list with bullet points is displayed.

    header : str or None, default=None
        Optional section title displayed above the list.

    header_color : Color, default=Color.BLACK
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str or None, default=None
        Optional text displayed below the content.

    escape_html : bool, default=True
        Whether to escape HTML markup in ``header`` and ``footer``.
        If ``True``, HTML characters are escaped and rendered as plain text.
        If ``False``, the values are inserted directly into the generated
        HTML and any valid HTML markup is rendered. Disabling this option
        should only be done for trusted content.

    Notes
    -----
    The function displays the generated HTML directly and returns ``None``.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "18px"
    text_font_size = "14px"
    content_margin_top = "14px"

    if escape_html:
        header = html_escape(header) if header else header
        footer = html_escape(footer) if footer else footer

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
    if header:
        html_title = f"""
            <h3 style="text-align:{header_align.value}; font-size:{header_font_size}; margin:0; color:{header_color.value};">{header}</h3>
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


def print_html_dict(
    content: dict,
    header: str | None = None,
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    footer: str | None = None,
    escape_html: bool = True
) -> None:
    """
    Display a dictionary as a formatted section.

    The dictionary is displayed as two aligned columns, with keys on the
    left and values on the right.

    Parameters
    ----------
    content : dict
        Dictionary containing the key-value pairs to display.
    
    header : str or None, default=None
        Optional section title displayed above the content.

    header_color : Color, default=DEFAULT_COLOR
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str or None, default=None
        Optional text displayed below the content.

    escape_html : bool, default=True
        Whether to escape HTML markup in ``header``, ``content`` and ``footer``.
        If ``True``, HTML characters are escaped and rendered as plain text.
        If ``False``, the values are inserted directly into the generated
        HTML and any valid HTML markup is rendered. Disabling this option
        should only be done for trusted content.
    """

    if not isinstance(content, dict):
        raise TypeError("Content must be a dict")
    
    # Convertimos todos los valores a string
    content_str = {k: str(v) if not isinstance(v, str) else v for k, v in content.items()}

    content_text = pd.Series(content_str).to_string()

    print_html_section(
        header = header, 
        content = content_text,
        header_color = header_color,
        header_align = header_align,
        footer = footer,
        escape_html = escape_html
    )


def print_html_section(
    content: str,
    header: str | None = None,
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    footer: str | None = None,
    escape_html: bool = True
) -> None:
    """
    Display a formatted HTML section.

    The function displays a styled section containing a header, text content,
    and an optional footer. The content is inserted directly into the
    generated HTML.

    Parameters
    ----------
    content : str
        Text or HTML content displayed inside the section.

    header : str or None, default=None
        Optional section title displayed above the content.

    header_color : Color, default=Color.BLACK
        Color used for the section header and separator line.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the section header.

    footer : str or None, default=None
        Optional text displayed below the content.

    escape_html : bool, default=True
        Whether to escape HTML markup in ``header``, ``content`` and ``footer``.
        If ``True``, HTML characters are escaped and rendered as plain text.
        If ``False``, the values are inserted directly into the generated
        HTML and any valid HTML markup is rendered. Disabling this option
        should only be done for trusted content.

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

    if escape_html:
        header = html_escape(header) if header else header
        content = html_escape(content)
        footer = html_escape(footer) if footer else footer

    # Bloque del título
    if header:
        html_title = f"""
            <h2 style='text-align:{header_align.value}; color:{header_color.value}; 
                    font-size:{header_font_size}; margin:0;'>{header}</h2>
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

# endregion HTML Functions -----------------------------------------------------

# region Markdown Functions ----------------------------------------------------

def print_markdown_table(
    content: pd.DataFrame | dict | list,
    header: str | None = None,
    header_separator: bool = True,
    show_table_index: bool = True,
    footer: str | None = None,
) -> None:
    """
    Display a formatted Markdown table.

    The function accepts a pandas DataFrame, a dictionary, or a list of
    records and converts the input into a Markdown table. Dictionaries
    with list-like values are converted directly into a DataFrame, while
    dictionaries with scalar values are displayed as key-value pairs.

    Parameters
    ----------
    content : pd.DataFrame, dict, or list
        Tabular content to display.

        - ``pd.DataFrame``: displayed directly as a Markdown table.
        - ``dict`` with list values: each key is treated as a column name.
        - ``dict`` with scalar values: displayed as key-value pairs.
        - ``list`` containing lists or dictionaries: converted to a
          DataFrame.

    header : str or None, default=None
        Optional section header displayed above the table.
        Markdown syntax is supported.

    header_separator : bool, default=True
        Whether to display a horizontal separator below the header.

    show_table_index : bool, default=True
        Whether to display the DataFrame index.

    footer : str or None, default=None
        Optional footer displayed below the table as a Markdown block quote.
        Markdown syntax is supported. Multi-line notes are supported, with
        each line rendered as part of the block quote.

    Raises
    ------
    TypeError
        If ``content`` is not a supported type or a list contains elements
        that cannot be converted into a table.

    Notes
    -----
    Empty content is displayed as a message instead of an empty table.

    The table column headers are always displayed, as they are required by
    the Markdown table syntax.

    The function displays the generated Markdown directly and returns
    ``None``.
    """

    if len(content) == 0:
        table_markdown = "La tabla no tiene ningún elemento"

    elif isinstance(content, pd.DataFrame):
        table_markdown = content.to_markdown(
            index=show_table_index,
            tablefmt="pipe",
        )

    elif isinstance(content, dict):
        is_dict_of_lists = all(
            isinstance(value, list)
            for value in content.values()
        )

        if is_dict_of_lists:
            df = pd.DataFrame(content)
        else:
            df = pd.DataFrame(
                list(content.items()),
                columns=["Key", "Value"],
            )

        table_markdown = df.to_markdown(
            index=False,
            tablefmt="pipe",
        )

    elif isinstance(content, list):
        if all(isinstance(item, (list, dict)) for item in content):
            df = pd.DataFrame(content)

            table_markdown = df.to_markdown(
                index=False,
                tablefmt="pipe",
            )
        else:
            raise TypeError(
                "Content list should be a list of lists or list of dict"
            )

    else:
        raise TypeError(
            "Content must be a pandas DataFrame, dict or list of "
            "(lists or dict)"
        )

    markdown = ""

    if header:
        markdown += f"# {header}\n\n"

        if header_separator:
            markdown += "---\n\n<br>\n\n"

    markdown += table_markdown

    if footer:
        quoted_note = "\n".join(
            f"> {line}" if line else ">"
            for line in footer.splitlines()
        )

        markdown += f"\n\n<br>\n\n{quoted_note}"

    display(Markdown(markdown))


def print_markdown_list(
    items: Iterable[str],
    numbered: bool = False,
    header: str | None = None,
    header_separator: bool = True,
    footer: str | None = None,
) -> None:
    """
    Display a formatted Markdown list section.

    The function displays a list of items as either an ordered or unordered
    Markdown list, with an optional header, separator, and footer.

    Parameters
    ----------
    items : Iterable[str]
        Items to display in the list.

    numbered : bool, default=False
        Whether to display the items as a numbered list. If ``False``,
        an unordered list with bullet points is displayed.

    header : str or None, default=None
        Optional section title displayed above the list.
        Markdown syntax is supported.

    header_separator : bool, default=True
        Whether to display a horizontal separator below the header.

    footer : str or None, default=None
        Optional footer displayed below the list as a Markdown block quote.
        Markdown syntax is supported. Multi-line notes are supported, with
        each line rendered as part of the block quote.

    Notes
    -----
    The function displays the generated Markdown directly and returns
    ``None``.
    """
    markdown = ""

    if header:
        markdown += f"# {header}\n\n"

        if header_separator:
            markdown += "---\n\n"

    if numbered:
        markdown += "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(items, start=1)
        )
    else:
        markdown += "\n".join(
            f"- {item}"
            for item in items
        )

    if footer:
        quoted_note = "\n".join(
            f"> {line}" if line else ">"
            for line in footer.splitlines()
        )

        markdown += f"\n\n{quoted_note}"

    display(Markdown(markdown))


def print_markdown_section(
    content: str,
    header: str | None = None,
    header_separator: bool = True,
    footer: str | None = None,
) -> None:
    """
    Display a formatted Markdown section with an optional header and footer.

    The section consists of an optional title, a horizontal separator,
    the provided Markdown content, and an optional footer displayed as
    a Markdown block quote.

    Parameters
    ----------
    content : str
        Markdown content displayed inside the section.

    header : str or None, default=None
        Optional section title displayed above the separator.

    header_separator : bool, default=True
        Whether to display a horizontal separator below the header.

    footer : str or None, default=None
        Optional footer displayed below the mathematical expression as a
        Markdown block quote. Markdown syntax is supported. Multi-line notes
        are supported, with each line rendered as part of the block quote.

    Returns
    -------
    None
        The generated Markdown is displayed directly.
    """
    markdown = ""

    if header:
        markdown += f"# {header}\n\n"

    if header_separator:
        markdown += "---\n\n"

    markdown += content

    if footer:
        quoted_note = "\n".join(
            f"> {line}" if line else ">"
            for line in footer.splitlines()
        )

        markdown += f"\n\n{quoted_note}"

    display(Markdown(markdown))

# endregion Markdown Functions -------------------------------------------------

# region Console Functions -----------------------------------------------------

def print_console_table(
    content: pd.DataFrame | dict | list,
    header: str | None = None,
    show_table_index: bool = True,
    footer: str | None = None,
    separator_length: int = 80,
    tablefmt: Literal[
        "simple",
        "plain",
        "grid",
        "fancy_grid",
        "outline",
        "simple_outline",
        "rounded_outline",
        "heavy_outline",
        "mixed_outline",
        "double_outline",
        "psql",
        "jira",
        "presto",
        "pretty",
        "rst",
        "mediawiki",
        "orgtbl",
        "double_grid",
        "mixed_grid",
    ] = "fancy_grid",
) -> None:
    """
    Display a formatted table in the console using Unicode box-drawing
    characters.

    Parameters
    ----------
    content : pd.DataFrame, dict, or list
        Tabular content to display.

        - ``pd.DataFrame``: displayed directly.
        - ``dict`` with list values: each key is treated as a column name.
        - ``dict`` with scalar values: displayed as key-value pairs.
        - ``list`` containing lists or dictionaries: converted into a table.

    header : str or None, default=None
        Optional section header displayed above the table in uppercase.
        

    show_table_index : bool, default=True
        Whether to display the DataFrame index.

    footer : str or None, default=None
        Optional footer displayed below the table.

    separator_length : int, default=80
        Number of characters used for the separator lines.

    tablefmt : {"simple","plain","grid","fancy_grid","outline","simple_outline",
    "rounded_outline","heavy_outline","mixed_outline","double_outline","psql",
    "jira","presto","pretty","rst","mediawiki","orgtbl","double_grid",
    "mixed_grid"}, default="fancy_grid"
        Format used to render the table in the console.

    Raises
    ------
    TypeError
        If ``content`` is not a supported type or a list contains elements
        that cannot be converted into a table.

    Notes
    -----
    Empty content is displayed as a message instead of an empty table.

    The table is rendered using the ``fancy_grid`` format from
    ``tabulate``.

    The function displays the generated table directly and returns
    ``None``.
    """

    valid_table_formats = {
        "simple",
        "plain",
        "grid",
        "fancy_grid",
        "outline",
        "simple_outline",
        "rounded_outline",
        "heavy_outline",
        "mixed_outline",
        "double_outline",
        "psql",
        "jira",
        "presto",
        "pretty",
        "rst",
        "mediawiki",
        "orgtbl",
        "double_grid",
        "mixed_grid",
    }

    if tablefmt not in valid_table_formats:
        raise ValueError(
            f"Invalid table format: {tablefmt!r}. "
            f"Expected one of: {', '.join(sorted(valid_table_formats))}"
        )

    separator = "=" * separator_length
    separator_footer = "_" * separator_length

    if isinstance(content, pd.DataFrame):
        if content.empty:
            print("La tabla no tiene ningún elemento")
            return

        table = tabulate(
            content, # type: ignore[arg-type] # Se ignora porque si puede trabajar internamente con df
            headers="keys",
            tablefmt=tablefmt,
            showindex=show_table_index,
        )

    elif isinstance(content, dict):
        if not content:
            print("La tabla no tiene ningún elemento")
            return

        is_dict_of_lists = all(
            isinstance(value, list)
            for value in content.values()
        )

        if is_dict_of_lists:
            table = tabulate(
                content,
                headers="keys",
                tablefmt=tablefmt,
                showindex=False,
            )
        else:
            table = tabulate(
                content.items(),
                headers=["Key", "Value"],
                tablefmt=tablefmt,
                showindex=False,
            )

    elif isinstance(content, list):
        if not content:
            print("La tabla no tiene ningún elemento")
            return

        if not all(isinstance(item, (list, dict)) for item in content):
            raise TypeError(
                "Content list should be a list of lists or list of dict"
            )

        table = tabulate(
            content,
            headers="keys" if isinstance(content[0], dict) else (),
            tablefmt=tablefmt,
            showindex=False,
        )

    else:
        raise TypeError(
            "Content must be a pandas DataFrame, dict or list of "
            "(lists or dict)"
        )

    if header is not None:
        print(separator)
        print(header.upper())
        print(separator)

    print(table)

    if footer is not None:
        print(separator_footer)
        print(footer)


def print_console_list(
    items: Iterable[str],
    numbered: bool = False,
    header: str | None = None,
    footer: str | None = None,
    separator_length: int = 80
) -> None:
    """
    Display a formatted list section in the console.

    The function displays a list of items as either an ordered or unordered
    text list, with an optional header and footer.

    Parameters
    ----------
    items : Iterable[str]
        Items to display in the list.

    numbered : bool, default=False
        Whether to display the items as a numbered list. If ``False``,
        an unordered list with bullet points is displayed.

    header : str or None, default=None
        Optional section header displayed above the list in uppercase.

    footer : str or None, default=None
        Optional footer displayed below the list.

    separator_length : int, default=80
        Number of characters used for the separator lines.

    Notes
    -----
    The function displays the generated list directly and returns
    ``None``.
    """
    if numbered:
        content = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(items, start=1)
        )
    else:
        content = "\n".join(
            f"• {item}"
            for item in items
        )

    print_console_section(
        content=content,
        header=header,
        footer=footer,
        separator_length=separator_length
    )
        

def print_console_section( 
    content: str,
    header: str | None = None,
    footer: str | None = None,
    separator_length: int = 80
) -> None:
    """
    Display a formatted section in the console using plain text.

    The section consists of a title surrounded by separator lines,
    followed by the provided content and an optional footer.

    Parameters
    ----------
    content : str
        Text or results to display below the title.

    header : str or None, default=None
        Optional section header displayed above the table in uppercase.

    footer : str or None, default=None
        Optional text displayed below the content and a separator line.

    separator_length : int, default=80
        Number of characters used for the separator lines.
    """

    separator = "=" * separator_length
    separator_footer = "_" * separator_length


    if header is not None:
        print(separator)
        print(header.upper())
        print(separator)

    print(content)

    if footer is not None:
        print(separator_footer)
        print(footer)

# endregion Console Functions --------------------------------------------------

# region Math Functions --------------------------------------------------------

def print_math_list_expressions(
    items: Iterable[str],
    header: str | None = None,
    footer: str | None = None,
    header_size: Literal[
        "tiny",
        "small",
        "normalsize",
        "large",
        "Large",
        "LARGE",
        "huge",
        "Huge",
    ] = "huge",
) -> None:
    """
    Display multiple mathematical expressions as a single formatted
    LaTeX block.

    Each item is treated as a single mathematical expression and
    displayed on a separate line inside an ``aligned`` environment.
    Expressions automatically use ``\\displaystyle`` so that fractions,
    sums, integrals, and other display-style constructs retain their
    full mathematical size.

    An optional header is displayed above the expressions using the
    requested LaTeX font size, while an optional footer is displayed
    below them in italic style. The complete section is rendered as a
    single :class:`IPython.display.Math` output.

    Parameters
    ----------
    items : Iterable[str]
        Mathematical expressions written in LaTeX. Each expression is
        displayed on a separate line.

    header : str or None, default=None
        Optional title displayed above the mathematical expressions.

    footer : str or None, default=None
        Optional explanatory text displayed below the expressions in
        italic style.

    header_size : Literal[...], default="huge"
        LaTeX font size used for the header. Available sizes are
        ``"tiny"``, ``"small"``, ``"normalsize"``, ``"large"``,
        ``"Large"``, ``"LARGE"``, ``"huge"``, and ``"Huge"``.

    Notes
    -----
    The expressions are wrapped in a LaTeX ``aligned`` environment.
    The ``&`` character can be used to define a common alignment point
    between rows.

    The expressions are inserted directly into the generated LaTeX;
    no escaping or validation is performed.

    Examples
    --------
    Align several expressions at the equality operator:

        >>> print_math_list(
        ...     [
        ...         r"x^2 + 2x + 1 &= 0",
        ...         r"x^2 + 2x &= -1",
        ...         r"x &= -1",
        ...     ],
        ...     header="Solving the equation",
        ... )
    """

    valid_header_sizes = {"tiny","small","normalsize","large","Large","LARGE",
        "huge","Huge",}

    if header_size not in valid_header_sizes:
        raise ValueError(
            f"Invalid header_size: {header_size!r}. "
            f"Expected one of: {', '.join(sorted(valid_header_sizes))}."
        )

    expressions = list(items)

    content = "\n".join(
        rf"\displaystyle {expression} \\"
        for expression in expressions
    )

    lines: list[str] = [r"\begin{array}{l}"]

    if header is not None:
        lines.extend(
            [
                rf"\{header_size} \text{{{header}}}",
                r"\\[1.5em]",
            ]
        )

    lines.extend(
        [
            r"\begin{aligned}",
            content,
            r"\end{aligned}",
        ]
    )

    if footer is not None:
        lines.extend(
            [
                r"\\[1.5em]",
                rf"\text{{\it {footer}}}",
            ]
        )

    lines.append(r"\end{array}")

    display(Math("\n".join(lines)))


def print_math_expression(
    content: str,
    header: str | None = None,
    footer: str | None = None,
    header_size: Literal[
        "tiny",
        "small",
        "normalsize",
        "large",
        "Large",
        "LARGE",
        "huge",
        "Huge",
    ] = "huge",
) -> None:
    """
    Display a single mathematical expression as a formatted LaTeX block.

    The expression is rendered inside a ``gathered`` environment and
    automatically uses ``\\displaystyle`` so that fractions, sums,
    integrals, and other display-style constructs retain their full
    mathematical size.

    An optional header is displayed above the expression using the
    requested LaTeX font size, while an optional footer is displayed
    below the expression in italic text. The complete section is
    rendered as a single :class:`IPython.display.Math` output.

    This function is intended for displaying a single mathematical
    expression with a consistent presentation layout. For multiple
    expressions or fully custom LaTeX layouts, use the corresponding
    specialized functions instead.

    Parameters
    ----------
    content : str
        Mathematical expression written in LaTeX. The expression is
        inserted directly into the generated ``gathered`` environment;
        no escaping or validation is performed.

    header : str or None, default=None
        Optional title displayed above the mathematical expression.

    footer : str or None, default=None
        Optional explanatory text displayed below the expression in italic style.

    header_size : Literal[...], default="huge"
        LaTeX font size used for the header. Available sizes are
        ``"tiny"``, ``"small"``, ``"normalsize"``, ``"large"``,
        ``"Large"``, ``"LARGE"``, ``"huge"``, and ``"Huge"``.

    Notes
    -----
    The header and footer are treated as plain text within LaTeX
    ``\\text{...}`` commands. They should therefore not contain
    unescaped LaTeX commands or characters with special meaning in
    LaTeX.

    The function produces one ``Math`` display containing the header,
    expression, and footer rather than separate notebook outputs.


    Examples
    --------
    Display a simple mathematical expression:

        >>> print_math_expression(
        ...     r"x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
        ...     header="Quadratic formula",
        ...     footer="The formula gives the solutions of the quadratic equation.",
        ... )
    """

    valid_header_sizes = {"tiny","small","normalsize","large","Large","LARGE",
        "huge","Huge",}

    if header_size not in valid_header_sizes:
        raise ValueError(
            f"Invalid header_size: {header_size!r}. "
            f"Expected one of: {', '.join(sorted(valid_header_sizes))}."
        )

    lines: list[str] = [r"\begin{array}{l}"]

    if header is not None:
        lines.extend(
            [
                rf"\{header_size} \text{{{header}}}",
                r"\\[1.5em]",
            ]
        )

    lines.extend(
        [
            r"\begin{gathered}",
            rf"\displaystyle {content}",
            r"\end{gathered}",
        ]
    )

    if footer is not None:
        lines.extend(
            [
                r"\\[1.5em]",
                rf"\text{{\it {footer}}}",
            ]
        )

    lines.append(r"\end{array}")

    display(Math("\n".join(lines)))


def print_math_section(
    content: str,
    header: str | None = None,
    header_separator: bool = False,
    footer: str | None = None,
) -> None:
    """
    Display raw LaTeX content with optional Markdown header and footer.

    This function provides full control over the mathematical content.
    The value of ``content`` is passed directly to
    :class:`IPython.display.Math` without any escaping, validation, or
    automatic formatting. This allows arbitrary LaTeX structures and
    environments such as ``aligned``, ``gathered``, ``array``, ``cases``,
    and custom combinations of them.

    Unlike :func:`print_math_expression` and :func:`print_math_list`, this
    function does not apply any automatic mathematical layout or font
    sizing to ``content``. It is intended for cases where the caller
    needs direct control over the generated LaTeX.

    Parameters
    ----------
    content : str
        Raw LaTeX content to display. The content is passed directly to
        :class:`IPython.display.Math` and may contain any valid LaTeX
        expression or environment supported by the rendering backend.

    header : str or None, default=None
        Optional title displayed above the mathematical content as a
        Markdown heading. Markdown syntax is supported.

    header_separator : bool, default=False
        Whether to display a horizontal Markdown separator below the
        header.

    footer : str or None, default=None
        Optional footer displayed below the mathematical content as a
        Markdown block quote. Markdown syntax is supported. Multi-line
        footers are supported, with each line rendered as part of the
        block quote.

    Notes
    -----
    No escaping or validation is performed on ``content``. The caller is
    responsible for providing valid LaTeX supported by the rendering
    backend.

    The header, mathematical content, and footer are displayed as
    separate notebook outputs. Unlike the formatted mathematical
    helpers, this function intentionally does not combine them into a
    single ``Math`` block.

    Examples
    --------
    Display a simple mathematical expression:

        >>> print_math_raw(r"x^2 + 2x + 1 = 0")

    Use a custom LaTeX environment:

        >>> print_math_raw(
        ...     r'''
        ...     \\begin{aligned}
        ...     x^2 + 2x + 1 &= 0 \\
        ...     x^2 + 2x &= -1 \\
        ...     x &= -1
        ...     \\end{aligned}
        ...     ''',
        ...     header="Solving the equation",
        ...     header_separator=True,
        ...     footer="Each step is aligned at the equality operator.",
        ... )
    """
    if header is not None:
        markdown = f"# {header}\n\n"

        if header_separator:
            markdown += "---\n\n"

        display(Markdown(markdown))

    display(Math(content))

    if footer:
        quoted_note = "\n".join(
            f"> {line}" if line else ">"
            for line in footer.splitlines()
        )

        display(Markdown(quoted_note))

# endregion Math Functions -----------------------------------------------------

def preview_dataframe(
    df: pd.DataFrame, 
    header: str | None = None, 
    header_color: Color = DEFAULT_COLOR,
    header_align: Alignment = Alignment.LEFT,
    rows: int = 5,
) -> None:
    """
    Display a styled preview of a DataFrame with a formatted title.

    The DataFrame is rendered using pandas ``Styler`` to preserve the
    notebook's rich HTML representation. An optional header can be added
    above the preview.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to preview.

    header : str or None, default=None
        Optional title displayed above the DataFrame.

    header_color : Color, default=Color.BLACK
        Color used for the title and bottom border.

    header_align : Alignment, default=Alignment.LEFT
        Horizontal alignment of the title.

    rows : int, default=5
        Number of rows to display.

    Notes
    -----
    Only the first ``rows`` rows of the DataFrame are displayed. The
    resulting table is rendered using pandas ``Styler`` and is intended
    for interactive inspection in Jupyter notebooks.
    """

    # --- LOCAL VISUAL CONFIGURATION ---
    header_font_size = "24px"

    styled_df = df.head(rows).style

    if header is not None:
        caption_style = (
            f"<h2 style='text-align:{header_align.value}; color:{header_color.value}; font-size:{header_font_size}; "
            f"border-bottom: 2px solid {header_color.value}; padding-bottom: 5px;'>"
            f"{header}</h2>"
        )

        styled_df = styled_df.set_caption(caption_style) # Necesario para generar/renderizar HTML en pandas


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
    
        n_missing, _ = preprocessing_utilities.count_missing_timestamps(df, col_time, freq=freq_missing, realign=False)

        content += f"\n- Total Filas con frecuencia de {freq_missing} que faltan: {n_missing}"

    print_html_section(
        content=content,
        header="Resumen de dataframe".upper(),
    )


    if full_info:

        df_columns_summary = preprocessing_utilities.get_columns_summary(
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

        print_html_table(
            content=df_columns_summary,
            header="Resumen de Columnas".upper(),
        )


        if freq_missing and missing_gap_limit:

            gap_duration_str = format_time_gap(freq_missing, missing_gap_limit)

            # Ver gaps que hay
            df_gaps = preprocessing_utilities.find_large_gaps(df, col_time, freq=freq_missing, limit=missing_gap_limit) # 30 min
            print_html_table( 
                content=df_gaps,
                header=f"Gaps mayores a {gap_duration_str}".upper(),
                show_table_index=False
            )

    if resume_info:

        print_html_table( 
            content=df.describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]),
            header="Descripción del dataframe".upper()
        )

        print_console_section(
            content="",
            header="Resumen Raw del dataframe".upper(),
            separator_length=60
        )
        
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

    print_html_section(
        content=content,
        header="Diferencias entre dataframes".upper(),
    )



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

    print_html_section(
        content=content,
        header="Resultados de ruido analizados".upper(),
    )


# endregion Preview Functions --------------------------------------------------

# region Preview Functions with "lazy import" ----------------------------------

# endregion Preview Functions with "lazy import" -------------------------------