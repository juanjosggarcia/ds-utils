from enum import Enum

class Alignment(Enum):
    """Text alignment options for EDA presentation utilities."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class Color(Enum):
    """Color palette used by EDA presentation utilities."""

    BLACK = "#000000"
    WHITE = "#ffffff"
    ORANGE = "#e64a19"
    BLUE = "#1976d2"
    PURPLE = "#7b1fa2"
    YELLOW = "#fbc02d"
    GREEN = "#2e7d32"
    RED = "#d32f2f"
    GRAY = "#4e4e4e"


# Canonical configuration shared by all preview-related utilities.
DEFAULT_COLOR = Color.WHITE
"""
the default color, which is important for ensuring consistency with the editor's light and dark themes
"""