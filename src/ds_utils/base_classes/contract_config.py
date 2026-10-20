from dataclasses import dataclass

@dataclass(frozen=True)
class ContractNamingConfig:
    """Naming conventions used by dataset contracts."""

    COLUMN_PREFIX: str = "COL_"
    INDEX_CONSTANT: str = "COL_INDEX"

    DF_NAME: str = "DF_NAME"
    DATA_COLUMNS: str = "DATA_COLUMNS"
    REQUIRED_COLUMNS: str = "REQUIRED_COLUMNS"
    USED_COLUMNS: str = "USED_COLUMNS"


# Canonical configuration shared by all feature-related utilities.
CONTRACT_CONFIG = ContractNamingConfig()
