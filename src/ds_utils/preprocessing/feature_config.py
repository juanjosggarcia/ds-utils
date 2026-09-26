from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureNamingConfig:
    """Naming convention used by the feature engineering utilities."""

    CLEAN_SUFFIX: str = "_clean"
    IS_NOISE_SUFFIX: str = "_is_noise"

    DIFF_SUFFIX: str = "_diff"
    DIFF_CLEAN_SUFFIX: str = "_diff_clean"

    NOISE_MAGNITUDE_SUFFIX: str = "_noise_magnitude"
    ROLLING_STD_SUFFIX: str = "_rolling_std"

    GLOBAL_NOISE_COLUMN: str = "any_sensor_noise"

    ROLLING_WINDOW: int = 5



# Canonical configuration shared by all feature-related utilities.
FEATURES_CONFIG = FeatureNamingConfig()