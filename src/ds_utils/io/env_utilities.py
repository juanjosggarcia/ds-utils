import os

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Environment Functions -------------------------------------------------

def get_required_env_var(name: str) -> str:
    """
    Retrieve a required environment variable.

    Parameters
    ----------
    name : str
        Name of the environment variable.

    Returns
    -------
    str
        Value of the environment variable.

    Raises
    ------
    ValueError
        If the environment variable is not defined or contains
        an empty value.

    Notes
    -----
    Environment variables must be available in the execution
    environment before calling this function. For environment-specific
    configuration, `python-dotenv` can optionally be installed
    (`pip install python-dotenv`) to load variables from a `.env` file
    using `load_dotenv()` from the `dotenv` module.
    """
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Environment variable '{name}' is not defined "
            "or contains an empty value."
        )

    return value

# endregion Environment Functions ----------------------------------------------
