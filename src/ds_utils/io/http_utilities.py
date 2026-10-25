import os

from ds_utils.io.env_utilities import get_required_env_var

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "HTTP utilities require requests. "
        "Install it with: pip install requests"
    ) from exc

# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Http Functions --------------------------------------------------------

def get_http_session(
    token_env_var: str | None = None,
) -> requests.Session:
    """
    Create and configure an HTTP session.

    Parameters
    ----------
    token_env_var : str, optional
        Name of the environment variable containing the Bearer token.
        If None, the session is created without authentication.

    Returns
    -------
    requests.Session
        Configured HTTP session.

    Raises
    ------
    ValueError
        If the specified environment variable is empty or not defined.
    """
    session = requests.Session()

    session.headers.update({
        "accept": "application/json",
    })

    if token_env_var is not None:
        token = get_required_env_var(token_env_var)

        session.headers.update({
            "Authorization": f"Bearer {token}",
        })

    return session

# endregion Http Functions -----------------------------------------------------