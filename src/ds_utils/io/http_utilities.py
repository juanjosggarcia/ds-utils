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
    Create and configure an HTTP session for making API requests.

    The session is configured with default headers for JSON responses
    and can optionally include Bearer token authentication. The token
    is retrieved from the specified environment variable.

    Parameters
    ----------
    token_env_var : str, optional
        Name of the environment variable containing the Bearer token.
        If None, the session is created without authentication headers.

    Returns
    -------
    requests.Session
        Configured HTTP session with default headers and optional
        Bearer token authentication.

    Raises
    ------
    ValueError
        If `token_env_var` is specified but the corresponding
        environment variable is not defined or contains an empty value.

    Notes
    -----
    The returned session can be reused across multiple HTTP requests,
    allowing connection pooling and consistent configuration.

    Environment variables must be available in the execution
    environment before calling this function. The `python-dotenv`
    package can optionally be used to load them from a `.env` file.
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
