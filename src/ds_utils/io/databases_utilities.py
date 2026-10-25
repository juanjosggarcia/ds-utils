import os
from typing import Any

from ds_utils.io.env_utilities import get_required_env_var


# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

# region SQL Functions ---------------------------------------------------------

def get_sql_engine(
    url: str,
    username_env_var: str | None = None,
    password_env_var: str | None = None,
    **engine_kwargs: Any,
) -> Any:
    """
    Create and validate a SQLAlchemy engine.

    SQLAlchemy requires the appropriate database driver to be installed
    for the database system specified in the connection URL.

    Database credentials can be included in the connection URL or
    provided separately through environment variables.

    Parameters
    ----------
    url : str
        SQLAlchemy database connection URL, with or without credentials.

    username_env_var : str, optional
        Name of the environment variable containing the database username.
        Must be provided together with `password_env_var` when credentials
        are supplied separately.

    password_env_var : str, optional
        Name of the environment variable containing the database password.
        Must be provided together with `username_env_var` when credentials
        are supplied separately.

    **engine_kwargs : Any
        Additional keyword arguments passed to `sqlalchemy.create_engine()`.

    Returns
    -------
    sqlalchemy.engine.Engine
        Configured SQLAlchemy engine with validated connectivity.

    Raises
    ------
    ImportError
        If SQLAlchemy or the database driver required by the connection
        URL is not installed.
    ValueError
        If the URL is empty, a required environment variable is not
        defined or is empty, or only one credential variable is provided.
    Exception
        If the connection check fails.

    Notes
    -----
    Authentication credentials can be configured in either of two ways:

    - Include the username and password in the connection URL.
    - Provide `username_env_var` and `password_env_var` to retrieve
      credentials from separate environment variables.

    If neither credential parameter is provided, the URL is used as-is.
    This supports connections with or without authentication.

    Environment variables must be available in the execution environment
    before calling this function. The `python-dotenv` package can
    optionally be used to load them from a `.env` file.
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL, make_url
    except ImportError as exc:
        raise ImportError(
            "SQL support requires SQLAlchemy. "
            "Install it with: pip install sqlalchemy"
        ) from exc

    if not url:
        raise ValueError("The database URL cannot be empty.")

    if (username_env_var is None) != (password_env_var is None):
        raise ValueError(
            "Both 'username_env_var' and 'password_env_var' "
            "must be provided together."
        )

    connection_url = make_url(url)

    if username_env_var is not None and password_env_var is not None:
        username = get_required_env_var(username_env_var)
        password = get_required_env_var(password_env_var)

        connection_url = connection_url.set(
            username=username,
            password=password,
        )

    engine = create_engine(connection_url, **engine_kwargs)

    # create_engine() does not establish a physical connection.
    # Open a connection to validate connectivity and authentication before
    # returning the engine. Dispose of the engine if validation fails.
    try:
        with engine.connect():
            pass
    except Exception:
        engine.dispose()
        raise

    return engine

# endregion SQL Functions ------------------------------------------------------

# region NO SQL Functions ------------------------------------------------------

def get_mongo_client(
    uri: str,
    username_env_var: str | None = None,
    password_env_var: str | None = None,
    **client_kwargs: Any,
) -> Any:
    """
    Create and validate a MongoDB client.

    Authentication credentials can be embedded in the MongoDB URI
    or provided separately through environment variables.

    Parameters
    ----------
    uri : str
        MongoDB connection URI. It can include authentication credentials
        or specify only the server connection details.

    username_env_var : str, optional
        Name of the environment variable containing the MongoDB username.
        Must be provided together with `password_env_var` when credentials
        are supplied separately.

    password_env_var : str, optional
        Name of the environment variable containing the MongoDB password.
        Must be provided together with `username_env_var` when credentials
        are supplied separately.

    **client_kwargs : Any
        Additional keyword arguments passed to `pymongo.MongoClient()`.

    Returns
    -------
    pymongo.MongoClient
        Configured MongoDB client with validated connectivity.

    Raises
    ------
    ImportError
        If PyMongo is not installed.
    ValueError
        If the URI is empty, a required environment variable is missing
        or empty, or only one credential variable name is provided.
    Exception
        If the connection or authentication check fails.

    Notes
    -----
    Authentication can be configured in either of two ways:

    - Include the username and password in the URI.
    - Provide `username_env_var` and `password_env_var` to retrieve
      credentials from separate environment variables.

    If neither credential parameter is provided, the URI is used as-is.
    This supports both authenticated URIs and connections without
    authentication.

    Environment variables must be available in the execution environment
    before calling this function. The `python-dotenv` package can
    optionally be used to load them from a `.env` file.
    """
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError(
            "MongoDB support requires PyMongo. "
            "Install it with: pip install pymongo"
        ) from exc

    if not uri:
        raise ValueError("The MongoDB URI cannot be empty.")

    if (username_env_var is None) != (password_env_var is None):
        raise ValueError(
            "Both 'username_env_var' and 'password_env_var' "
            "must be provided together."
        )

    if username_env_var is not None and password_env_var is not None:
        client_kwargs["username"] = get_required_env_var(username_env_var)
        client_kwargs["password"] = get_required_env_var(password_env_var)

    client = MongoClient(uri, **client_kwargs)

    # MongoClient uses lazy connections, so creating the client does not
    # guarantee that the server is reachable or that authentication succeeds.
    # Execute a ping to validate connectivity and credentials before returning
    # the client. Close it if validation fails to avoid leaking resources.
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        raise

    return client



# endregion NO SQL Functions ---------------------------------------------------