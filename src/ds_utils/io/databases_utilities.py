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
    url_env_var: str,
    username_env_var: str,
    password_env_var: str,
    **engine_kwargs: Any,
) -> Any:
    """
    Create and validate a SQLAlchemy engine.

    Parameters
    ----------
    url_env_var : str
        Name of the environment variable containing the database URL.
    username_env_var : str
        Name of the environment variable containing the database username.
    password_env_var : str
        Name of the environment variable containing the database password.
    **engine_kwargs : Any
        Additional keyword arguments passed to sqlalchemy.create_engine().

    Returns
    -------
    sqlalchemy.engine.Engine
        Configured SQLAlchemy engine with validated connectivity.

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed.
    ValueError
        If a required environment variable is not defined or is empty.
    Exception
        If the connection check fails.
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL, make_url
    except ImportError as exc:
        raise ImportError(
            "SQL support requires SQLAlchemy. "
            "Install it with: pip install sqlalchemy"
        ) from exc


    url = get_required_env_var(url_env_var)
    username = get_required_env_var(username_env_var)
    password = get_required_env_var(password_env_var)

    parsed_url = make_url(url)

    connection_url = URL.create(
        drivername=parsed_url.drivername,
        username=username,
        password=password,
        host=parsed_url.host,
        port=parsed_url.port,
        database=parsed_url.database,
        query=parsed_url.query,
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
    url_env_var: str,
    username_env_var: str,
    password_env_var: str,
    **client_kwargs: Any,
) -> Any:
    """
    Create and validate a MongoDB client.

    Parameters
    ----------
    url_env_var : str
        Name of the environment variable containing the MongoDB URI.
    username_env_var : str
        Name of the environment variable containing the MongoDB username.
    password_env_var : str
        Name of the environment variable containing the MongoDB password.
    **client_kwargs : Any
        Additional keyword arguments passed to pymongo.MongoClient().

    Returns
    -------
    pymongo.MongoClient
        Configured MongoDB client with validated connectivity.

    Raises
    ------
    ImportError
        If PyMongo is not installed.
    ValueError
        If a required environment variable is not defined or is empty.
    Exception
        If the connection or authentication check fails.
    """
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError(
            "MongoDB support requires PyMongo. "
            "Install it with: pip install pymongo"
        ) from exc

    uri = get_required_env_var(url_env_var)
    username = get_required_env_var(username_env_var)
    password = get_required_env_var(password_env_var)

    client = MongoClient(
        uri,
        username=username,
        password=password,
        **client_kwargs,
    )

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