"""
Database connection management for QueryMind.

Provides two connection modes:
- Read-write: Used ONLY by the setup script to create/populate tables.
- Read-only:  Used by the application pipeline. This is a defense-in-depth
              safety layer - even if the SQL safety pipeline fails, the
              database physically cannot be modified.

Uses SQLAlchemy for database-agnostic abstraction. The project currently
targets SQLite, but switching to PostgreSQL or another backend would only
require changing the connection string.
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# Compute the project root dynamically so this module works regardless
# of the current working directory. The project root is two levels up
# from src/database/connection.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"


def get_engine(readonly: bool = True) -> Engine:
    """
    Create a SQLAlchemy engine for the Olist SQLite database.

    Parameters
    ----------
    readonly : bool, default True
        If True, the connection enforces PRAGMA query_only = ON so that
        any INSERT, UPDATE, DELETE, DROP, etc. will be rejected at the
        database driver level. Set to False only in the setup script.

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    if not DB_PATH.exists() and readonly:
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run 'python -m src.database.setup' to create it."
        )

    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        # echo=True  # Uncomment during debugging to see all SQL statements
    )

    if readonly:

        @event.listens_for(engine, "connect")
        def _set_query_only(dbapi_connection, connection_record):
            """
            Called every time a new low-level connection is created.
            PRAGMA query_only makes the connection reject any writes.
            """
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA query_only = ON")
            cursor.close()

    return engine
