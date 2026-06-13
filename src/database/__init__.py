"""
Database module for QueryMind.

Provides connection management and setup utilities for the Olist SQLite database.
"""

from src.database.connection import DB_PATH, get_engine

__all__ = ["get_engine", "DB_PATH"]
