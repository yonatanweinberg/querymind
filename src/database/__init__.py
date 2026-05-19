"""
Database module for QueryMind.

Provides connection management and setup utilities for the Olist SQLite database.
"""

from src.database.connection import get_engine, test_connection, DB_PATH

__all__ = ["get_enginer", "test_connection", "DB_PATH"]