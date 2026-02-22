"""
Database setup script for QueryMind.

Reads the 9 Olist CSV files from data/raw/ and loads them into a local
SQLite database at data/olist.db. This script is idempotent — running it
again will drop and recreate all tables with fresh data.

Usage:
    python -m src.database.setup

Why SQLite? (Blueprint Section 3.3)
    Zero configuration, no server process, single file, and anyone who
    clones the repo can recreate the database with one command. The
    architecture is database-agnostic via SQLAlchemy, so pointing at
    PostgreSQL would only require a connection string change.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine, DB_PATH, PROJECT_ROOT

# ---------------------------------------------------------------------------
# CSV-to-table mapping
# ---------------------------------------------------------------------------
# Maps each CSV filename to the table name it will become in SQLite.
# We strip the "_dataset" suffix to get cleaner table names that match
# the blueprint's schema diagram (Section 3.2).
#
# The table names intentionally keep the "olist_" prefix — this avoids
# collisions with SQL reserved words (e.g., "orders") and makes it
# immediately clear which tables belong to the Olist dataset.
CSV_TO_TABLE = {
    "olist_customers_dataset.csv": "olist_customers",
    "olist_orders_dataset.csv": "olist_orders",
    "olist_order_items_dataset.csv": "olist_order_items",
    "olist_order_payments_dataset.csv": "olist_order_payments",
    "olist_order_reviews_dataset.csv": "olist_order_reviews",
    "olist_products_dataset.csv": "olist_products",
    "olist_sellers_dataset.csv": "olist_sellers",
    "olist_geolocation_dataset.csv": "olist_geolocation",
    "product_category_name_translation.csv": "product_category_name_translation",
}

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csvs_to_sqlite() -> dict[str, int]:
    """
    Load all Olist CSVs into the SQLite database.

    Returns
    -------
    dict[str, int]
        Mapping of table_name -> row_count for each loaded table.
    """
    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    missing = [f for f in CSV_TO_TABLE if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print("ERROR: The following CSV files are missing from data/raw/:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nDownload the Olist dataset from:"
            "\n  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            "\nand place the CSV files in data/raw/"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create engine in read-WRITE mode (this is the only place we do this)
    # ------------------------------------------------------------------
    engine = get_engine(readonly=False)

    row_counts = {}

    for csv_filename, table_name in CSV_TO_TABLE.items():
        csv_path = RAW_DATA_DIR / csv_filename

        # Read CSV into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Write to SQLite — if_exists="replace" makes this idempotent.
        # index=False avoids creating an extra auto-increment column.
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)

        row_counts[table_name] = len(df)
        print(f"  ✓ {table_name:<45} {len(df):>8,} rows")

    return row_counts


def verify_database(engine) -> None:
    """
    Run a few sanity checks to make sure the database was created correctly.
    Queries a known relationship: orders -> order_items -> products.
    """
    with engine.connect() as conn:
        # Check that we can join across the three core tables
        result = conn.execute(text("""
            SELECT COUNT(*) AS item_count
            FROM olist_orders o
            JOIN olist_order_items oi ON o.order_id = oi.order_id
            JOIN olist_products p ON oi.product_id = p.product_id
        """))
        item_count = result.scalar()
        print(f"\n  Verification: orders → items → products join returned {item_count:,} rows")

        # Check table count
        result = conn.execute(text(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ))
        table_count = result.scalar()
        print(f"  Verification: {table_count} tables found in database")


def main() -> None:
    """Entry point for the setup script."""
    print("=" * 60)
    print("QueryMind — Database Setup")
    print("=" * 60)
    print(f"\nSource:  {RAW_DATA_DIR}")
    print(f"Target:  {DB_PATH}\n")

    start = time.time()

    print("Loading CSVs into SQLite...\n")
    row_counts = load_csvs_to_sqlite()

    total_rows = sum(row_counts.values())
    elapsed = time.time() - start

    print(f"\n{'─' * 60}")
    print(f"  Total: {total_rows:>10,} rows across {len(row_counts)} tables")
    print(f"  Time:  {elapsed:.1f} seconds")
    print(f"  DB:    {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # Run verification with a read-only connection to confirm it works
    print(f"\nRunning verification queries...")
    engine_ro = get_engine(readonly=True)
    verify_database(engine_ro)

    print(f"\n{'=' * 60}")
    print("Database setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

