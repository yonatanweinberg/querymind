"""
First-boot data preparation for the deployed app.

A fresh Streamlit Community Cloud container starts with an empty data/
directory: both the SQLite database and the Chroma vector store are
gitignored, so neither ships with the repo. This module reconstructs them
once per container lifecycle.

The two artifacts get different treatment, because they cost very different
amounts to produce:

    - olist.db is the expensive build-once artifact (it needs the full Olist
      CSVs to build). We do not rebuild it on the container; we download a
      prebuilt copy hosted as a public GitHub Release asset.
    - The Chroma store is cheap, and its on-disk format is tied to the
      chromadb version, so shipping a prebuilt store risks a load failure if
      the hosted install resolves a different version. We rebuild it from the
      config YAMLs instead. build_vector_store() reads only these YAMLs (never
      the database), embeds 66 chunks in a few seconds, and as a side effect
      downloads and loads the embedding model, so the first user query does
      not pay that cost mid-request.

ensure_ready() is wrapped in @st.cache_resource, so its body runs exactly
once per container. Every later rerun (each click, each keystroke) hits the
cache and skips it.

Local development never reaches the download path: the README quick start
builds both artifacts with `python -m src.database.setup` and
`python -m src.rag.embedder`, so DB_PATH already exists.
"""

import logging
import shutil
import urllib.request

import streamlit as st

from src.database.connection import DB_PATH
from src.rag.embedder import build_vector_store

# Public GitHub Release asset holding the prebuilt SQLite database. This is
# the asset *download* URL (/releases/download/<tag>/<file>), which returns
# the file itself.
DATA_ASSET_URL = (
    "https://github.com/yonatanweinberg/querymind/releases/download/data-v1/olist.db"
)

# Every SQLite 3 database begins with this exact 16-byte header. Checking it
# after download is a cheap guard against the most likely failure: a wrong or
# stale URL that quietly returns an HTML page, which would otherwise be saved
# as olist.db and only fail later when SQLAlchemy tries to open it.
SQLITE_MAGIC = b"SQLite format 3\x00"

# Upper bound on how long any single network read may block before we give
# up, so a stalled connection cannot hang the container's boot indefinitely.
DOWNLOAD_TIMEOUT_S = 120

logger = logging.getLogger(__name__)


def _download_database() -> None:
    """Download the prebuilt SQLite database from the GitHub Release asset.

    Writes to a temporary file and only moves it into place once the SQLite
    header has been verified. A failed or partial download therefore never
    leaves a half-written file at DB_PATH, which matters because
    ensure_ready() treats an existing DB_PATH as "already downloaded".
    """
    logger.info("Database not found locally; downloading from release asset.")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DB_PATH.with_name(DB_PATH.name + ".partial")

    # The User-Agent header keeps GitHub's CDN (Content Delivery Network) happy.
    # urllib follows the 302 redirect from github.com to the asset's storage
    # backend on its own.
    request = urllib.request.Request(
        DATA_ASSET_URL, headers={"User-Agent": "querymind-deploy"}
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
        with open(tmp_path, "wb") as f:
            # Stream in chunks rather than response.read(), so a ~107MB file
            # never sits fully in memory on a 1 GB container.
            shutil.copyfileobj(response, f)

    with open(tmp_path, "rb") as f:
        header = f.read(len(SQLITE_MAGIC))
    if header != SQLITE_MAGIC:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file is not a SQLite database (unexpected header). "
            "Check that DATA_ASSET_URL points at the release asset, not the "
            f"release page: {DATA_ASSET_URL}"
        )

    # Path.replace uses os.replace: atomic on the same filesystem, so DB_PATH
    # appears only once the file is complete and validated.
    tmp_path.replace(DB_PATH)
    logger.info("Database downloaded and verified at %s", DB_PATH)


@st.cache_resource(show_spinner="Preparing data (first launch only)...")
def ensure_ready() -> None:
    """Make the database and vector store available, once per container.

    Called near the top of streamlit_app.py, before the database engine or
    the retriever are first used. The @st.cache_resource wrapper means the
    body runs only on the first script execution per container.
    """
    if not DB_PATH.exists():
        _download_database()

    # Safe to run on every cold boot: build_vector_store() is idempotent (it
    # drops and recreates the collection) and reads only the config YAMLs.
    # Rebuilding sidesteps any chromadb-version coupling that a prebuilt store
    # would carry, and costs only a few seconds at 66 chunks.
    build_vector_store()
