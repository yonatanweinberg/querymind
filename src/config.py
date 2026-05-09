"""
Centralized configuration loader for QueryMind

Loads config/settings.yaml once per process and exposes its contents as
typed dataclasses. First call to get_settings() parses the file; every
subsequent call returns the cached Settings object (no file I/O).

Usage:
    from src.config import get_settings

    settings = get_settings()
    model = settings.llm.model
    in_price = settings.llm.pricing.input_per_mtok_usd
    limit = settings.safety.default_limit

Why dataclasses (alternative was dicts):
    - Typed access (settings.llm.model) instead of settings["llm"]["model"].
    - Missing keys raise at load time with a clear message, not at first
      use in some obscure code path, 2 weeks from now.

Why lazy-load
    - Importing src.config shouldn't do I/O. First get_settings() call
      does the read and caches; subsequent calls are free.

Why not Pydantic:
    - Pydantic is the production-grade tool for this purpose. For a
      single config file with ~10 fields, dataclasses keep the dependency
      footprint smaller and the pattern explicit. Easy to migrate later
      if config grows.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config file location
# ---------------------------------------------------------------------------

# Project root is one level up from src/ (src/config.py --> querymind/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Typed config structures
# ---------------------------------------------------------------------------
# These mirror the shape of settings.yaml exactly. Adding a new setting
# requires (a) adding it to settings.yaml and (b) adding it to the matching
# dataclass below - load step fails LOUDLY if either is forgotten.
# ---------------------------------------------------------------------------

@dataclass(frozen=True) # Makes dataclass immutable after construction
class PricingConfig:
    """LLM pricing per million tokens, in USD.

    Used by LLMUsage.estimated_cost_usd to convert raw token counts into
    a dollar figure for the UI. Lives under LLMConfig because pricing
    is per-model: switching from Sonnet($3in./$15out.) to Haiku ($1/$5)
    changes both 'model' and 'pricing' together.
    """
    input_per_mtok_usd: float
    output_per_mtok_usd: float

@dataclass(frozen=True) 
class LLMConfig:
    # LLM provider settings
    model: str
    max_tokens: int
    temperature: float
    pricing: PricingConfig


@dataclass(frozen=True)
class SafetyConfig:
    # SQL safety pipeline knobs
    default_limit: int
    max_limit: int
    max_subquery_depth: int
    large_table_threshold: int

@dataclass(frozen=True)
class RAGConfig:
    """RAG retrieval counts (stratified retrieval, one per source type)
    
    Note: embedding model and collection name are NOT here - those live under
    src/rag/_config.py - since they must be identical between embedder and
    retriever. Architectural invariants, not runtime knobs.
    """
    n_schema: int
    n_glossary: int
    n_examples: int
    n_join_paths: int


@dataclass(frozen=True)
class Settings:
    # Top-level settings object — the root of the config tree
    llm: LLMConfig
    safety: SafetyConfig
    rag: RAGConfig


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_cached_settings: Settings | None = None

def get_settings(path: Path | None = None) -> Settings:
    """Load and return the cached Settings object.

    On first call, reads config/settings.yaml and parses it into the
    typed dataclasses defined above. Subsequent calls return the cached
    object with no file I/O.

    Args:
        path: Optional override for the settings file path. Primarily for
        tests that need to load from a fixture. If None, uses the default
        SETTINGS_PATH and participates in the cache.
    
    Returns:
        Settings: The parsed settings object.
    
    Raises:
        FileNotFoundError: If the settings file is missing.
        ValueError: If a required section or field is missing.
    """
    global _cached_settings

    # Explicit path bypasses the cache - used by tests that load a fixture.
    if path is not None:
        return _load_settings(path)
    
    if _cached_settings is None:
        _cached_settings = _load_settings(SETTINGS_PATH)

    return _cached_settings

def reset_cache() -> None:
    """Clear the cached Settings. Again, primarily for tests.
    
    After calling, the next get_settings() call reloads from disk.
    Use if a test writes a different settings.yaml and needs
    the production code to pick it up.
    """
    global _cached_settings
    _cached_settings = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_settings(path: Path) -> Settings:
    """Read and parse the settings YAML into a Settings object.

    Fails LOUD with a clear error message if anything is missing
    or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Settings file not found at {path}. "
            f"Expected a YAML file with 'llm', 'safety', and 'rag' sections."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Settings file at {path} did not parse as a dict. "
            f"Got {type(raw).__name__}."
        )
    
    # Parse each section. _require returns the value or raises with a
    # message that names the specific missing piece.
    llm_raw = _require(raw, "llm", path)
    safety_raw = _require(raw, "safety", path)
    rag_raw = _require(raw, "rag", path)

    # Pricing is nested under llm; pull it out and parse separately so
    # the error message can name "llm.pricing.input_per_mtok_usd" exactly
    # if a leaf field is missing.
    pricing_raw = _require(llm_raw, "pricing", path, "llm")

    return Settings(
        llm=LLMConfig(
            model=_require(llm_raw, "model", path, "llm"),
            max_tokens=_require(llm_raw, "max_tokens", path, "llm"),
            temperature=_require(llm_raw, "temperature", path, "llm"),
            pricing=PricingConfig(
                input_per_mtok_usd=_require(
                    pricing_raw, "input_per_mtok_usd", path, "llm.pricing"
                ),
                output_per_mtok_usd=_require(
                    pricing_raw, "output_per_mtok_usd", path, "llm.pricing"
                ),
            ),
        ),
        safety=SafetyConfig(
            default_limit=_require(safety_raw, "default_limit", path, "safety"),
            max_limit=_require(safety_raw, "max_limit", path, "safety"),
            max_subquery_depth=_require(
                safety_raw, "max_subquery_depth", path, "safety"
            ),
            large_table_threshold=_require(
                safety_raw, "large_table_threshold", path, "safety"
            ),
        ),
        rag=RAGConfig(
            n_schema=_require(rag_raw, "n_schema", path, "rag"),
            n_glossary=_require(rag_raw, "n_glossary", path, "rag"),
            n_examples=_require(rag_raw, "n_examples", path, "rag"),
            n_join_paths=_require(rag_raw, "n_join_paths", path, "rag"),
        ),
    )


def _require(
        section: dict, key: str, path: Path, section_name: str = ""
) -> object:
    # Fetch a required key from a settings section, or raise ValueError
    if key not in section:
        location = f"{section_name}.{key}" if section_name else key
        raise ValueError(
            f"Missing required setting '{location}' in {path}."
        )
    return section[key]
