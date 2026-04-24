"""
Bundle the QueryMind project source into a single text file for review.
Run from the project root: python scripts/bundle_for_review.py
"""

from pathlib import Path
from datetime import datetime

OUTPUT_FILE = "querymind_bundle.txt"

INCLUDE_DIRS = ["src", "app", "tests", "config", "evaluation"]

INCLUDE_FILES = [
    "pyproject.toml",
    "Makefile",
    "Dockerfile",
    ".env.example",
    ".gitignore",
    "README.md",
]

INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg"}

# Any path containing one of these substrings is skipped
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".venv",
    ".git/",
    ".git\\",
    "chroma_db",
    "data/raw",
    "data\\raw",
    "olist.db",
    ".pytest_cache",
    ".egg-info",
    "node_modules",
]

SEPARATOR = "=" * 40


def should_skip(path: Path) -> bool:
    """Return True if this path matches any exclusion pattern."""
    path_str = str(path)
    return any(pattern in path_str for pattern in EXCLUDE_PATTERNS)


def read_text_safely(path: Path) -> str:
    """Read a text file, falling back gracefully on encoding issues."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[SKIPPED - non-UTF8 content: {path.name}]"


def main() -> None:
    root = Path.cwd()
    lines: list[str] = []

    # Header
    lines.append(SEPARATOR)
    lines.append("QUERYMIND PROJECT BUNDLE")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(SEPARATOR)
    lines.append("")
    lines.append("### DIRECTORY TREE ###")
    lines.append("")

    # Full file listing (tree view), respecting exclusions
    all_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not should_skip(p)
    )
    for f in all_files:
        lines.append(str(f.relative_to(root)))

    # Top-level files
    for filename in INCLUDE_FILES:
        filepath = root / filename
        if filepath.is_file():
            lines.append("")
            lines.append(SEPARATOR)
            lines.append(f"### FILE: {filename} ###")
            lines.append(SEPARATOR)
            lines.append(read_text_safely(filepath))

    # Files inside each included directory
    for dirname in INCLUDE_DIRS:
        dirpath = root / dirname
        if not dirpath.is_dir():
            continue

        files_in_dir = sorted(
            p for p in dirpath.rglob("*")
            if p.is_file()
            and p.suffix in INCLUDE_EXTENSIONS
            and not should_skip(p)
        )
        for f in files_in_dir:
            rel = f.relative_to(root)
            lines.append("")
            lines.append(SEPARATOR)
            lines.append(f"### FILE: {rel} ###")
            lines.append(SEPARATOR)
            lines.append(read_text_safely(f))

    # Write the bundle
    output_path = root / OUTPUT_FILE
    output_path.write_text("\n".join(lines), encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"\nBundle created: {OUTPUT_FILE} ({size_kb:.1f} KB)")
    print(f"Files included: {sum(1 for l in lines if l.startswith('### FILE:'))}")
    print()
    print("BEFORE UPLOADING - scan the bundle for secrets.")
    print("In VS Code, open the bundle and Ctrl+F for:")
    print("  sk-ant-   (Anthropic API keys)")
    print("  sk-proj-  (OpenAI API keys)")
    print("  api_key   (generic)")
    print("  password  (generic)")


if __name__ == "__main__":
    main()