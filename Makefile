# ============================================
# QueryMind - Common Commands
# ============================================
# On Windows, you may not have `make` installed.
# You can either:
#   1. Install Make via: choco install make (if you have Chocolatey)
#   2. Or just copy-paste the commands below directly into your terminal.
#
# Usage: make <target>
# ============================================

.PHONY: setup install install-dev test test-cov lint format run db-setup clean

# First-time setup: create venv + install all dependencies
setup:
	python -m venv .venv
	.venv\Scripts\activate && pip install -e ".[dev]"

# Install production dependencies only
install:
	pip install -e .

# Install with dev dependencies (pytest, ruff, etc.)
install-dev:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest -v

# Run tests with coverage report
test-cov:
	pytest --cov=src --cov-report=term-missing -v

# Lint code (check for issues without fixing)
lint:
	ruff check src/ tests/ app/ evaluation/ scripts/

# Format code (auto-fix style issues)
format:
	ruff check --fix src/ tests/ app/ evaluation/ scripts/
	ruff format src/ tests/ app/ evaluation/ scripts/

# Run the Streamlit app
run:
	streamlit run app/streamlit_app.py

# Load Olist data into SQLite
db-setup:
	python -m src.database.setup

# Remove generated files
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
