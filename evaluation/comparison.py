"""
Result-table comparison for the QueryMind evaluation harness.

QueryMind's accuracy eval judges generated SQL on RESULT-CORRECTNESS
(a.k.a. denotation / execution accuracy): the model's SQL and the gold SQL
are each executed, and their result tables are compared. This module offers
two complementary verdicts, both independent of column NAMES and (unless the
question is a ranking) row ORDER:

  * STRICT result-correctness - compare_results(). The model's table must have
    the SAME columns as gold (any order) and the same rows, with numbers equal
    to two decimals. This is the rigorous, exact-output criterion.

  * ANSWER-CONTAINMENT - compare_contains(). Gold's columns must all be PRESENT
    in the model's output with matching values, but the model may return EXTRA
    columns, and numbers are compared at one decimal so reasonable precision
    differences are tolerated. Row count must still match (extra columns are
    credited, extra rows are not). This reflects how a conversational BI agent
    is actually used: returning the requested figures plus relevant context
    (the count behind an average, a descriptive label) answers the question,
    even though it would fail an exact-output match.

Reporting both, and the gap between them, is the point: it quantifies how often
the model returns the correct figures while enriching or rounding the output
differently than gold. Genuine errors (a wrong value, an off-by-one count, a
missing column, padded rows) fail under BOTH.

The module is pure - DataFrames in, a verdict out, no I/O - so this bug-prone
core can be unit-tested in isolation (tests/test_eval_comparison.py). It mirrors
the denotation-accuracy methodology of text-to-SQL benchmarks such as Spider and
BIRD, extended with column-permutation invariance, per-question order
sensitivity, and the containment view for a conversational product.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from dataclasses import dataclass
from itertools import permutations

import pandas as pd

# Numbers are rounded to this many decimals before comparison; the rounding IS
# the numeric tolerance. STRICT uses 2 dp (matching the gold queries'
# ROUND(..., 2) and absorbing float representation noise). CONTAINMENT uses 1 dp,
# tolerating coarser-but-reasonable model precision (e.g. 12.3 vs 12.34 days)
# while still failing on any difference a correct query would not produce.
DECIMALS_STRICT = 2
DECIMALS_CONTAINS = 1


@dataclass
class ComparisonResult:
    """Verdict from comparing a model result table against the gold table."""

    match: bool
    reason: str = ""  # short explanation when match is False (for the log)


def _is_null(value) -> bool:
    """True for SQL NULL / pandas NaN / None. Scalar-safe."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_number(value):
    """Return value as a float if it is genuinely numeric, else None.

    Covers Python and numpy int/float (both register as numbers.Number).
    Strings are NOT coerced, even if they look numeric: a text column and a
    numeric column must never compare equal.
    """
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        f = float(value)
        return None if math.isnan(f) else f
    return None


def _canon_cell(value, decimals):
    """Map a cell to a hashable canonical form under the eval's equality rules.

    NULLs collapse to one sentinel; numbers to their value rounded to `decimals`;
    everything else to its trimmed string. Two cells are equal iff their
    canonical forms are equal.
    """
    if _is_null(value):
        return ("null",)
    number = _as_number(value)
    if number is not None:
        return ("num", round(number, decimals))
    return ("str", str(value).strip())


def _canon_row(row, decimals) -> tuple:
    return tuple(_canon_cell(cell, decimals) for cell in row)


def values_equal(a, b) -> bool:
    """Public helper: are two scalar cells equal under the STRICT rules?"""
    return _canon_cell(a, DECIMALS_STRICT) == _canon_cell(b, DECIMALS_STRICT)


def _rows(df: pd.DataFrame, column_order, decimals):
    """Canonical rows of df with columns taken in the given index order."""
    reordered = df.iloc[:, list(column_order)]
    return [
        _canon_row(r, decimals) for r in reordered.itertuples(index=False, name=None)
    ]


def _match(
    model_df, gold_df, order_sensitive, decimals, allow_extra_columns
) -> ComparisonResult:
    """Core comparison shared by both verdicts.

    Tries every way to line up gold's columns against the model's columns
    (a permutation when columns must match exactly, an injection when extra
    model columns are allowed) and checks whether some alignment reproduces
    the gold table - as an ordered sequence for ranking questions, otherwise
    as an unordered multiset of rows.
    """
    if model_df is None:
        return ComparisonResult(False, "model produced no result table")

    n_model, n_gold = model_df.shape[1], gold_df.shape[1]
    if allow_extra_columns:
        if n_model < n_gold:
            return ComparisonResult(
                False, f"model has fewer columns than gold ({n_model} < {n_gold})"
            )
    else:
        if n_model != n_gold:
            return ComparisonResult(
                False, f"column count differs (model {n_model}, gold {n_gold})"
            )

    # Extra COLUMNS may be credited (containment); extra ROWS never are.
    if len(model_df) != len(gold_df):
        return ComparisonResult(
            False, f"row count differs (model {len(model_df)}, gold {len(gold_df)})"
        )

    gold_rows = _rows(gold_df, range(n_gold), decimals)
    gold_multiset = None if order_sensitive else Counter(gold_rows)

    # permutations(range(n_model), n_gold): full permutations when n_model ==
    # n_gold (strict), or ordered injections selecting n_gold of the model's
    # columns when extras are allowed (containment). Cheap at a few columns.
    for sel in permutations(range(n_model), n_gold):
        model_rows = _rows(model_df, sel, decimals)
        if order_sensitive:
            if model_rows == gold_rows:
                return ComparisonResult(True)
        else:
            if Counter(model_rows) == gold_multiset:
                return ComparisonResult(True)

    return ComparisonResult(False, "values differ (no column alignment matches)")


def compare_results(
    model_df: pd.DataFrame | None, gold_df: pd.DataFrame, order_sensitive: bool
) -> ComparisonResult:
    """STRICT result-correctness: gold and model must have the same columns
    (any order) and the same rows, with numbers equal to two decimals."""
    return _match(
        model_df, gold_df, order_sensitive, DECIMALS_STRICT, allow_extra_columns=False
    )


def compare_contains(
    model_df: pd.DataFrame | None, gold_df: pd.DataFrame, order_sensitive: bool
) -> ComparisonResult:
    """ANSWER-CONTAINMENT: every gold column is present in the model's output
    with matching values (extra model columns allowed), the row count matches,
    and numbers are compared at one decimal. Credits a correct answer returned
    with additional context or coarser rounding."""
    return _match(
        model_df, gold_df, order_sensitive, DECIMALS_CONTAINS, allow_extra_columns=True
    )
