"""
Unit tests for the evaluation result-table comparator (evaluation/comparison.py).

The comparator is the bug-prone core of the eval harness: a silent mistake here
would corrupt every accuracy number. These tests pin down each rule -
column-name/order invariance, row-order sensitivity, numeric rounding, NULL
handling, type strictness, shape mismatches, and the answer-containment verdict -
against small hand-built tables.
"""

import numpy as np
import pandas as pd

from evaluation.comparison import compare_contains, compare_results, values_equal


def _df(rows, columns):
    return pd.DataFrame(rows, columns=columns)


# --- cell-level equality -------------------------------------------------


def test_values_equal_numeric_rounding():
    assert values_equal(4.0712, 4.07)  # agree once rounded to 2 dp
    assert not values_equal(4.07, 4.20)  # genuinely different


def test_values_equal_count_off_by_one_fails():
    assert not values_equal(1234, 1235)  # a wrong count must fail


def test_values_equal_null_semantics():
    assert values_equal(np.nan, None)  # NULL == NULL
    assert not values_equal(np.nan, 0)  # NULL != a value


def test_values_equal_string_is_trimmed_but_case_sensitive():
    assert values_equal(" SP ", "SP")  # surrounding whitespace ignored
    assert not values_equal("sp", "SP")  # case is meaningful


def test_values_equal_number_vs_string_differ():
    assert not values_equal(5, "5")  # type mismatch is not equal


# --- strict table-level equality -----------------------------------------


def test_identical_tables_match():
    a = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    b = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    assert compare_results(a, b, order_sensitive=False).match


def test_column_names_ignored():
    model = _df([["SP", 100.0]], ["uf", "total"])
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_column_order_ignored():
    model = _df([[100.0, "SP"]], ["revenue", "state"])
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_row_order_ignored_when_not_sensitive():
    model = _df([["RJ", 50.0], ["SP", 100.0]], ["state", "revenue"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_row_order_matters_when_sensitive():
    model = _df([["RJ", 50.0], ["SP", 100.0]], ["state", "revenue"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    assert not compare_results(model, gold, order_sensitive=True).match


def test_correct_ranking_order_matches():
    model = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    assert compare_results(model, gold, order_sensitive=True).match


def test_numeric_tolerance_in_table():
    model = _df([["A", 4.0712]], ["cat", "score"])
    gold = _df([["A", 4.07]], ["cat", "score"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_value_difference_fails():
    model = _df([["A", 4.20]], ["cat", "score"])
    gold = _df([["A", 4.07]], ["cat", "score"])
    assert not compare_results(model, gold, order_sensitive=False).match


def test_null_cells_match():
    model = _df([["A", np.nan]], ["cat", "val"])
    gold = _df([["A", None]], ["cat", "val"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_null_vs_value_fails():
    model = _df([["A", np.nan]], ["cat", "val"])
    gold = _df([["A", 5.0]], ["cat", "val"])
    assert not compare_results(model, gold, order_sensitive=False).match


def test_column_count_mismatch_reported():
    model = _df([["SP", 100.0, 1]], ["state", "revenue", "orders"])
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    res = compare_results(model, gold, order_sensitive=False)
    assert not res.match
    assert "column count" in res.reason


def test_row_count_mismatch_reported():
    model = _df([["SP", 100.0]], ["state", "revenue"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    res = compare_results(model, gold, order_sensitive=False)
    assert not res.match
    assert "row count" in res.reason


def test_empty_tables_match():
    model = _df([], ["state", "revenue"])
    gold = _df([], ["state", "revenue"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_model_none_fails():
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    assert not compare_results(None, gold, order_sensitive=False).match


def test_multi_row_unordered_with_duplicate_values():
    model = _df([["B", 50.0], ["A", 50.0]], ["cat", "val"])
    gold = _df([["A", 50.0], ["B", 50.0]], ["cat", "val"])
    assert compare_results(model, gold, order_sensitive=False).match


def test_integer_vs_float_same_value_match():
    model = _df([[1234]], ["n"])
    gold = _df([[1234.0]], ["n"])
    assert compare_results(model, gold, order_sensitive=False).match


# --- answer-containment --------------------------------------------------


def test_contains_credits_extra_column():
    # Model returns the requested figures PLUS an extra count column.
    model = _df([["SP", 100.0, 5], ["RJ", 50.0, 3]], ["state", "revenue", "orders"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    assert compare_contains(model, gold, order_sensitive=False).match
    # ...and strict correctly rejects the shape difference.
    assert not compare_results(model, gold, order_sensitive=False).match


def test_contains_tolerates_coarser_precision():
    model = _df([["A", 12.3]], ["cat", "days"])
    gold = _df([["A", 12.34]], ["cat", "days"])
    assert compare_contains(model, gold, order_sensitive=False).match
    assert not compare_results(model, gold, order_sensitive=False).match


def test_contains_rejects_extra_rows():
    # Extra COLUMNS are credited; extra ROWS are not (e.g. full ranking vs top-1).
    model = _df([["SP", 100.0], ["RJ", 50.0], ["MG", 25.0]], ["state", "revenue"])
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    res = compare_contains(model, gold, order_sensitive=False)
    assert not res.match
    assert "row count" in res.reason


def test_contains_rejects_missing_gold_column():
    model = _df([["SP"]], ["state"])
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    res = compare_contains(model, gold, order_sensitive=False)
    assert not res.match
    assert "fewer columns" in res.reason


def test_contains_rejects_wrong_values_even_with_extra_column():
    model = _df([["SP", 999.0, 5]], ["state", "revenue", "orders"])
    gold = _df([["SP", 100.0]], ["state", "revenue"])
    assert not compare_contains(model, gold, order_sensitive=False).match


def test_contains_off_by_one_count_still_fails():
    model = _df([[1235]], ["n"])
    gold = _df([[1234]], ["n"])
    assert not compare_contains(model, gold, order_sensitive=False).match


def test_contains_ranking_respects_order():
    gold = _df([["SP", 100.0], ["RJ", 50.0]], ["state", "revenue"])
    right = _df([["SP", 100.0, 5], ["RJ", 50.0, 3]], ["state", "revenue", "orders"])
    wrong = _df([["RJ", 50.0, 3], ["SP", 100.0, 5]], ["state", "revenue", "orders"])
    assert compare_contains(right, gold, order_sensitive=True).match
    assert not compare_contains(wrong, gold, order_sensitive=True).match
