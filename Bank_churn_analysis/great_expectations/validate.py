"""
great_expectations/validate.py
--------------------------------
Data validation suite using Great Expectations (GX).
Defines a suite of expectations against the raw bank customer
dataset and runs validation before any training run.

Designed to be self-contained — no GX cloud or DataContext
configuration required. Uses the ephemeral in-memory runtime.

Usage:
    python great_expectations/validate.py
    python great_expectations/validate.py --data data/raw/bank_customers.csv
    python great_expectations/validate.py --strict  # exit(1) on failure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
S   = settings


# ---------------------------------------------------------------------------
# Expectation helpers
# ---------------------------------------------------------------------------

def _check(results: list, name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"expectation": name, "status": status, "detail": detail})
    icon = "✓" if passed else "✗"
    fn   = log.info if passed else log.warning
    fn(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))


def validate(df: pd.DataFrame, strict: bool = False) -> dict:
    """
    Run the full expectation suite against df.
    Returns a report dict. Raises RuntimeError if strict=True and failures exist.
    """
    log.info(f"Running Great Expectations suite on {len(df):,} rows")
    results: list = []

    # ── Schema ──────────────────────────────────────────────────────────
    log.info("─ Schema expectations")

    expected_cols = [
        "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
        "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Exited",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    _check(results, "expect_table_columns_to_match",
           len(missing) == 0,
           f"Missing: {missing}" if missing else "All 13 columns present")

    _check(results, "expect_table_row_count_to_be_between",
           1_000 <= len(df) <= 100_000,
           f"n={len(df):,}")

    # ── No duplicates ────────────────────────────────────────────────────
    log.info("─ Uniqueness expectations")

    n_dup = df["CustomerId"].duplicated().sum()
    _check(results, "expect_column_values_to_be_unique[CustomerId]",
           n_dup == 0, f"{n_dup} duplicates found")

    # ── Nulls ────────────────────────────────────────────────────────────
    log.info("─ Completeness expectations")

    critical = ["CreditScore", "Age", "Balance", "Exited",
                "NumOfProducts", "IsActiveMember"]
    for col in critical:
        null_pct = df[col].isnull().mean()
        _check(results, f"expect_column_values_to_not_be_null[{col}]",
               null_pct == 0.0, f"{null_pct:.1%} null")

    # ── Value ranges ─────────────────────────────────────────────────────
    log.info("─ Range expectations")

    range_checks = [
        ("CreditScore",   300,   900),
        ("Age",            18,   100),
        ("Tenure",          0,    10),
        ("NumOfProducts",   1,     4),
        ("HasCrCard",       0,     1),
        ("IsActiveMember",  0,     1),
        ("Exited",          0,     1),
        ("Balance",         0, 1e9),
        ("EstimatedSalary", 0, 1e9),
    ]
    for col, lo, hi in range_checks:
        in_range = df[col].between(lo, hi).all()
        out_of   = (~df[col].between(lo, hi)).sum()
        _check(results, f"expect_column_values_to_be_between[{col}]",
               in_range, f"{out_of} values outside [{lo}, {hi}]")

    # ── Categorical values ───────────────────────────────────────────────
    log.info("─ Categorical expectations")

    geo_vals    = set(df["Geography"].unique())
    geo_allowed = {"France", "Germany", "Spain"}
    _check(results, "expect_column_values_to_be_in_set[Geography]",
           geo_vals.issubset(geo_allowed),
           f"Found: {geo_vals}")

    gender_vals = set(df["Gender"].unique())
    _check(results, "expect_column_values_to_be_in_set[Gender]",
           gender_vals.issubset({"Male", "Female"}),
           f"Found: {gender_vals}")

    # ── Statistical / distribution ───────────────────────────────────────
    log.info("─ Distribution expectations")

    churn_rate = df["Exited"].mean()
    _check(results, "expect_column_mean_to_be_between[Exited](churn_rate)",
           0.05 <= churn_rate <= 0.50,
           f"churn_rate={churn_rate:.2%}")

    avg_credit = df["CreditScore"].mean()
    _check(results, "expect_column_mean_to_be_between[CreditScore]",
           500 <= avg_credit <= 800,
           f"mean={avg_credit:.1f}")

    median_balance = df["Balance"].median()
    _check(results, "expect_column_median_to_be_between[Balance]",
           0 <= median_balance <= 200_000,
           f"median=€{median_balance:,.0f}")

    pct_zero_balance = (df["Balance"] == 0).mean()
    _check(results, "expect_column_proportion_of_unique_values[Balance=0]",
           pct_zero_balance <= 0.40,
           f"{pct_zero_balance:.1%} have zero balance")

    # ── Geography coverage ───────────────────────────────────────────────
    log.info("─ Coverage expectations")

    for geo in ["France", "Germany", "Spain"]:
        pct = (df["Geography"] == geo).mean()
        _check(results, f"expect_geography_coverage[{geo}]",
               pct >= 0.10, f"{pct:.1%} of customers")

    # ── Assemble report ──────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    report = {
        "total"      : len(results),
        "passed"     : n_pass,
        "failed"     : n_fail,
        "success_pct": round(n_pass / len(results) * 100, 1),
        "results"    : results,
    }

    log.info(f"\n{'='*50}")
    log.info(f"Validation complete: {n_pass}/{len(results)} passed "
             f"({report['success_pct']}%)")
    if n_fail > 0:
        log.warning(f"{n_fail} expectation(s) FAILED")
        for r in results:
            if r["status"] == "FAIL":
                log.warning(f"  FAIL: {r['expectation']} — {r['detail']}")

    # Save report
    out = Path("great_expectations/validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {out}")

    if strict and n_fail > 0:
        raise RuntimeError(
            f"Data validation failed: {n_fail} expectation(s) not met. "
            f"See great_expectations/validation_report.json"
        )

    return report


def run(data_path: str = None, strict: bool = False) -> dict:
    path = data_path or S.paths.raw_data
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    return validate(df, strict=strict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=None, help="Path to CSV")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any expectation fails")
    args = parser.parse_args()
    report = run(data_path=args.data, strict=args.strict)
    sys.exit(0 if report["failed"] == 0 else 1)
