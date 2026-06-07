# ============================================================
# run_all_queries.py — Execute All SQL Analyses
# ============================================================
# Connects to data/processed/instacart.db, runs the three
# analytical SQL files in order, prints a summary of each
# result, and saves outputs to outputs/reports/ as CSVs.
#
# Each SQL file is a single SELECT statement with CTEs,
# so cursor.execute() handles them cleanly — no splitting
# or multi-statement juggling needed.
# ============================================================

import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd


def check_db():
    if not os.path.exists(config.DB_PATH):
        print(f"\n  ERROR: Database not found at {config.DB_PATH}")
        print("  Run sql/00_create_database.py first.")
        sys.exit(1)


def run_sql_file(conn, sql_path, label, output_csv):
    """
    Execute a single SQL file, return results as a DataFrame,
    print a summary, and save to CSV.
    """
    if not os.path.exists(sql_path):
        print(f"\n  ERROR: {sql_path} not found.")
        return None

    with open(sql_path, 'r', encoding='utf-8') as f:
        sql = f.read().strip()

    # Strip trailing semicolon if present — sqlite3 cursor.execute()
    # doesn't want it for SELECT queries
    if sql.endswith(';'):
        sql = sql[:-1].strip()

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        df   = pd.DataFrame(rows, columns=cols)

        # Save to CSV
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        df.to_csv(output_csv, index=False)

        print(f"\n  {label}")
        print(f"  {'-' * 52}")
        print(f"  Rows returned:  {len(df):,}")
        print(f"  Columns:        {len(df.columns)}")
        print(f"  Saved:          {os.path.basename(output_csv)}")
        print()

        # Print the result table — truncate wide DataFrames for readability
        display = df.copy()
        for col in display.select_dtypes('object').columns:
            display[col] = display[col].astype(str).str[:20]
        pd.set_option('display.max_columns', 12)
        pd.set_option('display.width', 110)
        pd.set_option('display.float_format', '{:,.1f}'.format)
        print(display.to_string(index=False))

        return df

    except Exception as e:
        print(f"\n  ERROR running {os.path.basename(sql_path)}: {e}")
        return None


# -------------------------------------------------------
# Setup
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — SQL Query Runner")
print("=" * 60)

check_db()
print(f"\nConnecting to {config.DB_PATH}")
conn = sqlite3.connect(config.DB_PATH)
conn.row_factory = sqlite3.Row

# Confirm tables are there
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print(f"  Tables in DB: {', '.join(tables)}")

os.makedirs(config.REPORTS_DIR, exist_ok=True)

print("\nRunning queries...")


# -------------------------------------------------------
# Query 1 — RFM cohort analysis
# -------------------------------------------------------
rfm_results = run_sql_file(
    conn,
    sql_path   = config.SQL_RFM,
    label      = "01 — RFM Cohort Analysis",
    output_csv = os.path.join(config.REPORTS_DIR, "01_rfm_cohort.csv"),
)


# -------------------------------------------------------
# Query 2 — Shrinkage risk by segment
# -------------------------------------------------------
shrink_results = run_sql_file(
    conn,
    sql_path   = config.SQL_SHRINKAGE,
    label      = "02 — Basket Shrinkage by Segment",
    output_csv = os.path.join(config.REPORTS_DIR, "02_shrinkage_by_segment.csv"),
)


# -------------------------------------------------------
# Query 3 — Category exposure Pareto
# -------------------------------------------------------
cat_results = run_sql_file(
    conn,
    sql_path   = config.SQL_CATEGORY,
    label      = "03 — Category Revenue Pareto",
    output_csv = os.path.join(config.REPORTS_DIR, "03_category_pareto.csv"),
)


# -------------------------------------------------------
# Summary
# -------------------------------------------------------
conn.close()

print("\n" + "=" * 60)
print("ALL QUERIES COMPLETE")
print("=" * 60)

report_files = [f for f in os.listdir(config.REPORTS_DIR) if f.endswith('.csv')]
for f in sorted(report_files):
    path = os.path.join(config.REPORTS_DIR, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {f:<40} {size_kb:>6.1f} KB")

print(f"\n  All reports saved to: {config.REPORTS_DIR}")
