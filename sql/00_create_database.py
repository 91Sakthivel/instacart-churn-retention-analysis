# ============================================================
# 00_create_database.py — Build SQLite Database
# ============================================================
# Loads all 6 raw CSVs + the 4 pipeline-generated processed
# tables into a single SQLite database at data/processed/.
#
# Why load both raw and processed?
#   Raw tables let you run from-scratch SQL to prove the
#   analysis can live entirely in a warehouse.
#   Processed tables (rfm_segments, customer_features, etc.)
#   let the analytical SQL queries in 01–03 run in seconds
#   instead of minutes — they're the indexed source of truth.
#
# The order_products table (33.8M rows) uses executemany()
# with explicit transactions — about 3x faster than to_sql()
# for large SQLite loads.
# ============================================================

import sys
import os
import sqlite3
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np


def check_file(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: {path} not found.")
        sys.exit(1)


# -------------------------------------------------------
# Verify all source files exist before touching the DB
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — 00 Build SQLite Database")
print("=" * 60)

print("\nChecking source files...")
raw_files = [
    config.ORDERS, config.ORDER_PRODUCTS_PRIOR, config.ORDER_PRODUCTS_TRAIN,
    config.PRODUCTS, config.AISLES, config.DEPARTMENTS,
]
processed_files = [
    config.RFM_SEGMENTS, config.CUSTOMER_FEATURES,
    config.INTERVENTION_REPORT, config.CATEGORY_RISK_REPORT,
]
for f in raw_files + processed_files:
    check_file(f)
    print(f"  OK  {os.path.basename(f)}")


# -------------------------------------------------------
# Connect — create fresh DB (overwrite if exists)
# -------------------------------------------------------
os.makedirs(config.PROCESSED_DIR, exist_ok=True)

if os.path.exists(config.DB_PATH):
    os.remove(config.DB_PATH)
    print(f"\nRemoved existing database.")

conn = sqlite3.connect(config.DB_PATH)
# Tune SQLite for bulk loading — WAL mode + large cache
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA synchronous  = NORMAL")
conn.execute("PRAGMA cache_size   = -64000")   # 64MB page cache
conn.execute("PRAGMA temp_store   = MEMORY")
print(f"Database created: {config.DB_PATH}")


# -------------------------------------------------------
# Helper — load a small CSV directly via to_sql
# -------------------------------------------------------
def load_small_table(csv_path, table_name, conn, dtype=None):
    t0 = time.time()
    df = pd.read_csv(csv_path, dtype=dtype)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    elapsed = time.time() - t0
    print(f"  {table_name:<30} {len(df):>10,} rows  ({elapsed:.1f}s)")
    return len(df)


# -------------------------------------------------------
# Helper — load a large CSV in chunks via executemany.
#   executemany() with an explicit transaction is roughly
#   3x faster than to_sql() for multi-million row tables.
# -------------------------------------------------------
def load_large_table_chunked(csv_path, table_name, conn,
                              create_sql, insert_sql,
                              dtype=None, chunksize=500_000):
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(create_sql)
    conn.execute("BEGIN")

    cursor     = conn.cursor()
    total_rows = 0
    t0         = time.time()

    for chunk in pd.read_csv(csv_path, dtype=dtype, chunksize=chunksize):
        # fillna for any nullable integer columns before converting to tuples
        records = chunk.where(pd.notna(chunk), None).itertuples(index=False, name=None)
        cursor.executemany(insert_sql, records)
        total_rows += len(chunk)
        if total_rows % 2_000_000 == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
            elapsed = time.time() - t0
            print(f"    ...{total_rows:,} rows  ({elapsed:.0f}s)")

    conn.execute("COMMIT")
    elapsed = time.time() - t0
    print(f"  {table_name:<30} {total_rows:>10,} rows  ({elapsed:.0f}s)")
    return total_rows


# -------------------------------------------------------
# 1. Small lookup tables — load directly
# -------------------------------------------------------
print("\nLoading raw tables...")

load_small_table(config.AISLES,       'aisles',       conn)
load_small_table(config.DEPARTMENTS,  'departments',  conn)
load_small_table(config.PRODUCTS,     'products',     conn)
load_small_table(config.ORDERS,       'orders',       conn,
                 dtype={'user_id': 'int32', 'order_id': 'int32', 'order_number': 'int16'})


# -------------------------------------------------------
# 2. order_products — train first (establishes schema),
#    then prior appended in chunks.
#
#    We combine prior + train into a single order_products
#    table so SQL queries don't need to UNION them.
# -------------------------------------------------------
print("  Loading order_products (train + prior combined)...")
print("  Prior file is 577MB — this is the slow step, ~3–5 min")

OP_CREATE = """
    CREATE TABLE order_products (
        order_id           INTEGER,
        product_id         INTEGER,
        add_to_cart_order  INTEGER,
        reordered          INTEGER
    )
"""
OP_INSERT = "INSERT INTO order_products VALUES (?, ?, ?, ?)"
OP_DTYPE  = {'order_id': 'int32', 'product_id': 'int32',
             'add_to_cart_order': 'int16', 'reordered': 'int8'}

# Load train first
conn.execute("DROP TABLE IF EXISTS order_products")
conn.execute(OP_CREATE)
conn.execute("BEGIN")
cursor = conn.cursor()

t0 = time.time()
train_df = pd.read_csv(config.ORDER_PRODUCTS_TRAIN, dtype=OP_DTYPE)
cursor.executemany(OP_INSERT, train_df.itertuples(index=False, name=None))
conn.execute("COMMIT")
train_rows = len(train_df)
del train_df
print(f"    train: {train_rows:,} rows loaded")

# Now append prior in chunks
conn.execute("BEGIN")
prior_rows = 0
for chunk in pd.read_csv(config.ORDER_PRODUCTS_PRIOR, dtype=OP_DTYPE,
                         chunksize=config.CHUNK_SIZE):
    cursor.executemany(OP_INSERT, chunk.itertuples(index=False, name=None))
    prior_rows += len(chunk)
    if prior_rows % 5_000_000 == 0:
        conn.execute("COMMIT")
        conn.execute("BEGIN")
        elapsed = time.time() - t0
        print(f"    prior: {prior_rows:,} rows  ({elapsed:.0f}s)")

conn.execute("COMMIT")
elapsed = time.time() - t0
total_op = train_rows + prior_rows
print(f"  {'order_products':<30} {total_op:>10,} rows  ({elapsed:.0f}s)")


# -------------------------------------------------------
# 3. Processed pipeline outputs — load as tables too.
#    These are the fast path for SQL queries in 01–03.
# -------------------------------------------------------
print("\nLoading processed tables...")

load_small_table(config.RFM_SEGMENTS,       'rfm_segments',       conn)
load_small_table(config.CUSTOMER_FEATURES,  'customer_features',  conn)
load_small_table(config.INTERVENTION_REPORT,'intervention_report',conn)
load_small_table(config.CATEGORY_RISK_REPORT,'category_risk_report',conn)


# -------------------------------------------------------
# 4. Indexes — critical for query performance.
#    Without these, joins on order_products (33M rows)
#    would do full table scans.
# -------------------------------------------------------
print("\nCreating indexes...")

indexes = [
    ("idx_orders_user",      "CREATE INDEX idx_orders_user     ON orders(user_id)"),
    ("idx_orders_order",     "CREATE INDEX idx_orders_order    ON orders(order_id)"),
    ("idx_op_order",         "CREATE INDEX idx_op_order        ON order_products(order_id)"),
    ("idx_op_product",       "CREATE INDEX idx_op_product      ON order_products(product_id)"),
    ("idx_products_dept",    "CREATE INDEX idx_products_dept   ON products(department_id)"),
    ("idx_rfm_user",         "CREATE INDEX idx_rfm_user        ON rfm_segments(user_id)"),
    ("idx_rfm_segment",      "CREATE INDEX idx_rfm_segment     ON rfm_segments(segment)"),
    ("idx_cf_user",          "CREATE INDEX idx_cf_user         ON customer_features(user_id)"),
    ("idx_cf_shrinking",     "CREATE INDEX idx_cf_shrinking    ON customer_features(shrinking_flag)"),
]
for name, sql in indexes:
    t0 = time.time()
    conn.execute(sql)
    conn.commit()
    print(f"  {name:<30} ({time.time() - t0:.1f}s)")


# -------------------------------------------------------
# 5. Verify — row counts for all tables
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DATABASE SUMMARY")
print("=" * 60)

tables = [
    'aisles', 'departments', 'products', 'orders',
    'order_products',
    'rfm_segments', 'customer_features',
    'intervention_report', 'category_risk_report',
]
for tbl in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:<30} {count:>12,} rows")

db_size_mb = os.path.getsize(config.DB_PATH) / 1e6
print(f"\n  Database file size: {db_size_mb:,.0f} MB")
print(f"  Location: {config.DB_PATH}")

conn.close()

print()
print("Next up — run sql/run_all_queries.py")
