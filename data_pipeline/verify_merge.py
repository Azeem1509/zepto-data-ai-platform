import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")

def verify_join_vs_merge():
    conn = sqlite3.connect(DB_PATH)

    # 1. SQL JOIN Execution
    sql_join_query = """
        SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY b.price_inr DESC
        LIMIT 10;
    """
    df_sql = pd.read_sql(sql_join_query, conn)

    # 2. Read full raw tables into pandas DataFrames using pd.read_sql
    df_books = pd.read_sql("SELECT * FROM books;", conn)
    df_categories = pd.read_sql("SELECT * FROM categories;", conn)
    conn.close()

    # 3. Reproduce JOIN directly in pandas using pd.merge
    df_merged = (
        pd.merge(df_books, df_categories, on="category_id")
        .query("rating >= 4")
        .sort_values(by="price_inr", ascending=False)
        [['title', 'category_name', 'price_gbp', 'price_inr', 'rating']]
        .head(10)
        .reset_index(drop=True)
    )

    print("\n" + "="*50)
    print(" SIDE-BY-SIDE COMPARISON: SQL JOIN vs PANDAS MERGE ")
    print("="*50)
    print("\n--- 1. pd.read_sql Output (SQL JOIN) ---")
    print(df_sql.to_string(index=False))

    print("\n--- 2. pd.merge Output (Pandas In-Memory) ---")
    print(df_merged.to_string(index=False))

    # Verify equivalence
    are_equal = df_sql.equals(df_merged)
    print(f"\nEquivalence Check (Outputs Match): {are_equal}")

if __name__ == "__main__":
    verify_join_vs_merge()
