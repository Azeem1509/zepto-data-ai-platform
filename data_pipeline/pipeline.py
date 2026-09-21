###Performs scraping, cleaning, baseline INR conversion, normalized SQLite insertion, and query execution.



import os
import re
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- CONSTANTS ---
BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR = 105.50  # Fixed project baseline conversion rate
DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# --- TASK 1: SCRAPING ---
def scrape_books_data():
    """Scrapes at least 3 categories to get >= 60 books."""
    response = requests.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # Select 3 specific category links
    category_tags = soup.select(".side_categories ul li ul li a")[:3]
    categories_to_scrape = []
    
    for tag in category_tags:
        cat_name = tag.text.strip()
        cat_url = BASE_URL + tag["href"]
        categories_to_scrape.append({"name": cat_name, "url": cat_url})

    raw_books = []
    
    for cat in categories_to_scrape:
        cat_name = cat["name"]
        current_url = cat["url"]

        while current_url:
            res = requests.get(current_url)
            if res.status_code != 200:
                break
            
            cat_soup = BeautifulSoup(res.content, "html.parser")
            book_articles = cat_soup.find_all("article", class_="product_pod")

            for article in book_articles:
                # Title
                title_tag = article.h3.find("a")
                title = title_tag["title"] if "title" in title_tag.attrs else title_tag.text

                # Price (GBP)
                price_text = article.find("p", class_="price_color").text

                # Star Rating
                star_tag = article.find("p", class_="star-rating")
                rating_class = [c for c in star_tag["class"] if c != "star-rating"][0]

                # Availability
                avail_text = article.find("p", class_="instock availability").text.strip()

                raw_books.append({
                    "title": title,
                    "price_raw": price_text,
                    "rating_raw": rating_class,
                    "availability_raw": avail_text,
                    "category_name": cat_name
                })

            # Handle pagination inside category if present
            next_tag = cat_soup.find("li", class_="next")
            if next_tag:
                next_page_rel = next_tag.find("a")["href"]
                current_url = current_url.rsplit("/", 1)[0] + "/" + next_page_rel
            else:
                current_url = None

    print(f"[Scraper] Successfully scraped {len(raw_books)} books across {len(categories_to_scrape)} categories.")
    return raw_books

# --- TASK 2: CLEANING & ENRICHMENT ---
def clean_and_transform(raw_books):
    """Cleans fields, imputes missing values, and converts currency."""
    df = pd.DataFrame(raw_books)

    # 1. Strip currency symbol & convert price to float
    df["price_gbp"] = df["price_raw"].str.replace(r"[^\d.]", "", regex=True).astype(float)

    # Impute missing/unexpected price values with median if any fail
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # 2. Map word rating to integer (1-5)
    df["rating"] = df["rating_raw"].map(RATING_MAP).fillna(3).astype(int)

    # 3. Parse availability to boolean integer (1 = True, 0 = False)
    df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False).astype(int)

    # 4. Fixed Baseline Conversion: GBP to INR (1 GBP = 105.50 INR)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_name"]]

# --- TASK 3: NORMALIZE & LOAD INTO SQLITE ---
def create_and_load_db(df):
    """Creates a 2-table PK/FK schema and populates clean data."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Table 1: categories (PK)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
    """)

    # Table 2: books (FK referencing categories)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
    """)

    # Insert distinct categories
    unique_categories = df["category_name"].unique()
    for cat in unique_categories:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat,))

    conn.commit()

    # Map category names to category_id
    cat_lookup = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)
    df_merged = df.merge(cat_lookup, on="category_name")

    # Insert books
    for _, row in df_merged.iterrows():
        cursor.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]), int(row["in_stock"]), int(row["category_id"])))

    conn.commit()
    conn.close()
    print(f"[Database] SQLite DB successfully created and populated at '{DB_PATH}'.")

# --- TASK 4: EXECUTE SQL QUERIES ---
def run_sql_queries():
    """Executes required SQL queries demonstrating key SQL clauses."""
    conn = sqlite3.connect(DB_PATH)
    
    queries = {
        "Query 1 (SELECT/WHERE/BETWEEN)": """
            SELECT title, price_gbp, price_inr, rating 
            FROM books 
            WHERE price_gbp BETWEEN 20.0 AND 40.0;
        """,
        "Query 2 (DISTINCT/ORDER BY)": """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating DESC;
        """,
        "Query 3 (IN/LIMIT)": """
            SELECT title, price_inr, rating 
            FROM books 
            WHERE rating IN (4, 5) 
            ORDER BY price_inr DESC 
            LIMIT 5;
        """,
        "Query 4 (WHERE/ORDER BY)": """
            SELECT title, price_gbp 
            FROM books 
            WHERE in_stock = 1 
            ORDER BY price_gbp ASC 
            LIMIT 5;
        """,
        "Query 5 (JOIN across tables)": """
            SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating >= 4
            ORDER BY b.price_inr DESC
            LIMIT 10;
        """
    }

    print("\n" + "="*50)
    print(" EXECUTING REQUIRED SQL QUERIES ")
    print("="*50)

    for name, query in queries.items():
        print(f"\n--- {name} ---")
        df_result = pd.read_sql(query, conn)
        print(df_result.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    raw_data = scrape_books_data()
    clean_df = clean_and_transform(raw_data)
    create_and_load_db(clean_df)
    run_sql_queries()
