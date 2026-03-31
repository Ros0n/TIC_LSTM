"""
Daraz Nepal Review Collector
============================
Collects product reviews from Daraz Nepal API for intent classification dataset.

Output:
    daraz_reviews_dataset.csv  — with columns: id, review_text, source, product_category, rating, label
"""

import requests
import pandas as pd
import time
import json
import random
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# How many pages to fetch per product  (each page = up to 20 reviews)
PAGES_PER_PRODUCT = 3

# Delay between requests in seconds (be polite to the server)
REQUEST_DELAY = 1.5

# Output file
OUTPUT_FILE = "daraz_reviews_dataset.csv"

# ─────────────────────────────────────────────
# PRODUCTS TO SCRAPE
# (item_id, human-readable category label)
# Add more item IDs from daraz.com.np as needed.
# You can find the item ID in the product URL: ...i{ITEM_ID}-s...
# ─────────────────────────────────────────────

PRODUCTS = [
    # ── Electronics ──────────────────────────
    {"item_id": "157384253",  "category": "Electronics"},  
    {"item_id": "05494427",  "category": "Electronics"},   
    {"item_id": "115339034",  "category": "Electronics"},   
    {"item_id": "111626830",  "category": "Electronics"},     

    # ── Fashion and Clothing ────────────────────
    {"item_id": "127806127",  "category": "Fashion and Clothing"},      
    {"item_id": "136740311",  "category": "Fashion and Clothing"},      
    {"item_id": "128040605",  "category": "Fashion and Clothing"},      
    {"item_id": "110983051",  "category": "Fashion and Clothing"},      

]

# ─────────────────────────────────────────────
# HTTP HEADERS  (mimic a real browser)
# ─────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.daraz.com.np/",
    "x-requested-with": "XMLHttpRequest",
}

BASE_URL = "https://my.daraz.com.np/pdp/review/getReviewList"

# ─────────────────────────────────────────────
# HELPER: fetch one page of reviews
# ─────────────────────────────────────────────

def fetch_reviews(item_id: str, page: int, page_size: int = 20,
                  filter_val: int = 0, sort_val: int = 0) -> list[dict]:
    """
    Fetch a single page of reviews for a product.
    Returns a list of raw review dicts, or [] on failure.

    filter values (observed from Daraz UI):
        0 = All reviews
        1 = With images
        2 = Positive  (≥4 stars)
        3 = Critical  (≤2 stars)

    sort values:
        0 = Default / Relevance
        1 = Most recent
        2 = Most helpful
    """
    params = {
        "itemId":   item_id,
        "pageSize": page_size,
        "filter":   filter_val,
        "sort":     sort_val,
        "pageNo":   page,
    }
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("model", {}).get("items", [])
        return items if items else []
    except requests.exceptions.RequestException as e:
        log.warning(f"  ✗ Request failed (item={item_id}, page={page}): {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        log.warning(f"  ✗ JSON parse error (item={item_id}, page={page}): {e}")
        return []


# ─────────────────────────────────────────────
# HELPER: extract clean fields from raw review
# ─────────────────────────────────────────────

def parse_review(raw: dict, category: str, item_id: str) -> dict | None:
    """Extract only the fields we need for the dataset."""
    text = (raw.get("reviewContent") or "").strip()
    if not text:
        return None

    return {
        "review_id":       raw.get("reviewRateId"),
        "review_text":     text,
        "source":          "daraz.com.np",
        "product_category": category,
        "rating":          raw.get("rating"),
        "label":           None,   # ← to be filled during annotation
    }




FETCH_STRATEGIES = [
    # ── Per-star filters (recent sort) ─────────────────
    {"filter": 5, "sort": 1, "desc": "5★ / recent"},
    {"filter": 4, "sort": 1, "desc": "4★ / recent"},
    {"filter": 3, "sort": 1, "desc": "3★ / recent"},
    {"filter": 2, "sort": 1, "desc": "2★ / recent"},
    {"filter": 1, "sort": 1, "desc": "1★ / recent"},
    # ── All-stars sorted passes (catch stragglers) ──────
    {"filter": 0, "sort": 2, "desc": "all / high→low"},
    {"filter": 0, "sort": 3, "desc": "all / low→high"},
]


# ─────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ─────────────────────────────────────────────

def collect_all() -> pd.DataFrame:
    all_rows: list[dict] = []
    seen_ids: set = set()   # deduplicate by reviewRateId

    for product in PRODUCTS:
        item_id  = product["item_id"]
        category = product["category"]

        log.info(f"\n{'='*60}")
        log.info(f"Product: {item_id}  |  Category: {category}")

        for strategy in FETCH_STRATEGIES:
            log.info(f"  Strategy: {strategy['desc']}")
            consecutive_empty = 0

            for page in range(1, PAGES_PER_PRODUCT + 1):
                raw_reviews = fetch_reviews(
                    item_id,
                    page=page,
                    filter_val=strategy["filter"],
                    sort_val=strategy["sort"],
                )

                if not raw_reviews:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        log.info(f"    → No more reviews after page {page - 1}, stopping.")
                        break
                    continue

                consecutive_empty = 0
                new_count = 0

                for raw in raw_reviews:
                    rid = raw.get("reviewRateId")
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    parsed = parse_review(raw, category, item_id)
                    if parsed:
                        all_rows.append(parsed)
                        new_count += 1

                log.info(f"    Page {page:02d}: +{new_count} new reviews (total so far: {len(all_rows)})")

                # Polite delay with slight jitter to avoid rate limiting
                time.sleep(REQUEST_DELAY + random.uniform(0.2, 0.8))

    df = pd.DataFrame(all_rows)
    return df


# ─────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        log.warning("No reviews collected!")
        return df

    # Reset index → becomes the "id" column
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index + 1)

    # Keep only the required output columns (+ extras for context)
    output_cols = [
        "id",
        "review_text",
        "source",
        "product_category",
        "rating",
        "label",
    ]
    df = df[[c for c in output_cols if c in df.columns]]

    # Basic stats
    log.info(f"\n{'='*60}")
    log.info(f"Total reviews collected : {len(df)}")
    log.info(f"Unique categories       : {df['product_category'].nunique()}")
    log.info("\nRating distribution:")
    log.info(df["rating"].value_counts().sort_index().to_string())
    log.info("\nCategory distribution:")
    log.info(df["product_category"].value_counts().to_string())

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Daraz Nepal review collection...")
    log.info(f"Products to scrape : {len(PRODUCTS)}")
    log.info(f"Pages per product  : {PAGES_PER_PRODUCT} × {len(FETCH_STRATEGIES)} strategies")
    log.info(f"Request delay      : {REQUEST_DELAY}s\n")

    df = collect_all()
    df = post_process(df)

    if not df.empty:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        log.info(f"\n✓ Saved {len(df)} reviews → {OUTPUT_FILE}")
    else:
        log.error("No data collected. Check your internet connection or product IDs.")