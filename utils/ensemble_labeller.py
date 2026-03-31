"""
Ensemble Labeller — Daraz Reviews

"""

import os, re, time, json, csv
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")  # LLaMA
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")  # GPT-OSS
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")  # Qwen

INPUT_FILE    = "daraz_reviews_cleaned2.csv"
OUTPUT_FILE   = "..data/daraz_reviews_labelled.csv"
PROGRESS_FILE = "labelling_progress.json"

DAILY_LIMIT  = 950   # per key — stop before hitting 1000/day
MAX_RETRIES  = 3
BACKOFF_BASE = 30    # seconds — doubles on each retry
ROW_DELAY    = 3.5   # seconds between rows — 3 parallel calls so only need ~3.5s gap

VALID_LABELS   = ["customer_service", "delivery", "product_feedback" ]
PRIORITY_ORDER = ["customer_service", "delivery", "product_feedback" ]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a review intent classifier for a Nepali e-commerce platform (Daraz Nepal).
Reviews may be in English, Romanized Nepali, or a mix of both.

Classify into EXACTLY ONE label using this priority order (assign highest that applies):
1. customer_service  — return/refund/exchange, seller not responding, warranty, complaint
2. delivery          — delivery speed, packaging, wrong item, courier
3. product_feedback  — quality, sound, battery, size, defects, brand comparison

EXAMPLES:
"Product is good but delivery was late"           → delivery
"Support didn't respond and refund delayed"       → customer_service
"Battery is bad"                                  → product_feedback
"Return garnu paryako, product nai ramro chaina"  → customer_service

Reply with ONLY one of these exact words — nothing else:
customer_service | delivery | product_feedback | recommendation"""

# ─────────────────────────────────────────────────────────────────────────────
# LABEL EXTRACTOR — handles reasoning/thinking model output
# ─────────────────────────────────────────────────────────────────────────────

def extract_label(text: str) -> str | None:
    if not text:
        return None
    t = text.strip().lower()

    # Exact match — ideal case
    if t in VALID_LABELS:
        return t

    # Strip <think>...</think> block (Qwen3) then check again
    t2 = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    if t2 in VALID_LABELS:
        return t2

    # Scan full response, return LAST occurrence (after reasoning chain)
    positions = []
    for label in VALID_LABELS:
        for pattern in [label, label.replace("_", " ")]:
            for m in re.finditer(re.escape(pattern), t):
                positions.append((m.start(), label))
    if not positions:
        return None
    positions.sort(key=lambda x: x[0], reverse=True)
    return positions[0][1]

# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            p = json.load(f)
        if p.get("date") != str(date.today()):
            p["date"]         = str(date.today())
            p["daily_counts"] = {"llama": 0, "gpt_oss": 0, "qwen": 0}
            print("New day — daily counts reset")
        return p
    return {
        "date":         str(date.today()),
        "done_ids":     [],
        "daily_counts": {"llama": 0, "gpt_oss": 0, "qwen": 0},
    }

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# GENERIC CALLER with retry + backoff
# ─────────────────────────────────────────────────────────────────────────────

def call_model(client: Groq, params: dict, model_key: str,
               progress: dict) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(**params)
            raw   = completion.choices[0].message.content or ""
            label = extract_label(raw)
            if label:
                progress["daily_counts"][model_key] += 1
                return label
            print(f"\n  [{model_key}] Bad response: '{raw[:60]}'")
            return None
        except RateLimitError:
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"\n  [{model_key}] Rate limit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"\n  [{model_key} error] {e}")
            return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL MODEL PARAMS
# ─────────────────────────────────────────────────────────────────────────────

def llama_params(review: str) -> dict:
    return {
        "model":       "llama-3.3-70b-versatile",
        "messages":    [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": review}],
        "temperature": 0,
        "max_tokens":  20,
        "stream":      False,
    }

def gpt_oss_params(review: str) -> dict:
    return {
        "model":                 "openai/gpt-oss-120b",
        "messages":              [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",   "content": review}],
        "temperature":           0,
        "max_completion_tokens": 300,
        "reasoning_effort":      "low",
        "stream":                False,
    }

def qwen_params(review: str) -> dict:
    return {
        "model":                 "qwen/qwen3-32b",
        "messages":              [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",   "content": review}],
        "temperature":           0,
        "max_completion_tokens": 500,
        "stream":                False,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL CALL — all 3 models at the same time
# ─────────────────────────────────────────────────────────────────────────────

def call_all_parallel(review: str, c1: Groq, c2: Groq, c3: Groq,
                      progress: dict) -> tuple:
    """
    Fires all 3 model calls simultaneously using threads.
    Returns (llama_label, gpt_oss_label, qwen_label).
    Each model uses its own client/key so no rate limit conflict.
    """
    results = {"llama": None, "gpt_oss": None, "qwen": None}

    tasks = [
        ("llama",   c1, llama_params(review)),
        ("gpt_oss", c2, gpt_oss_params(review)),
        ("qwen",    c3, qwen_params(review)),
    ]

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(call_model, client, params, key, progress): key
            for key, client, params in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    # Save progress once after all 3 complete
    save_progress(progress)

    return results["llama"], results["gpt_oss"], results["qwen"]

# ─────────────────────────────────────────────────────────────────────────────
# VOTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_final_label(votes: list) -> tuple:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None, 1
    counts = Counter(valid)
    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_label, 0       # majority or unanimous → accepted
    for p in PRIORITY_ORDER:      # all differ → priority tiebreak, flag
        if p in valid:
            return p, 1
    return valid[0], 1

# ─────────────────────────────────────────────────────────────────────────────
# SAVE ROW — appends immediately, one row at a time
# ─────────────────────────────────────────────────────────────────────────────

def save_row(row_data: dict):
    write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    pd.DataFrame([row_data]).to_csv(
        OUTPUT_FILE, index=False, encoding="utf-8-sig",
        mode="a", header=write_header
    )

# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_test(c1, c2, c3, progress) -> bool:
    print("Testing all 3 models in parallel...\n")
    review = "Battery drains too fast, not worth the price"
    l, g, q = call_all_parallel(review, c1, c2, c3, progress)
    print(f"  llama   → {l}")
    print(f"  gpt_oss → {g}")
    print(f"  qwen    → {q}\n")
    return any(v is not None for v in [l, g, q])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(df: pd.DataFrame, c1, c2, c3, progress):
    done_ids  = set(progress["done_ids"])
    remaining = df[~df["id"].isin(done_ids)].reset_index(drop=True)
    counts    = progress["daily_counts"]

    print(f"Total rows     : {len(df)}")
    print(f"Already done   : {len(done_ids)}")
    print(f"Remaining      : {len(remaining)}")
    print(f"Daily counts   : llama={counts['llama']}  gpt_oss={counts['gpt_oss']}  qwen={counts['qwen']}")
    est_mins = len(remaining) * ROW_DELAY / 60
    print(f"Estimated time : ~{est_mins:.0f} min  (parallel calls)\n")

    row_count = 0
    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Labelling"):

        c = progress["daily_counts"]
        if any(c[k] >= DAILY_LIMIT for k in ["llama", "gpt_oss", "qwen"]):
            print(f"\nDaily limit reached after {row_count} rows. Run again tomorrow.")
            break

        review = str(row["review_text"])

        # All 3 models called at the same time
        llama_label, gpt_oss_label, qwen_label = call_all_parallel(
            review, c1, c2, c3, progress
        )

        final_label, needs_review = get_final_label(
            [llama_label, gpt_oss_label, qwen_label]
        )

        save_row({
            "id":               row["id"],
            "review_text":      review,
            "source":           row.get("source", "daraz.com.np"),
            "product_category": row.get("product_category", ""),
            "rating":           row.get("rating", ""),
            "label":            final_label,
            "needs_review":     needs_review,
        })

        progress["done_ids"].append(int(row["id"]))
        save_progress(progress)
        row_count += 1

        time.sleep(ROW_DELAY)

    print(f"\nSession done — {row_count} rows labelled this run.")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.exists(OUTPUT_FILE):
        return
    df = pd.read_csv(OUTPUT_FILE)
    if df.empty:
        return
    total = len(df)
    print(f"\n{'='*50}")
    print(f"Rows labelled      : {total}")
    print(f"needs_review = 1   : {df['needs_review'].sum()}  ← check these manually")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = [k for k in ["GROQ_API_KEY_1","GROQ_API_KEY_2","GROQ_API_KEY_3"]
               if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing in .env: {', '.join(missing)}")
        exit(1)

    c1 = Groq(api_key=GROQ_API_KEY_1)  # LLaMA
    c2 = Groq(api_key=GROQ_API_KEY_2)  # GPT-OSS
    c3 = Groq(api_key=GROQ_API_KEY_3)  # Qwen

    progress = load_progress()

    if not run_test(c1, c2, c3, progress):
        print("All models failed. Check your API keys.")
        exit(1)

    print("Models OK. Starting...\n")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows\n")

    run_ensemble(df, c1, c2, c3, progress)
    print_summary()

    done  = len(progress["done_ids"])
    total = len(df)
    if done < total:
        print(f"\n{total - done} rows remaining — run again tomorrow.")
    else:
        print(f"\nAll {total} rows done! Output → {OUTPUT_FILE}")
        flagged = pd.read_csv(OUTPUT_FILE)
        flagged = flagged[flagged["needs_review"] == 1]
        if not flagged.empty:
            flagged.to_csv("needs_manual_review.csv", index=False, encoding="utf-8-sig")
            print(f"{len(flagged)} flagged rows → needs_manual_review.csv")