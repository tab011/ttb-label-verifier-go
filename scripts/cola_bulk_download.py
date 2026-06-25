#!/usr/bin/env python3
"""
TTB COLA bulk downloader — captures ALL registry fields including Applicant Name,
Serial No, DSP permit, approval date, and origin for the fraud-detection matrix.

Strategy:
  1. Use Selenium (attached Firefox) to perform a search and establish a session.
  2. Call publicSaveSearchResultsToFile.do with those session cookies to pull
     the full CSV — TTB generates it server-side with every column.
  3. Save to testdata/cola_whisky.csv (inside the project, survives reboots).

Usage:
    python3 scripts/cola_bulk_download.py [--query bourbon] [--date-from 01/01/2020]
"""

import argparse
import csv
import io
import ssl
import time
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "vscode/webdriver-tools"))
from attach import attach

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SEARCH_URL    = "https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do"
BULK_URL      = "https://www.ttbonline.gov/colasonline/publicSaveSearchResultsToFile.do"
OUT_DIR       = Path(__file__).parent.parent / "testdata"
COLA_CSV_OUT  = OUT_DIR / "cola_whisky.csv"

# Fields we care about from the TTB bulk CSV.
# The actual CSV has ~20 columns; we keep the ones useful for compliance + fraud detection.
KEEP_FIELDS = [
    "TTB ID",
    "Serial Number",
    "Permit Number",        # DSP / Basic Permit number of the applicant
    "Applicant Name",       # Who filed — cross-reference for fraud matrix
    "Brand Name",
    "Fanciful Name",
    "Class/Type Code",
    "Class/Type Desc",
    "Origin",               # State or country of origin
    "Approved Date",
    "Status",
]


def wait_for(driver, css, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, css))
    )


def do_search(driver, query: str, date_from: str, date_to: str):
    driver.get(SEARCH_URL)
    wait_for(driver, 'input[name="searchCriteria.dateCompletedFrom"]')
    driver.find_element(By.NAME, "searchCriteria.dateCompletedFrom").clear()
    driver.find_element(By.NAME, "searchCriteria.dateCompletedFrom").send_keys(date_from)
    driver.find_element(By.NAME, "searchCriteria.dateCompletedTo").clear()
    driver.find_element(By.NAME, "searchCriteria.dateCompletedTo").send_keys(date_to)
    pn = driver.find_element(By.NAME, "searchCriteria.productOrFancifulName")
    pn.clear()
    pn.send_keys(query)
    driver.find_element(By.CSS_SELECTOR, 'input[type="submit"][value="Search"]').click()
    time.sleep(3)
    print(f"Search submitted: '{query}' from {date_from} to {date_to}")


def bulk_download(driver) -> bytes:
    """POST to the bulk-save endpoint using the live session cookies."""
    # Pull session cookies from Selenium
    selenium_cookies = driver.get_cookies()
    cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in selenium_cookies)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        BULK_URL,
        method="GET",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; TTB-scraper/1.0)",
            "Cookie":     cookie_header,
            "Referer":    SEARCH_URL,
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = resp.read()
    print(f"Bulk download: {len(data):,} bytes received")
    return data


def parse_and_filter(raw_bytes: bytes) -> tuple[list[str], list[dict]]:
    """Parse the TTB CSV and return (headers, rows) keeping only KEEP_FIELDS."""
    # TTB exports in latin-1 with Windows line endings
    text = raw_bytes.decode("latin-1", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    raw_headers = reader.fieldnames or []
    print(f"TTB CSV columns ({len(raw_headers)}): {raw_headers}")

    # Map actual column names to our canonical names (TTB column names vary slightly)
    col_map = {}
    for keep in KEEP_FIELDS:
        for h in raw_headers:
            if keep.lower() in h.lower():
                col_map[keep] = h
                break

    print(f"Matched columns: {list(col_map.keys())}")

    rows = []
    for row in reader:
        mapped = {k: row.get(v, "").strip() for k, v in col_map.items()}
        rows.append(mapped)

    return list(col_map.keys()), rows


def write_csv(headers: list[str], rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows):,} records → {path}")


def print_summary(rows: list[dict]):
    from collections import Counter
    class_counts = Counter(r.get("Class/Type Desc", "").strip() for r in rows)
    applicant_counts = Counter(r.get("Applicant Name", "").strip() for r in rows)
    origin_counts = Counter(r.get("Origin", "").strip() for r in rows)

    print(f"\nTop class types:")
    for ct, n in class_counts.most_common(5):
        print(f"  {n:4d}  {ct}")

    print(f"\nTop applicants (potential parent-company matrix seed):")
    for ap, n in applicant_counts.most_common(10):
        print(f"  {n:4d}  {ap}")

    print(f"\nTop origins:")
    for og, n in origin_counts.most_common(8):
        print(f"  {n:4d}  {og}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query",     default="bourbon",     help="Product/fanciful name filter")
    ap.add_argument("--date-from", default="01/01/2020",  help="Approval date range start")
    ap.add_argument("--date-to",   default="06/25/2026",  help="Approval date range end")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Attaching to Firefox session...")
    driver = attach()

    do_search(driver, args.query, args.date_from, args.date_to)

    print("Pulling bulk CSV from TTB...")
    raw = bulk_download(driver)

    headers, rows = parse_and_filter(raw)

    if not rows:
        print("ERROR: No rows parsed. Check that the search returned results.")
        sys.exit(1)

    write_csv(headers, rows, COLA_CSV_OUT)
    print_summary(rows)

    # Quick fraud-matrix preview: applicant → brands mapping
    from collections import defaultdict
    applicant_brands: dict[str, set] = defaultdict(set)
    for row in rows:
        ap_name = row.get("Applicant Name", "").strip()
        brand   = row.get("Brand Name", "").strip() or row.get("Fanciful Name", "").strip()
        if ap_name and brand:
            applicant_brands[ap_name].add(brand)

    print(f"\nFraud matrix preview (applicant → brand count):")
    for ap, brands in sorted(applicant_brands.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {ap}: {len(brands)} brands — e.g. {sorted(brands)[:3]}")


if __name__ == "__main__":
    main()
