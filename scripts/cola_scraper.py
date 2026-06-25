#!/usr/bin/env python3
"""
Scrape TTB COLA Public Registry for bourbon label images and field data.

Strategy:
  - Extract brand_name and class_type directly from the results table (no detail page visit)
  - Visit detail page only to locate the label image URL
  - Download images and write labels.csv

Usage:
    python3 scripts/cola_scraper.py [--max 50] [--query "bourbon"]
    Output: testdata/images/*.jpg  +  testdata/labels.csv
"""

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / "vscode/webdriver-tools"))
from attach import attach

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SEARCH_URL = "https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do"
BASE_URL   = "https://ttbonline.gov/colasonline/"
OUT_DIR    = Path(__file__).parent.parent / "testdata"
IMG_DIR    = OUT_DIR / "images"
CSV_PATH   = OUT_DIR / "labels.csv"
FIELDNAMES = ["filename", "brand_name", "class_type", "abv_percent", "net_contents"]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TTB-scraper/1.0)"}


def wait_for(driver, css, timeout=8):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, css))
    )


def do_search(driver, query: str, date_from: str = "01/01/2024"):
    driver.get(SEARCH_URL)
    wait_for(driver, 'input[name="searchCriteria.dateCompletedFrom"]')
    driver.find_element(By.NAME, "searchCriteria.dateCompletedFrom").clear()
    driver.find_element(By.NAME, "searchCriteria.dateCompletedFrom").send_keys(date_from)
    driver.find_element(By.NAME, "searchCriteria.dateCompletedTo").clear()
    driver.find_element(By.NAME, "searchCriteria.dateCompletedTo").send_keys("06/24/2026")
    pn = driver.find_element(By.NAME, "searchCriteria.productOrFancifulName")
    pn.clear()
    pn.send_keys(query)
    driver.find_element(By.CSS_SELECTOR, 'input[type="submit"][value="Search"]').click()
    time.sleep(2)


def parse_results_table(driver) -> list[dict]:
    """Extract brand name, class type, and detail URL from the results table rows."""
    rows = []
    # Result table rows have 10 columns; col 0 = TTB ID link, 4 = fanciful, 5 = brand, 8 = class code, 9 = class desc
    for a in driver.find_elements(By.CSS_SELECTOR, 'a[href*="viewColaDetails"]'):
        href = a.get_attribute("href")
        ttb_id = a.text.strip()
        # Get the parent row
        try:
            row_el = a.find_element(By.XPATH, "./ancestor::tr")
            tds = row_el.find_elements(By.TAG_NAME, "td")
            if len(tds) >= 10:
                brand      = tds[5].text.strip() or tds[4].text.strip()  # brand or fanciful
                class_desc = tds[9].text.strip()
            else:
                brand, class_desc = "", ""
        except Exception:
            brand, class_desc = "", ""

        rows.append({
            "ttb_id":     ttb_id,
            "detail_url": href,
            "brand_name": brand,
            "class_type": class_desc,
        })
    return rows


def has_next_page(driver) -> bool:
    links = driver.find_elements(By.PARTIAL_LINK_TEXT, "Next")
    return bool(links)


def go_next_page(driver):
    driver.find_element(By.PARTIAL_LINK_TEXT, "Next").click()
    time.sleep(1.5)


def find_image_url(driver, detail_url: str) -> str | None:
    driver.get(detail_url)
    time.sleep(1)
    src = driver.page_source

    # TTBOnline serves label images via getImgPdf or similar endpoints
    for pattern in ["getImgPdf", "getImg", "getLabelImage", "colaImage"]:
        idx = src.lower().find(pattern.lower())
        if idx >= 0:
            # Extract the full URL from surrounding context
            chunk = src[max(0, idx-10):idx+200]
            import re
            m = re.search(r'(?:href|src)=["\']([^"\']+' + pattern + r'[^"\']*)["\']', chunk, re.IGNORECASE)
            if m:
                url = m.group(1)
                if not url.startswith("http"):
                    url = "https://ttbonline.gov/colasonline/" + url.lstrip("/")
                return url

    # Also check for inline images
    for img in driver.find_elements(By.TAG_NAME, "img"):
        s = img.get_attribute("src") or ""
        if s and "ttbonline" in s and "gif" not in s.lower() and len(s) > 40:
            return s

    # Check all links for image files
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href") or ""
        if any(href.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".pdf"]):
            return href

    return None


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            dest.write_bytes(r.read())
        return dest.stat().st_size > 1000  # reject tiny/empty files
    except Exception as e:
        print(f"    download failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="bourbon", help="Product name search (default: bourbon)")
    ap.add_argument("--max", type=int, default=50, help="Max labels to collect (default: 50)")
    ap.add_argument("--date-from", default="01/01/2024", help="Date range start (default: 01/01/2024)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    print("Attaching to Firefox...")
    driver = attach()

    print(f"Searching for: '{args.query}' from {args.date_from}")
    do_search(driver, args.query, args.date_from)

    # Collect up to args.max result rows across pages
    all_rows = []
    page = 1
    while len(all_rows) < args.max:
        page_rows = parse_results_table(driver)
        if not page_rows:
            print(f"  page {page}: no results found")
            break
        need = args.max - len(all_rows)
        all_rows.extend(page_rows[:need])
        print(f"  page {page}: {len(page_rows)} results — total so far: {len(all_rows)}")
        if len(all_rows) >= args.max or not has_next_page(driver):
            break
        go_next_page(driver)
        page += 1

    print(f"\nCollected {len(all_rows)} results. Fetching images...")

    csv_rows = []
    for i, row in enumerate(all_rows, 1):
        fname = f"cola_{i:04d}.jpg"
        img_path = IMG_DIR / fname
        print(f"[{i}/{len(all_rows)}] {row['brand_name']} ({row['class_type']})")

        img_url = find_image_url(driver, row["detail_url"])
        got_image = False
        if img_url:
            print(f"    image: {img_url[:80]}")
            got_image = download(img_url, img_path)
        else:
            print("    no image found — skipping image")

        csv_rows.append({
            "filename":    fname if got_image else "",
            "brand_name":  row["brand_name"],
            "class_type":  row["class_type"],
            "abv_percent": "",   # on label image, extracted by vision agent
            "net_contents": "",  # on label image, extracted by vision agent
        })

    # Write CSV (include rows even without images for reference)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(csv_rows)

    with_images = sum(1 for r in csv_rows if r["filename"])
    print(f"\nDone. {with_images}/{len(csv_rows)} labels with images.")
    print(f"  CSV:    {CSV_PATH}")
    print(f"  Images: {IMG_DIR}/")


if __name__ == "__main__":
    main()
