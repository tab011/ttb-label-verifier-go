#!/usr/bin/env python3
"""
Auto-annotate the real bottle photo dataset.

Workflow per image:
  1. Read Pascal VOC XML bounding box → crop bottle from full photo
  2. Preprocess: grayscale → Otsu binarize → 2× upscale
  3. Tesseract OCR on crop
  4. CRF sequence tagger extracts candidate fields
  5. Fuzzy-match brand name against COLA registry → fill missing fields
  6. Write to annotation CSV; flag rows needing human review

Output:
  dataset/annotations.csv         — all auto-annotated rows
  dataset/needs_review.csv        — rows below confidence threshold

Usage:
    python3 scripts/annotate_dataset.py --zip ~/Downloads/archive.zip \
        --cola testdata/labels.csv --out dataset/

    # After human review of needs_review.csv, merge and retrain:
    python3 scripts/annotate_dataset.py --merge dataset/ --retrain
"""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Bottle crop from Pascal VOC XML
# ---------------------------------------------------------------------------

def parse_voc_xml(xml_bytes: bytes) -> dict | None:
    """Return {'filename', 'xmin', 'ymin', 'xmax', 'ymax'} or None."""
    try:
        root = ET.fromstring(xml_bytes)
        obj = root.find("object")
        if obj is None:
            return None
        bb = obj.find("bndbox")
        return {
            "filename": root.findtext("filename", ""),
            "xmin": int(float(bb.findtext("xmin", "0"))),
            "ymin": int(float(bb.findtext("ymin", "0"))),
            "xmax": int(float(bb.findtext("xmax", "0"))),
            "ymax": int(float(bb.findtext("ymax", "0"))),
        }
    except Exception:
        return None


def crop_bottle(img_bytes: bytes, bb: dict, pad: int = 30) -> bytes:
    """Crop and preprocess bottle region from raw JPEG bytes."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    x1 = max(0, bb["xmin"] - pad)
    y1 = max(0, bb["ymin"] - pad)
    x2 = min(w, bb["xmax"] + pad)
    y2 = min(h, bb["ymax"] + pad)

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upscaled = cv2.resize(binary, (binary.shape[1] * 2, binary.shape[0] * 2),
                          interpolation=cv2.INTER_CUBIC)

    _, buf = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_tesseract(img_bytes: bytes) -> str:
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_bytes)
        tmp = f.name
    try:
        r = subprocess.run(
            ["tesseract", tmp, "stdout", "--psm", "4"],
            capture_output=True, text=True,
        )
        return r.stdout
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# CRF extraction
# ---------------------------------------------------------------------------

def crf_extract(ocr_text: str) -> dict:
    """Call the CRF tagger via --extract mode; fall back to empty dict on error."""
    crf_script = ROOT / "scripts" / "crf_tagger.py"
    if not crf_script.exists():
        return {}
    r = subprocess.run(
        ["python3", str(crf_script), "--extract"],
        input=ocr_text, capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# COLA registry lookup
# ---------------------------------------------------------------------------

def load_cola(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cola_lookup(brand_candidate: str, cola_records: list[dict], threshold: int = 80):
    """Fuzzy-match brand candidate against COLA registry. Returns (record, score) or (None, 0)."""
    if not cola_records or not brand_candidate:
        return None, 0
    names = [r["brand_name"] for r in cola_records]
    result = process.extractOne(brand_candidate.upper(), names,
                                scorer=fuzz.token_sort_ratio)
    if result is None:
        return None, 0
    matched_name, score, idx = result
    if score >= threshold:
        return cola_records[idx], score
    return None, score


# ---------------------------------------------------------------------------
# ABV extractor (from raw CRF output line)
# ---------------------------------------------------------------------------

_RE_ABV = re.compile(r"(\d+\.?\d*)\s*%", re.IGNORECASE)

def parse_abv(raw: str) -> str:
    m = _RE_ABV.search(raw)
    return m.group(1) if m else raw.strip()


# ---------------------------------------------------------------------------
# Main annotation pipeline
# ---------------------------------------------------------------------------

def annotate(zip_path: Path, cola_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cola = load_cola(cola_path)
    print(f"COLA registry: {len(cola)} records loaded")

    rows_all    = []
    rows_review = []

    with zipfile.ZipFile(zip_path) as zf:
        # Build index of XML annotations
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        img_names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        print(f"Found {len(xml_names)} annotations, {len(img_names)} images")

        for xml_name in sorted(xml_names):
            bb = parse_voc_xml(zf.read(xml_name))
            if not bb:
                continue
            stem = Path(bb["filename"]).stem
            img_key = img_names.get(stem)
            if not img_key:
                print(f"  [skip] no image for {bb['filename']}")
                continue

            print(f"  {bb['filename']} ...", end=" ", flush=True)

            try:
                img_bytes    = zf.read(img_key)
                cropped      = crop_bottle(img_bytes, bb)
                ocr_text     = run_tesseract(cropped)
                crf_fields   = crf_extract(ocr_text)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            brand_candidate = crf_fields.get("brand_name", "").strip()
            cola_rec, cola_score = cola_lookup(brand_candidate, cola)

            # Merge CRF output with COLA fill-in
            brand_name   = brand_candidate or (cola_rec["brand_name"]  if cola_rec else "")
            class_type   = crf_fields.get("class_type", "").strip() or (cola_rec.get("class_type", "") if cola_rec else "")
            abv_raw      = crf_fields.get("abv_percent", "").strip()
            abv_percent  = parse_abv(abv_raw) if abv_raw else (cola_rec.get("abv_percent", "") if cola_rec else "")
            net_contents = crf_fields.get("net_contents", "").strip() or (cola_rec.get("net_contents", "") if cola_rec else "")
            gov_warning  = crf_fields.get("government_warning", "").strip()

            # Confidence: COLA match score + whether CRF extracted something
            crf_ok  = bool(brand_candidate and class_type)
            confidence = round((cola_score * 0.6 + (40 if crf_ok else 0)) / 100, 2)
            needs_review = confidence < 0.55 or not brand_name

            row = {
                "filename":          bb["filename"],
                "brand_name":        brand_name,
                "class_type":        class_type,
                "abv_percent":       abv_percent,
                "net_contents":      net_contents,
                "government_warning": gov_warning,
                "cola_match_score":  cola_score,
                "confidence":        confidence,
                "needs_review":      "YES" if needs_review else "no",
                "ocr_raw":           ocr_text[:200].replace("\n", " | "),
            }
            rows_all.append(row)
            if needs_review:
                rows_review.append(row)
            print(f"brand={brand_name[:30]!r}  cola={cola_score}  conf={confidence}  review={'YES' if needs_review else 'no'}")

    fieldnames = ["filename","brand_name","class_type","abv_percent","net_contents",
                  "government_warning","cola_match_score","confidence","needs_review","ocr_raw"]

    all_csv = out_dir / "annotations.csv"
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_all)

    review_csv = out_dir / "needs_review.csv"
    with open(review_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_review)

    auto_ok = len(rows_all) - len(rows_review)
    print(f"\nDone: {len(rows_all)} images processed")
    print(f"  Auto-annotated (high confidence): {auto_ok}")
    print(f"  Needs human review:               {len(rows_review)}")
    print(f"\nOutputs:")
    print(f"  {all_csv}")
    print(f"  {review_csv}")


# ---------------------------------------------------------------------------
# Retrain CRF on merged annotations
# ---------------------------------------------------------------------------

def retrain(dataset_dir: Path):
    """
    Retrain the CRF using the merged annotations.csv as ground truth.
    Copies annotations.csv to testdata/labels.csv (keeping 20% as test set),
    then runs --train.
    """
    import random
    ann_csv = dataset_dir / "annotations.csv"
    if not ann_csv.exists():
        print(f"No annotations.csv found in {dataset_dir}")
        sys.exit(1)

    with open(ann_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("needs_review","YES") != "YES"]

    if not rows:
        print("No high-confidence rows to train on. Run annotation first.")
        sys.exit(1)

    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    test_rows  = rows[split:]

    train_csv = ROOT / "testdata" / "labels.csv"
    test_csv  = ROOT / "testdata" / "labels_test.csv"

    fieldnames = ["filename","brand_name","class_type","abv_percent","net_contents"]
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(train_rows)

    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(test_rows)

    print(f"Train set: {len(train_rows)} rows → {train_csv}")
    print(f"Test  set: {len(test_rows)} rows → {test_csv}")
    print("\nRetraining CRF...")

    r = subprocess.run(
        ["python3", str(ROOT / "scripts" / "crf_tagger.py"), "--train"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        print("CRF training failed.")
        sys.exit(1)

    print("\nRunning benchmark on test set...")
    subprocess.run(
        ["python3", str(ROOT / "scripts" / "crf_tagger.py"), "--benchmark"],
        cwd=ROOT,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip",    help="Path to bottle photo ZIP (archive.zip)")
    ap.add_argument("--cola",   default=str(ROOT / "testdata" / "labels.csv"),
                    help="Path to COLA registry CSV for brand fuzzy-matching")
    ap.add_argument("--out",    default=str(ROOT / "dataset"),
                    help="Output directory for annotation CSVs")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain CRF using dataset/annotations.csv (after human review)")
    ap.add_argument("--merge",  help="Dataset dir containing reviewed annotations.csv (for --retrain)")
    args = ap.parse_args()

    if args.retrain or args.merge:
        dataset_dir = Path(args.merge or args.out)
        retrain(dataset_dir)
    elif args.zip:
        annotate(Path(args.zip), Path(args.cola), Path(args.out))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
