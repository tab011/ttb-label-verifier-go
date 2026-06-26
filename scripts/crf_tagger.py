#!/usr/bin/env python3
"""
CRF sequence tagger for TTB alcohol label OCR output.

Replaces the hand-tuned emission weights in hmm_tagger.py with learned
L-BFGS weights via sklearn-crfsuite.  Advantages over the HMM:
  - Discriminative: models P(states|observations) directly.
  - No independence assumption: all features interact freely.
  - BIO tagging: handles multi-line entities (I-WARN continuation lines).
  - Learned weights: derived from auto-labeled training data, not hand-tuned.

BIO tag set:
    B-BRAND  I-BRAND   — brand name (first / continuation lines)
    B-TYPE   I-TYPE    — class/type designation
    B-ABV               — alcohol by volume line
    B-NET               — net contents line
    B-WARN   I-WARN    — government warning (first / continuation lines)
    B-MFR    I-MFR     — manufacturer / bottler line
    O                   — other / decorative / noise

Usage:
    # Train from testdata labels and save model:
    python3 scripts/crf_tagger.py --train

    # Tag a single label image:
    python3 scripts/crf_tagger.py --image testdata/images/cola_0001.jpg

    # Compare CRF vs HMM accuracy on the test set:
    python3 scripts/crf_tagger.py --benchmark
"""

import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import sklearn_crfsuite
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
LABELS_CSV = ROOT / "testdata" / "labels.csv"
IMG_DIR    = ROOT / "testdata" / "images"
MODEL_PATH = ROOT / "testdata" / "crf_model.pkl"
MARKOV_JSON = ROOT / "testdata" / "brand_markov.json"

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
_RE_PCT = re.compile(r'\d+\.?\d*\s*%|alc\.|vol\.', re.IGNORECASE)
_RE_VOL = re.compile(r'\d+\.?\d*\s*(?:mL|L\b|fl\.?\s*oz|liters?)', re.IGNORECASE)
_RE_GOV = re.compile(r'^GOVERNMENT\s+WARNING', re.IGNORECASE)

_BOURBON_KW = {
    "BOURBON", "WHISKY", "WHISKEY", "VODKA", "GIN", "RUM",
    "TEQUILA", "BRANDY", "CORDIAL", "STRAIGHT", "BLENDED",
    "SCOTCH", "TENNESSEE", "SINGLE MALT", "TRIPLE SEC",
}
_WARNING_VOCAB = {
    "SURGEON", "PREGNANCY", "BIRTH DEFECTS", "ALCOHOLIC BEVERAGES",
    "OPERATE MACHINERY", "HEALTH PROBLEMS", "IMPAIRS", "CONSUMPTION",
    "ABILITY TO DRIVE",
}
_DISTILL_KW = {"DISTILL", "BOTTL", "BREW", "WINER", "IMPORT", "PRODUC"}

# Optional: load Markov chain for brand-name scoring feature
_markov = None

def _get_markov():
    global _markov
    if _markov is None and MARKOV_JSON.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from brand_markov import BrandMarkovChain
        _markov = BrandMarkovChain.from_dict(json.loads(MARKOV_JSON.read_text()))
    return _markov


def _allcaps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def line_features(lines: list[str], i: int) -> dict:
    """Feature dict for line i, with ±1 context window."""
    line  = lines[i]
    upper = line.upper()
    n     = len(lines)
    markov = _get_markov()

    f = {
        # Current line
        "is_allcaps":          _allcaps(line),
        "has_percent":         bool(_RE_PCT.search(line)),
        "has_volume":          bool(_RE_VOL.search(line)),
        "starts_gov_warning":  bool(_RE_GOV.match(line)),
        "has_bourbon_kw":      any(kw in upper for kw in _BOURBON_KW),
        "has_distill_kw":      any(kw in upper for kw in _DISTILL_KW),
        "has_warning_vocab":   any(kw in upper for kw in _WARNING_VOCAB),
        "has_digits":          any(c.isdigit() for c in line),
        "word_count_1":        len(line.split()) == 1,
        "word_count_2_3":      len(line.split()) in (2, 3),
        "word_count_4plus":    len(line.split()) >= 4,
        "char_count_bucket":   min(len(line) // 20, 5),   # 0-5 bucket
        "position_early":      i / max(1, n-1) < 0.25,
        "position_mid":        0.25 <= i / max(1, n-1) < 0.6,
        "position_late":       i / max(1, n-1) >= 0.6,
        "BOS":                 i == 0,
        "EOS":                 i == n - 1,
        # Markov brand score bucketed into 4 bins
        "markov_bucket":       _markov_bucket(markov, line),
    }

    # Previous line context
    if i > 0:
        prev = lines[i-1]
        f["prev.is_allcaps"]       = _allcaps(prev)
        f["prev.has_percent"]      = bool(_RE_PCT.search(prev))
        f["prev.has_volume"]       = bool(_RE_VOL.search(prev))
        f["prev.has_bourbon_kw"]   = any(kw in prev.upper() for kw in _BOURBON_KW)
        f["prev.starts_gov"]       = bool(_RE_GOV.match(prev))
        f["prev.has_warning_vocab"]= any(kw in prev.upper() for kw in _WARNING_VOCAB)
    else:
        f["prev.BOS"] = True

    # Next line context
    if i < n - 1:
        nxt = lines[i+1]
        f["next.has_percent"]     = bool(_RE_PCT.search(nxt))
        f["next.has_volume"]      = bool(_RE_VOL.search(nxt))
        f["next.starts_gov"]      = bool(_RE_GOV.match(nxt))
        f["next.is_allcaps"]      = _allcaps(nxt)
        f["next.has_bourbon_kw"]  = any(kw in nxt.upper() for kw in _BOURBON_KW)
    else:
        f["next.EOS"] = True

    return f


def _markov_bucket(markov, line: str) -> str:
    """Bucket Markov score into 4 named bins so CRF can weight it."""
    if markov is None:
        return "unknown"
    s = markov.score(line)
    if s > -1.0:
        return "high"
    if s > -2.5:
        return "medium"
    if s > -4.0:
        return "low"
    return "very_low"


def sentence_features(lines: list[str]) -> list[dict]:
    return [line_features(lines, i) for i in range(len(lines))]


# ---------------------------------------------------------------------------
# OCR runner
# ---------------------------------------------------------------------------
def run_tesseract(img_path: Path) -> list[str]:
    result = subprocess.run(
        ["tesseract", str(img_path), "stdout", "--psm", "4"],
        capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Auto-labeler: assigns BIO tags from ground-truth CSV
# ---------------------------------------------------------------------------
def auto_label(lines: list[str], gt: dict) -> list[str]:
    """
    Assign BIO tags to OCR lines given ground-truth field values.
    Uses fuzzy matching to align OCR output with expected field content.
    """
    tags = ["O"] * len(lines)

    # Government warning — identify by content match
    warn_started = False
    for i, line in enumerate(lines):
        upper = line.upper()
        if _RE_GOV.match(line):
            tags[i]      = "B-WARN"
            warn_started = True
        elif warn_started and any(kw in upper for kw in _WARNING_VOCAB):
            tags[i] = "I-WARN"
        elif warn_started and "O" not in tags[i]:
            pass  # already tagged
        elif warn_started:
            warn_started = False  # hit something unrelated — warning ended

    # ABV — line containing a percentage
    for i, line in enumerate(lines):
        if tags[i] != "O":
            continue
        abv_gt = gt.get("abv_percent", "")
        if _RE_PCT.search(line) and abv_gt and abv_gt in line.replace(" ", ""):
            tags[i] = "B-ABV"
        elif _RE_PCT.search(line) and "%" in line:
            # Even without exact match, a % line in context is ABV
            tags[i] = "B-ABV"

    # Net contents
    for i, line in enumerate(lines):
        if tags[i] != "O":
            continue
        if _RE_VOL.search(line):
            tags[i] = "B-NET"

    # Brand name and class/type — fuzzy match against ground truth
    brand_gt = gt.get("brand_name", "").strip().upper()
    type_gt  = gt.get("class_type", "").strip().upper()

    brand_scored = []
    type_scored  = []
    for i, line in enumerate(lines):
        if tags[i] != "O":
            continue
        if brand_gt:
            brand_scored.append((fuzz.ratio(brand_gt, line.upper()), i))
        if type_gt:
            type_scored.append((fuzz.partial_ratio(type_gt, line.upper()), i))

    if brand_scored:
        best_score, best_i = max(brand_scored)
        if best_score >= 60:
            tags[best_i] = "B-BRAND"

    if type_scored:
        best_score, best_i = max(type_scored)
        if best_score >= 60 and tags[best_i] == "O":
            tags[best_i] = "B-TYPE"

    # Manufacturer heuristic: allcaps line with distillery keyword before BRAND
    brand_idx = next((i for i, t in enumerate(tags) if t == "B-BRAND"), len(lines))
    for i, line in enumerate(lines):
        if i >= brand_idx:
            break
        if tags[i] == "O" and _allcaps(line) and any(k in line.upper() for k in _DISTILL_KW):
            tags[i] = "B-MFR"

    return tags


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_training_data() -> tuple[list, list]:
    """Run Tesseract on all training images and auto-label each."""
    X, y = [], []
    with open(LABELS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = fail = 0
    for row in rows:
        img = IMG_DIR / row["filename"]
        if not img.exists():
            continue
        lines = run_tesseract(img)
        if not lines:
            fail += 1
            continue
        tags = auto_label(lines, row)
        X.append(sentence_features(lines))
        y.append(tags)
        ok += 1

    print(f"Training data: {ok} images labelled, {fail} skipped")
    return X, y


def train(X, y) -> sklearn_crfsuite.CRF:
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.05,   # L1 regularisation — sparsity
        c2=0.1,    # L2 regularisation — smoothing
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X, y)
    return crf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class LabelCRF:
    def __init__(self, model: sklearn_crfsuite.CRF):
        self.model = model

    def decode_tokens(self, lines: list[str]) -> list[tuple[str, str]]:
        if not lines:
            return []
        feats = sentence_features(lines)
        tags  = self.model.predict([feats])[0]
        return list(zip(lines, tags))

    def decode(self, ocr_text: str) -> dict:
        lines  = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        tagged = self.decode_tokens(lines)

        fields: dict[str, list[str]] = {k: [] for k in
            ["brand_name", "class_type", "abv_percent", "net_contents",
             "government_warning", "manufacturer"]}

        tag_map = {
            "B-BRAND": "brand_name",   "I-BRAND": "brand_name",
            "B-TYPE":  "class_type",   "I-TYPE":  "class_type",
            "B-ABV":   "abv_percent",
            "B-NET":   "net_contents",
            "B-WARN":  "government_warning", "I-WARN": "government_warning",
            "B-MFR":   "manufacturer",       "I-MFR":  "manufacturer",
        }
        for line, tag in tagged:
            key = tag_map.get(tag)
            if key:
                fields[key].append(line)

        return {k: " ".join(v).strip() for k, v in fields.items()}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "LabelCRF":
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model)


def save_model(crf: sklearn_crfsuite.CRF, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(crf, f)
    print(f"CRF model saved → {path}")


# ---------------------------------------------------------------------------
# Benchmark: compare CRF vs HMM on the test set
# ---------------------------------------------------------------------------
def benchmark():
    if not MODEL_PATH.exists():
        print("No trained model found — run with --train first.")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent))
    from hmm_tagger import LabelHMM

    crf_tagger = LabelCRF.load()
    hmm_tagger = LabelHMM()

    with open(LABELS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    crf_hits = hmm_hits = total = 0

    for row in rows:
        img = IMG_DIR / row["filename"]
        if not img.exists():
            continue
        total += 1
        ocr = "\n".join(run_tesseract(img))

        # CRF
        crf_fields = crf_tagger.decode(ocr)
        # HMM
        hmm_fields = hmm_tagger.decode(ocr)

        gt_brand = row["brand_name"].strip().upper()
        gt_type  = row["class_type"].strip().upper()

        crf_brand_ok = fuzz.ratio(gt_brand, crf_fields.get("brand_name","").upper()) >= 70
        hmm_brand_ok = fuzz.ratio(gt_brand, hmm_fields.get("brand_name","").upper()) >= 70
        crf_type_ok  = fuzz.partial_ratio(gt_type, crf_fields.get("class_type","").upper()) >= 70
        hmm_type_ok  = fuzz.partial_ratio(gt_type, hmm_fields.get("class_type","").upper()) >= 70

        if crf_brand_ok and crf_type_ok:
            crf_hits += 1
        if hmm_brand_ok and hmm_type_ok:
            hmm_hits += 1

    print(f"\nBenchmark on {total} labels (brand + type both correct):")
    print(f"  CRF: {crf_hits}/{total} = {100*crf_hits//total}%")
    print(f"  HMM: {hmm_hits}/{total} = {100*hmm_hits//total}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",     action="store_true", help="Train CRF from testdata labels")
    ap.add_argument("--image",     help="Tag a single image with the trained CRF")
    ap.add_argument("--benchmark", action="store_true", help="Compare CRF vs HMM accuracy")
    ap.add_argument("--extract",   action="store_true", help="Read OCR text from stdin, print extracted fields as JSON")
    args = ap.parse_args()

    if args.train:
        X, y = build_training_data()
        print(f"Training CRF on {len(X)} sequences...")
        crf = train(X, y)
        save_model(crf)

        # Show learned top features
        print("\nTop CRF feature weights (B-WARN):")
        for fname, weight in sorted(
            crf.state_features_.items(), key=lambda x: -abs(x[1])
        )[:15]:
            attr, tag = fname
            if tag == "B-WARN":
                print(f"  {weight:+.3f}  {attr}")

    elif args.benchmark:
        benchmark()

    elif args.image:
        if not MODEL_PATH.exists():
            print("No trained model found — run with --train first.")
            sys.exit(1)
        tagger = LabelCRF.load()
        lines  = run_tesseract(Path(args.image))
        tagged = tagger.decode_tokens(lines)

        print("Token-level BIO tags (CRF):")
        print(f"  {'TOKEN':<50} TAG")
        print(f"  {'-'*50} ---")
        for line, tag in tagged:
            print(f"  {line:<50} {tag}")

        print()
        fields = tagger.decode("\n".join(lines))
        print("Extracted fields:")
        for k, v in fields.items():
            if v:
                print(f"  {k:<22} {v[:70]}")

    elif args.extract:
        if not MODEL_PATH.exists():
            json.dump({"error": "model not found – run --train first"}, sys.stdout)
            sys.exit(1)
        ocr_text = sys.stdin.read()
        tagger = LabelCRF.load()
        fields = tagger.decode(ocr_text)
        json.dump(fields, sys.stdout)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
