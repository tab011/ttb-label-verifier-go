# TTB Alcohol Label Verifier

Automated compliance verification for TTB (Alcohol and Tobacco Tax and Trade Bureau)
alcohol labels. Upload a label image, get a PASS / FAIL verdict with per-field
breakdown in under one second — fully offline, no cloud API required.

Built by **Todd Baker** as a prototype for the U.S. Department of the Treasury,
demonstrating how OCR + probabilistic sequence modeling can replace the 5–10 minute
manual field-matching process currently used during label review.

**[Live demo](https://ttb-label-verifier-go.fly.dev) &nbsp;·&nbsp; [Technical report (PDF)](https://ttb-label-verifier-go.fly.dev/static/baker-ttb-technical-report.pdf)**

---

## Quick start (local)

### System dependencies

**Fedora / RHEL:**
```bash
sudo dnf install opencv opencv-devel tesseract tesseract-langpack-eng golang
```

**Ubuntu / Debian:**
```bash
sudo apt-get install libopencv-dev tesseract-ocr tesseract-ocr-eng golang
```

**macOS (Homebrew):**
```bash
brew install opencv tesseract go
```

### Build and run

```bash
git clone https://github.com/tab011/ttb-label-verifier-go
cd ttb-label-verifier-go

make setup          # installs Python deps, generates test data, trains models
make run            # builds Go server and opens :8181
```

Open `http://localhost:8181`.

---

## Architecture

```
Label image (upload or batch CSV)
    │
    ▼
GoCV preprocessing
  MSER region crop → grayscale → Gaussian blur → Otsu binarize → 2× cubic upscale
    │
    ▼
Tesseract OCR  (--psm 4, ~0.65 s/label)
    │
    ▼
CRF sequence tagger  (sklearn-crfsuite, BIO tags, learned weights)
  B-BRAND  I-BRAND  B-TYPE  B-ABV  B-NET  B-WARN  I-WARN  B-MFR  O
    │
    ├── HMM tagger (Viterbi decoder + Forward likelihood score — fallback / comparison)
    │
    └── Markov chain brand scorer  (character bigram anomaly detection)
    │
    ▼
Comparison engine  (27 CFR §§ 5.36, 5.121, 16.21)
  brand_name     → Levenshtein ≥ 90 %
  class_type     → fuzzy ≥ 85 %
  abv_percent    → ±0.1 % float tolerance; BIB = exactly 50.0 %
  net_contents   → fuzzy ≥ 85 %
  government_warning → content ≥ 88 % AND ≥ 70 % uppercase letters (ALL CAPS required)
    │
    ▼
PASS / FAIL verdict  +  per-field table  +  anomaly scores
(JSON API → vanilla JS renders result; no framework dependencies)
```

---

## What the models do

### CRF tagger (primary field extractor)
A Linear-Chain CRF trained with L-BFGS on 30 annotated synthetic labels derived
from real TTB COLA registry records. Unlike the HMM (which uses hand-specified
transition/emission parameters), the CRF learns weights across all overlapping
features simultaneously: `is_allcaps`, `has_percent`, `has_bourbon_keyword`,
`prev.has_volume`, `markov_bucket`, position rank, ±1 context window, etc.

BIO tagging handles multi-line entities — government warning continuation lines
tag as `I-WARN` rather than falling through to `O`.

### HMM tagger (structural anomaly detection)
Runs in parallel with the CRF. The **Forward algorithm** sums over all possible
state-path combinations to produce `log P(token_sequence | HMM)`. A very
negative score (< −4.0 per token) means the label's token ordering does not
match any known bourbon label structure — useful for detecting scans of non-label
images, foreign-language labels, or fraudulent mock-ups.

The **Viterbi decoder** provides a second field extraction path benchmarked
against the CRF on each release.

### Brand Markov chain (counterfeit / OCR noise detection)
Character bigram model trained on 609 unique brand names from the TTB COLA
Public Registry (1,004 whisky records, 2020–2026). Scores the extracted brand
name against the transition matrix: scores near 0 indicate real bourbon naming
patterns; very negative scores flag OCR noise (`"BUFFAL0 TR4CE"`) or phonetic
counterfeits (`"Elijah Crais"`).

Threshold −3.5 triggers a ⚠ SUSPICIOUS annotation in the response JSON.

---

## API

### `POST /verify`

```
Content-Type: multipart/form-data

Fields:
  image          (required)  Label image file
  brand_name     (required)  Expected brand name from COLA registry
  class_type     (required)  Expected class/type designation
  abv_percent    (required)  Expected ABV as a decimal string ("45.0")
  net_contents   (required)  Expected net contents ("750ml")
```

Response:
```json
{
  "verdict": "PASS",
  "fields": {
    "brand_name":          { "status": "PASS", "extracted": "JIM BEAM", "expected": "JIM BEAM" },
    "class_type":          { "status": "PASS", "extracted": "STRAIGHT BOURBON WHISKY", "expected": "STRAIGHT BOURBON WHISKY" },
    "abv_percent":         { "status": "PASS", "extracted": "47.5", "expected": "47.5" },
    "net_contents":        { "status": "PASS", "extracted": "750ml", "expected": "750ml" },
    "government_warning":  { "status": "PASS", "extracted": "GOVERNMENT WARNING: ...", "expected": "..." }
  },
  "markov_score": -0.41,
  "hmm_likelihood": 1.50,
  "confidence": 0.4
}
```

### `GET /health`
Returns `{"status":"ok"}`.

---

## Regenerating test data

```bash
make regen-testdata   # 30 synthetic labels (20 PASS / 5 FAIL-warning / 5 FAIL-ABV)
make train-markov     # rebuild brand_markov.json
make train-crf        # retrain crf_model.pkl
make benchmark        # compare CRF vs HMM field extraction accuracy
```

---

## Deploying to Fly.io

```bash
fly launch --dockerfile Dockerfile --name ttb-label-verifier
fly deploy
```

The app requires no secrets. The GoCV OpenCV dependency is bundled via the
`gocv/opencv` base image in the Dockerfile.

---

## Architecture decisions

### Label region detection
The planned production approach was **MediaPipe TextDetector** — a lightweight,
purpose-built text detection model that returns individual text bounding boxes
directly from the image. Each bounding box would be fed to Tesseract independently,
giving the OCR engine a clean, isolated text region rather than the full label at
once. This is meaningfully better than OCR-on-the-whole-label because noise from
decorative elements, background texture, and unrelated regions never reaches the
OCR engine.

MediaPipe was chosen over YOLOv8 (which was the original plan) because YOLO is a
general object detector — it would find the bottle and infer the label location,
but it doesn't specifically find text regions. MediaPipe's TextDetector is purpose-
built for the task and considerably lighter.

The prototype ships **GoCV MSER** (Maximally Stable Extremal Regions) instead —
a purely geometric algorithm built into OpenCV that approximates text region
detection with zero model weights. It works well on flat synthetic labels but
degrades on real curved bottle photos where MediaPipe's learned detector would
handle perspective and distortion correctly. MSER was chosen for the prototype
because it has no external model dependency and keeps the binary fully offline.

Production path: replace `internal/preprocess/preprocess.go` MSER crop with
MediaPipe TextDetector output; the CRF tagger and compliance engine are
detector-agnostic.

## Compliance references

- 27 CFR § 5.32 — mandatory label information (brand name, class/type, ABV, net contents)
- 27 CFR § 5.36, § 5.121 — ABV ranges by class (BIB exactly 50 %; spirits 40–80 %)
- 27 CFR § 16.21 — government warning text and mandatory ALL CAPS

---

## Development note

This prototype was built with assistance from Claude Code (Anthropic). The
probabilistic models (CRF, HMM, Markov chain) and the compliance rule engine
were designed collaboratively; training data was generated from TTB-registered
brand names. The architecture decision to use offline OCR + sequence modeling
rather than a cloud vision API was driven by the government-firewall constraint
identified in the requirements.
