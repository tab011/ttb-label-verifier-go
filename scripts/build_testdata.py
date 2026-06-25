#!/usr/bin/env python3
"""
Build testdata for TTB label verifier.

1. Parse the bulk COLA CSV from /tmp/cola_whisky.csv → testdata/labels.csv
   (brand_name, class_type from registry; abv/contents blank — on image)

2. Generate synthetic PNG label images using Pillow for:
   - 20 PASS cases (all fields correct)
   - 5 FAIL-warning cases (warning in title case, not all-caps)
   - 5 FAIL-abv cases (ABV on label differs from CSV)

Run: python3 scripts/build_testdata.py
"""

import csv
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Prefer the project-local copy (survives reboots), fall back to /tmp download
_project_cola = Path(__file__).parent.parent / "testdata" / "cola_whisky.csv"
COLA_CSV   = _project_cola if _project_cola.exists() else Path("/tmp/cola_whisky.csv")
OUT_DIR    = Path(__file__).parent.parent / "testdata"
IMG_DIR    = OUT_DIR / "images"
CSV_OUT    = OUT_DIR / "labels.csv"

FIELDNAMES = ["filename", "brand_name", "class_type", "abv_percent", "net_contents"]

# 27 CFR § 16.21 requires the warning in ALL CAPS exactly as below.
GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT "
    "DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH "
    "DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO "
    "DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
)

# Title-case version — intentionally wrong for FAIL cases
GOV_WARNING_WRONG_CASE = (
    "Government Warning: (1) According to the Surgeon General, "
    "Women Should Not Drink Alcoholic Beverages During Pregnancy Because of "
    "the Risk of Birth Defects. (2) Consumption of Alcoholic Beverages Impairs "
    "Your Ability to Drive a Car or Operate Machinery, and May Cause Health Problems."
)

ABV_CHOICES = [40.0, 43.0, 45.0, 46.0, 47.5, 50.0, 51.5, 55.0, 57.5, 62.5]
NET_CHOICES = ["750ml", "1L", "375ml", "1750ml", "200ml"]


def load_font(size: int, mono: bool = False):
    # Sans-serif fonts OCR much better than monospace for numeric fields.
    # Monospace path kept as a fallback for systems with limited fonts.
    sans = [
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/gnu-free/FreeSans.ttf",
    ]
    mono_paths = [
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/gnu-free/FreeMono.ttf",
    ]
    paths = (mono_paths if mono else []) + sans + mono_paths
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_label(
    brand_name: str,
    class_type: str,
    abv: float,
    net: str,
    warning_text: str,
    img_path: Path,
    width: int = 900,
    height: int = 800,
):
    img = Image.new("RGB", (width, height), color=(245, 238, 220))
    draw = ImageDraw.Draw(img)

    font_large  = load_font(48)        # brand name — all-caps, sans-serif
    font_medium = load_font(32)        # class type, ABV
    font_net    = load_font(40)        # net contents — larger to survive OCR
    font_small  = load_font(17)        # government warning

    # Border
    draw.rectangle([10, 10, width-10, height-10], outline=(80, 40, 0), width=3)
    draw.rectangle([18, 18, width-18, height-18], outline=(80, 40, 0), width=1)

    y = 50
    # Brand name
    draw.text((width//2, y), brand_name.upper(), fill=(30, 10, 0),
              font=font_large, anchor="mt")
    y += 70

    # Class/type
    draw.text((width//2, y), class_type, fill=(60, 30, 0),
              font=font_medium, anchor="mt")
    y += 55

    # Divider
    draw.line([(50, y), (width-50, y)], fill=(80, 40, 0), width=2)
    y += 20

    # ABV
    draw.text((width//2, y), f"{abv:.1f}% Alc./Vol.", fill=(30, 10, 0),
              font=font_medium, anchor="mt")
    y += 55

    # Net contents — rendered larger so Tesseract reliably reads decimal values
    draw.text((width//2, y), net, fill=(30, 10, 0),
              font=font_net, anchor="mt")
    y += 65

    # Divider
    draw.line([(50, y), (width-50, y)], fill=(80, 40, 0), width=1)
    y += 20

    # Government warning — left-aligned with margin so Tesseract reads full line
    x_margin = 40
    lines = textwrap.wrap(warning_text, width=80)
    for line in lines:
        draw.text((x_margin, y), line, fill=(20, 20, 20), font=font_small)
        y += 22

    img.save(str(img_path), "JPEG", quality=92)


# Fallback brand list — used when COLA CSV is unavailable (e.g. after reboot)
_FALLBACK_BRANDS = [
    ("MAKER'S MARK", "STRAIGHT BOURBON WHISKY"),
    ("WILD TURKEY", "STRAIGHT BOURBON WHISKY"),
    ("BUFFALO TRACE", "STRAIGHT BOURBON WHISKY"),
    ("FOUR ROSES", "STRAIGHT BOURBON WHISKY"),
    ("KNOB CREEK", "STRAIGHT BOURBON WHISKY"),
    ("WOODFORD RESERVE", "STRAIGHT BOURBON WHISKY"),
    ("EVAN WILLIAMS", "STRAIGHT BOURBON WHISKY"),
    ("HEAVEN HILL", "STRAIGHT BOURBON WHISKY"),
    ("JIM BEAM", "STRAIGHT BOURBON WHISKY"),
    ("ELIJAH CRAIG", "STRAIGHT BOURBON WHISKY"),
    ("OLD FORESTER", "STRAIGHT BOURBON WHISKY"),
    ("LARCENY", "STRAIGHT BOURBON WHISKY"),
    ("BULLEIT", "STRAIGHT BOURBON WHISKY"),
    ("ANGEL'S ENVY", "STRAIGHT BOURBON WHISKY"),
    ("OLD DOMINICK", "STRAIGHT BOURBON WHISKY"),
    ("HENRY MCKENNA", "STRAIGHT BOURBON WHISKY"),
    ("BOOKER'S", "STRAIGHT BOURBON WHISKY"),
    ("BAKER'S", "STRAIGHT BOURBON WHISKY"),
    ("BASIL HAYDEN", "STRAIGHT BOURBON WHISKY"),
    ("ROWAN'S CREEK", "STRAIGHT BOURBON WHISKY"),
    ("NOAH'S MILL", "STRAIGHT BOURBON WHISKY"),
    ("KENTUCKY VINTAGE", "STRAIGHT BOURBON WHISKY"),
    ("1792 RIDGEMONT", "STRAIGHT BOURBON WHISKY"),
    ("VERY OLD BARTON", "STRAIGHT BOURBON WHISKY"),
    ("ANCIENT AGE", "STRAIGHT BOURBON WHISKY"),
    ("OLD CHARTER", "STRAIGHT BOURBON WHISKY"),
    ("W.L. WELLER", "STRAIGHT BOURBON WHISKY"),
    ("PAPPY VAN WINKLE", "STRAIGHT BOURBON WHISKY"),
    ("BUFFALO TRACE SELECT", "STRAIGHT BOURBON WHISKY"),
    ("EZRA BROOKS", "STRAIGHT BOURBON WHISKY"),
]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    # Load COLA CSV — fall back to built-in brand list if CSV unavailable
    if COLA_CSV.exists():
        raw = COLA_CSV.read_text("latin-1")
        all_rows = list(csv.DictReader(raw.splitlines()))
        bourbon_rows = [r for r in all_rows if "BOURBON" in (r.get("Class/Type Desc") or "")]
        print(f"COLA records loaded: {len(all_rows)} total, {len(bourbon_rows)} bourbon")
        brand_source = [
            (r["Brand Name"].strip() or r["Fanciful Name"].strip() or "UNKNOWN",
             r["Class/Type Desc"].strip())
            for r in bourbon_rows
        ]
    else:
        print(f"COLA CSV not found at {COLA_CSV} — using built-in brand list ({len(_FALLBACK_BRANDS)} brands)")
        brand_source = _FALLBACK_BRANDS * 2   # enough for 30 labels

    random.seed(42)
    random.shuffle(brand_source)

    csv_out_rows = []

    def add_label(i, brand, class_type, abv, net, warning, abv_expected, suffix=""):
        fname = f"cola_{i:04d}{suffix}.jpg"
        make_label(brand, class_type, abv, net, warning, IMG_DIR / fname)
        csv_out_rows.append({
            "filename":     fname,
            "brand_name":   brand,
            "class_type":   class_type,
            "abv_percent":  str(abv_expected),
            "net_contents": net,
        })

    idx = 1

    # 20 PASS cases — image matches the expected values exactly
    for brand, ct in brand_source[:20]:
        abv = random.choice(ABV_CHOICES)
        net = random.choice(NET_CHOICES)
        add_label(idx, brand, ct, abv, net, GOVERNMENT_WARNING, abv)
        idx += 1

    # 5 FAIL-warning cases — image has title-case warning
    for brand, ct in brand_source[20:25]:
        abv = random.choice(ABV_CHOICES)
        net = random.choice(NET_CHOICES)
        add_label(idx, brand, ct, abv, net, GOV_WARNING_WRONG_CASE, abv, suffix="_fail_warn")
        idx += 1

    # 5 FAIL-abv cases — image shows different ABV than CSV expected
    for brand, ct in brand_source[25:30]:
        abv_on_label = random.choice(ABV_CHOICES)
        abv_expected = round(abv_on_label + 5.0, 1)
        net = random.choice(NET_CHOICES)
        add_label(idx, brand, ct, abv_on_label, net, GOVERNMENT_WARNING, abv_expected, suffix="_fail_abv")
        idx += 1

    # Write CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(csv_out_rows)

    print(f"Generated {len(csv_out_rows)} label images → {IMG_DIR}/")
    print(f"CSV written to {CSV_OUT}")
    print(f"  PASS:         20")
    print(f"  FAIL-warning:  5")
    print(f"  FAIL-abv:      5")


if __name__ == "__main__":
    main()
