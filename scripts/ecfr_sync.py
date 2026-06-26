#!/usr/bin/env python3
"""
Sync TTB-relevant CFR sections from the eCFR public API and store them locally.

Fetches:
  27 CFR Part 5  — Labeling and Advertising of Distilled Spirits
    § 5.32   Mandatory label information
    § 5.36   Government warning statement requirements
    § 5.121  General class and type designations
  27 CFR Part 16 — Alcoholic Beverage Health Warning Statement
    § 16.21  Mandatory label information (warning text)

Output: data/ecfr/{section}.json  (one file per section)
        data/ecfr/manifest.json   (sync date, version, change hashes)

Usage:
    python3 scripts/ecfr_sync.py              # sync to data/ecfr/
    python3 scripts/ecfr_sync.py --check      # print which sections changed
    python3 scripts/ecfr_sync.py --show 5.32  # print stored section text

Schedule: monthly (regulations change infrequently)
    systemd timer: scripts/ecfr-sync.timer
    or cron:  0 6 1 * * /path/to/python3 /path/to/ecfr_sync.py
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "ecfr"

# Sections we care about: (part, section_number, human_label)
# Note: Part 5 was reorganized — legacy citations (§ 5.32, § 5.36) now
# correspond to the sections below in the current CFR.
TARGETS = [
    ("5",  "5.63",  "Mandatory label information"),
    ("5",  "5.64",  "Brand name"),
    ("5",  "5.65",  "Alcohol content"),
    ("5",  "5.70",  "Net contents"),
    ("5",  "5.121", "General prohibitions (misleading representations)"),
    ("5",  "5.143", "Whisky (class and type standards of identity)"),
    ("16", "16.21", "Mandatory health warning label information"),
]

ECFR_BASE = "https://www.ecfr.gov/api/versioner/v1"


def latest_issue_date(title: str = "27") -> str:
    """Return the most recent issue date for the given CFR title."""
    url = f"{ECFR_BASE}/titles"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.load(r)
    for t in data.get("titles", []):
        if str(t.get("number")) == title:
            return t["latest_issue_date"]
    raise RuntimeError(f"Title {title} not found in eCFR titles list")


def fetch_part_xml(title: str, part: str, issue_date: str) -> ET.Element:
    """Download a full CFR part as XML and return the root element."""
    url = f"{ECFR_BASE}/full/{issue_date}/title-{title}.xml?part={part}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return ET.fromstring(r.read())


def extract_section(root: ET.Element, section_n: str) -> dict:
    """
    Pull a single section out of a part XML root.
    Returns {"number": "5.32", "heading": "...", "text": "...", "paragraphs": [...]}
    """
    for elem in root.iter():
        n = elem.get("N", "")
        typ = elem.get("TYPE", "")
        if typ == "SECTION" and n == section_n:
            heading = ""
            paragraphs = []

            for child in elem:
                tag = child.tag
                if tag == "HEAD":
                    heading = (child.text or "").strip()
                elif tag == "P":
                    text = "".join(child.itertext()).strip()
                    if text:
                        paragraphs.append(text)
                elif tag == "I":
                    text = "".join(child.itertext()).strip()
                    if text:
                        paragraphs.append(text)

            full_text = "\n".join(paragraphs)
            return {
                "number":     section_n,
                "heading":    heading,
                "paragraphs": paragraphs,
                "text":       full_text,
            }

    return {"number": section_n, "heading": "", "paragraphs": [], "text": ""}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sync(out_dir: Path, verbose: bool = True) -> dict:
    """Fetch all target sections and write to out_dir. Returns manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    issue_date = latest_issue_date("27")
    if verbose:
        print(f"eCFR Title 27 — latest issue date: {issue_date}")

    # Cache part XML so we don't fetch the same part twice
    part_cache: dict[str, ET.Element] = {}
    manifest = {
        "synced":        str(date.today()),
        "issue_date":    issue_date,
        "sections":      {},
    }

    for part, section_n, label in TARGETS:
        if part not in part_cache:
            if verbose:
                print(f"  Fetching 27 CFR Part {part}...")
            part_cache[part] = fetch_part_xml("27", part, issue_date)

        sec = extract_section(part_cache[part], section_n)
        sec["label"]      = label
        sec["part"]       = part
        sec["issue_date"] = issue_date
        sec["citation"]   = f"27 CFR § {section_n}"

        out_file = out_dir / f"{section_n}.json"
        out_file.write_text(json.dumps(sec, indent=2))

        h = sha256(sec["text"])
        manifest["sections"][section_n] = {
            "citation":  sec["citation"],
            "heading":   sec["heading"],
            "hash":      h,
            "chars":     len(sec["text"]),
        }
        if verbose:
            print(f"  § {section_n}  {sec['heading'][:60]}  ({len(sec['text'])} chars, hash={h})")

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(f"\nWritten to {out_dir}/")
        print(f"Manifest:  {manifest_file}")
    return manifest


def check(out_dir: Path) -> None:
    """Compare stored hashes against a fresh fetch and report changes."""
    manifest_file = out_dir / "manifest.json"
    if not manifest_file.exists():
        print("No local data found — run without --check to do initial sync.")
        sys.exit(1)

    stored = json.loads(manifest_file.read_text())
    print(f"Stored sync date : {stored['synced']}")
    print(f"Stored issue date: {stored['issue_date']}")

    issue_date = latest_issue_date("27")
    print(f"Current issue date: {issue_date}")

    if issue_date == stored["issue_date"]:
        print("No new issue — regulations unchanged.")
        return

    print(f"\nNew issue available ({issue_date}). Checking for section changes...")
    part_cache: dict[str, ET.Element] = {}
    changed = []

    for part, section_n, _ in TARGETS:
        if part not in part_cache:
            part_cache[part] = fetch_part_xml("27", part, issue_date)
        sec = extract_section(part_cache[part], section_n)
        new_hash = sha256(sec["text"])
        old_hash = stored["sections"].get(section_n, {}).get("hash", "")
        status = "CHANGED" if new_hash != old_hash else "unchanged"
        print(f"  § {section_n}  {status}  (old={old_hash}  new={new_hash})")
        if new_hash != old_hash:
            changed.append(section_n)

    if changed:
        print(f"\n{len(changed)} section(s) changed: {', '.join(changed)}")
        print("Run without --check to update local cache.")
    else:
        print("\nAll sections unchanged in new issue.")


def show(out_dir: Path, section_n: str) -> None:
    """Print the stored text for a section."""
    f = out_dir / f"{section_n}.json"
    if not f.exists():
        print(f"Section {section_n} not in local cache — run sync first.")
        sys.exit(1)
    sec = json.loads(f.read_text())
    print(f"{sec['citation']} — {sec['heading']}")
    print(f"(issue date: {sec['issue_date']})")
    print()
    print(sec["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check",  action="store_true",
                    help="Compare stored hashes to current eCFR; report changes, no write")
    ap.add_argument("--show",   metavar="SECTION",
                    help="Print stored text for a section (e.g. 5.32)")
    ap.add_argument("--out",    default=str(OUT_DIR),
                    help=f"Output directory (default: {OUT_DIR})")
    args = ap.parse_args()

    out = Path(args.out)

    if args.show:
        show(out, args.show)
    elif args.check:
        check(out)
    else:
        sync(out)


if __name__ == "__main__":
    main()
