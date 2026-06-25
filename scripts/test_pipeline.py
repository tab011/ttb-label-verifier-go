#!/usr/bin/env python3
"""
Integration test: POST every label in testdata/labels.csv to the running
TTB Label Verifier server and compare the returned verdict to what we expect.

Expected verdict is derived from filename:
  *_fail_warn.*  → FAIL  (wrong-case government warning)
  *_fail_abv.*   → FAIL  (ABV mismatch)
  everything else → PASS

Usage:
    python3 scripts/test_pipeline.py [--url http://localhost:8181] [--verbose]
"""

import argparse
import csv
import sys
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

TESTDATA = Path(__file__).parent.parent / "testdata"
CSV_PATH = TESTDATA / "labels.csv"
IMG_DIR  = TESTDATA / "images"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def expected_verdict(filename: str) -> str:
    f = filename.lower()
    if "_fail_" in f:
        return "FAIL"
    return "PASS"


def post_verify(url: str, img_path: Path, brand: str, class_type: str,
                abv: str, net: str) -> dict:
    boundary = "----TTBTestBoundary7391"
    body_parts = []

    def field(name, value):
        body_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            .encode()
        )

    field("brand_name",   brand)
    field("class_type",   class_type)
    field("abv_percent",  abv)
    field("net_contents", net)

    img_bytes = img_path.read_bytes()
    body_parts.append(
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"image\"; filename=\"{img_path.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n".encode() + img_bytes + b"\r\n"
    )
    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    req = urllib.request.Request(
        f"{url}/verify",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def check_server(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url}/", timeout=5)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8181")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not check_server(args.url):
        print(f"{RED}Server not reachable at {args.url}{RESET}")
        print("Run:  cd ~/vscode/ttb-label-verifier-go && /tmp/ttb-server -addr :8181 -cola-csv testdata/labels.csv &")
        sys.exit(1)

    rows = list(csv.DictReader(CSV_PATH.read_text().splitlines()))
    rows = [r for r in rows if r.get("filename")]
    print(f"{BOLD}TTB Label Verifier — Integration Test{RESET}")
    print(f"Server:  {args.url}")
    print(f"Labels:  {len(rows)}")
    print()

    passed = failed = errors = 0
    results = []

    for row in rows:
        fname    = row["filename"]
        img_path = IMG_DIR / fname
        expected = expected_verdict(fname)

        if not img_path.exists():
            print(f"{YELLOW}SKIP{RESET}  {fname}  (image not found)")
            errors += 1
            continue

        t0 = time.monotonic()
        try:
            resp = post_verify(
                args.url, img_path,
                brand=row.get("brand_name", ""),
                class_type=row.get("class_type", ""),
                abv=row.get("abv_percent", ""),
                net=row.get("net_contents", ""),
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"{RED}ERROR{RESET} {fname}: HTTP {e.code} — {body[:80]}")
            errors += 1
            continue
        except Exception as e:
            print(f"{RED}ERROR{RESET} {fname}: {e}")
            errors += 1
            continue

        elapsed = time.monotonic() - t0
        verdict = resp.get("verdict", "ERROR")
        ok = verdict == expected

        if ok:
            passed += 1
            sym = f"{GREEN}PASS{RESET}"
        else:
            failed += 1
            sym = f"{RED}FAIL{RESET}"

        result_line = f"{sym}  {fname:<35s}  got={verdict:<5s} want={expected:<5s}  {elapsed:.2f}s"
        print(result_line)

        if args.verbose and not ok:
            for field, fv in (resp.get("fields") or {}).items():
                status = fv.get("status", "?")
                color  = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
                print(f"        {color}{status:<8}{RESET}  {field}")
                if status != "PASS":
                    print(f"          expected: {fv.get('expected','')[:70]}")
                    print(f"          got:      {fv.get('extracted','')[:70]}")

        results.append({
            "filename": fname,
            "expected": expected,
            "verdict":  verdict,
            "ok":       ok,
            "elapsed":  round(elapsed, 3),
        })

    total = passed + failed + errors
    print()
    print("─" * 60)
    print(f"{BOLD}Results:{RESET}  {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}  {YELLOW}{errors} errors{RESET}  / {total} total")

    if results:
        avg = sum(r["elapsed"] for r in results) / len(results)
        mx  = max(r["elapsed"] for r in results)
        print(f"Timing:   avg {avg:.2f}s   max {mx:.2f}s")

    accuracy = passed / total * 100 if total else 0
    print(f"Accuracy: {accuracy:.0f}%")

    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
