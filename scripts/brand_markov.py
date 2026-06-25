#!/usr/bin/env python3
"""
Brand-name character-level Markov chain for the TTB Label Verifier.

Two operational modes:

  MODE A — Synthetic Generation
    Generate realistic, non-existent bourbon brand names for test-data
    augmentation.  The generated names follow the character-transition
    statistics of real TTB-registered bourbon brands so they stress-test
    the CRF tagger without any hand-crafted bias.

  MODE B — Counterfeit / Anomaly Detection
    Score an OCR-extracted brand name against the transition matrix.
    A high score (near 0) means the character flow is typical of real
    bourbon brand names.  A very negative score flags the string as
    either an OCR artefact or a phonetic counterfeit attempt.

Training corpus: 990 bourbon COLA records from the TTB Public Registry
(testdata/cola_whisky.csv) or the 30-brand built-in fallback list.

Usage:
    python3 scripts/brand_markov.py --mode generate --count 10
    python3 scripts/brand_markov.py --mode score --brand "Elijah Crais"
    python3 scripts/brand_markov.py --mode matrix --out testdata/brand_markov.json
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_COLA_CSV   = _ROOT / "testdata" / "cola_whisky.csv"
_LABELS_CSV = _ROOT / "testdata" / "labels.csv"

# Built-in fallback — used when COLA CSV is unavailable
_FALLBACK_BRANDS = [
    "Maker's Mark", "Wild Turkey", "Buffalo Trace", "Four Roses",
    "Knob Creek", "Woodford Reserve", "Evan Williams", "Heaven Hill",
    "Jim Beam", "Elijah Craig", "Old Forester", "Larceny",
    "Bulleit", "Angel's Envy", "Old Dominick", "Henry McKenna",
    "Booker's", "Baker's", "Basil Hayden", "Rowan's Creek",
    "Noah's Mill", "Kentucky Vintage", "1792 Ridgemont", "Very Old Barton",
    "Ancient Age", "Old Charter", "W.L. Weller", "Pappy Van Winkle",
    "Ezra Brooks", "Buffalo Trace Select",
]


# ---------------------------------------------------------------------------
class BrandMarkovChain:
    """Character-level n-gram Markov chain trained on bourbon brand names."""

    START = "^"
    END   = "$"

    def __init__(self, n_gram: int = 2):
        self.n = n_gram
        self.transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    def _pad(self, name: str) -> str:
        return self.START * self.n + name.strip().lower() + self.END

    def train(self, brands: list[str]) -> "BrandMarkovChain":
        for brand in brands:
            s = self._pad(brand)
            for i in range(len(s) - self.n):
                ctx  = s[i : i + self.n]
                next_c = s[i + self.n]
                self.transitions[ctx][next_c] += 1
        return self

    # ------------------------------------------------------------------
    def generate(self, max_len: int = 30) -> str:
        ctx = self.START * self.n
        name = ""
        for _ in range(max_len):
            opts = self.transitions.get(ctx)
            if not opts:
                break
            chars  = list(opts.keys())
            counts = list(opts.values())
            total  = sum(counts)
            probs  = [c / total for c in counts]
            nxt = random.choices(chars, weights=probs)[0]
            if nxt == self.END:
                break
            name += nxt
            ctx = ctx[1:] + nxt
        return name.title()

    # ------------------------------------------------------------------
    def score(self, brand: str) -> float:
        """
        Return log-likelihood per transition, normalised by sequence length.

        Near 0  → character flow typical of real bourbon brand names.
        Very −  → unusual; likely OCR noise or counterfeit phonetic match.
        """
        s = self._pad(brand)
        ll = 0.0
        n_trans = 0
        for i in range(len(s) - self.n):
            ctx    = s[i : i + self.n]
            next_c = s[i + self.n]
            n_trans += 1
            counts = self.transitions.get(ctx, {})
            if counts and next_c in counts:
                prob = counts[next_c] / sum(counts.values())
                ll  += math.log(prob)
            else:
                ll  += math.log(1e-5)   # unseen-transition penalty
        return ll / max(1, n_trans)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "transitions": {k: dict(v) for k, v in self.transitions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BrandMarkovChain":
        m = cls(n_gram=d["n"])
        for ctx, counts in d["transitions"].items():
            m.transitions[ctx] = defaultdict(int, counts)
        return m


# ---------------------------------------------------------------------------
def load_brands() -> list[str]:
    if _COLA_CSV.exists():
        brands = []
        with open(_COLA_CSV, encoding="latin-1") as f:
            for row in csv.DictReader(f):
                name = (row.get("Brand Name") or "").strip()
                if name and "BOURBON" in (row.get("Class/Type Desc") or ""):
                    brands.append(name)
        print(f"[markov] Loaded {len(brands)} brands from {_COLA_CSV.name}")
        return brands
    # fallback
    print(f"[markov] COLA CSV not found — using {len(_FALLBACK_BRANDS)}-brand fallback list")
    return list(_FALLBACK_BRANDS)


# ---------------------------------------------------------------------------
def cmd_generate(model: BrandMarkovChain, count: int) -> None:
    print("--- Generated Synthetic Brands ---")
    for _ in range(count):
        print(f"  {model.generate()}")


def cmd_score(model: BrandMarkovChain, brands: list[str]) -> None:
    print("--- Anomaly Scores (near 0 = normal, very negative = suspicious) ---")
    for b in brands:
        s = model.score(b)
        flag = "  ⚠ SUSPICIOUS" if s < -3.5 else ""
        print(f"  {b!r:35s}  {s:+.3f}{flag}")


def cmd_matrix(model: BrandMarkovChain, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.to_dict(), indent=2))
    print(f"[markov] Matrix saved to {out}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TTB brand-name Markov chain")
    parser.add_argument("--mode", choices=["generate", "score", "matrix", "demo"],
                        default="demo")
    parser.add_argument("--n-gram", type=int, default=2)
    parser.add_argument("--count", type=int, default=10,
                        help="Number of names to generate (generate mode)")
    parser.add_argument("--brand", nargs="+", default=[],
                        help="Brand name(s) to score (score mode)")
    parser.add_argument("--out", default=str(_ROOT / "testdata" / "brand_markov.json"),
                        help="Output path for matrix JSON (matrix mode)")
    args = parser.parse_args()

    brands = load_brands()
    model  = BrandMarkovChain(n_gram=args.n_gram).train(brands)

    if args.mode == "generate":
        cmd_generate(model, args.count)
    elif args.mode == "score":
        cmd_score(model, args.brand or ["Williams", "Wllixmzs", "Elijah Crais"])
    elif args.mode == "matrix":
        cmd_matrix(model, Path(args.out))
    else:
        # Demo: run both modes
        print(f"Trained on {len(brands)} bourbon brand names (n={args.n_gram})\n")
        cmd_generate(model, 8)
        print()
        cmd_score(model, [
            "Williams",          # real word in COLA names — should score high
            "Elijah Craig",      # known brand — high
            "Elijah Crais",      # phonetic counterfeit — moderate/low
            "Wllixmzs",          # random OCR garbage — very low
            "Buffalo Trace",     # known brand — high
            "Buffal0 Tr4ce",     # OCR digit substitution — low
        ])
        print()
        cmd_matrix(model, Path(args.out))


if __name__ == "__main__":
    main()
