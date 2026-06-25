#!/usr/bin/env python3
"""
HMM sequence tagger for TTB alcohol label OCR output.

Replaces the regex heuristics in extractFromText with a proper
Hidden Markov Model over the Tesseract token stream.

States (hidden):
    START  MFR  BRAND  TYPE  ATTR_ABV  ATTR_NET  WARNING  OVR  END

Observations (visible):
    Each non-empty line from Tesseract output.

Emission features per token:
    - is_allcaps           (all alphabetic chars uppercase)
    - has_percent          (contains % or "alc" or "vol")
    - has_volume_unit      (ml / L / fl oz / liter)
    - starts_gov_warning   (begins with "GOVERNMENT WARNING")
    - is_short             (≤3 words — brand names are usually short)
    - markov_score         (log-likelihood under the brand Markov chain)
    - has_digits
    - position_rank        (0 = first token in document, normalised 0-1)

Usage (standalone demo):
    echo "OLD FORESTER\nSTRAIGHT BOURBON WHISKY\n43.0% Alc./Vol.\n750ml\nGOVERNMENT WARNING:" \
        | python3 scripts/hmm_tagger.py

Usage (import):
    from hmm_tagger import LabelHMM
    hmm = LabelHMM()
    fields = hmm.decode(ocr_text)
"""

import json
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the Markov matrix if available (built by brand_markov.py)
# ---------------------------------------------------------------------------
_MARKOV_PATH = Path(__file__).parent.parent / "testdata" / "brand_markov.json"
_markov_model = None

def _get_markov():
    global _markov_model
    if _markov_model is None and _MARKOV_PATH.exists():
        from brand_markov import BrandMarkovChain
        _markov_model = BrandMarkovChain.from_dict(json.loads(_MARKOV_PATH.read_text()))
    return _markov_model


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
STATES = ["START", "MFR", "BRAND", "TYPE", "ATTR_ABV", "ATTR_NET", "WARNING", "OVR", "END"]
S = {s: i for i, s in enumerate(STATES)}

# ---------------------------------------------------------------------------
# Transition matrix A  (hand-seeded from COLA label layout conventions)
# Each row must sum to 1.  Rows = from-state, cols = to-state.
# Order: START  MFR   BRAND  TYPE   ABV    NET    WARN   OVR   END
# ---------------------------------------------------------------------------
_A_RAW = {
    #              START  MFR   BRAND  TYPE   ABV    NET    WARN   OVR    END
    "START":      [0.00,  0.15,  0.50,  0.05,  0.02,  0.01,  0.02,  0.25,  0.00],
    "MFR":        [0.00,  0.05,  0.65,  0.05,  0.02,  0.01,  0.02,  0.20,  0.00],
    "BRAND":      [0.00,  0.02,  0.05,  0.65,  0.05,  0.02,  0.02,  0.18,  0.01],
    "TYPE":       [0.00,  0.01,  0.03,  0.05,  0.50,  0.10,  0.05,  0.25,  0.01],
    "ATTR_ABV":   [0.00,  0.01,  0.01,  0.02,  0.05,  0.50,  0.15,  0.25,  0.01],
    "ATTR_NET":   [0.00,  0.01,  0.01,  0.02,  0.05,  0.05,  0.55,  0.30,  0.01],
    "WARNING":    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.85,  0.05,  0.10],
    "OVR":        [0.00,  0.05,  0.10,  0.08,  0.05,  0.05,  0.10,  0.55,  0.02],
    "END":        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00],
}

# Convert to log-probabilities
import numpy as np
A = np.zeros((len(STATES), len(STATES)))
for i, s in enumerate(STATES):
    row = _A_RAW.get(s, [1/len(STATES)] * len(STATES))
    for j, p in enumerate(row):
        A[i][j] = math.log(max(p, 1e-9))


# ---------------------------------------------------------------------------
# Emission feature extractor
# ---------------------------------------------------------------------------
_RE_PCT     = re.compile(r'\d+\.?\d*\s*%|alc\.|vol\.', re.IGNORECASE)
_RE_VOL     = re.compile(r'\d+\.?\d*\s*(?:mL|L\b|fl\.?\s*oz|liters?)', re.IGNORECASE)
_RE_GOV     = re.compile(r'^GOVERNMENT\s+WARNING', re.IGNORECASE)
_GOV_WARNING = "GOVERNMENT WARNING"


def features(token: str, position_rank: float) -> dict:
    upper = token.upper()
    alpha_chars = [c for c in token if c.isalpha()]
    is_allcaps = bool(alpha_chars) and all(c.isupper() for c in alpha_chars)
    words = token.split()
    markov = _get_markov()
    markov_score = markov.score(token) if markov else -3.0

    return {
        "is_allcaps":         is_allcaps,
        "has_percent":        bool(_RE_PCT.search(token)),
        "has_volume_unit":    bool(_RE_VOL.search(token)),
        "starts_gov_warning": bool(_RE_GOV.match(token)),
        "is_short":           len(words) <= 4,
        "markov_score":       markov_score,
        "has_digits":         any(c.isdigit() for c in token),
        "position_rank":      position_rank,   # 0 = top of label
    }


# ---------------------------------------------------------------------------
# Emission log-probability  B(state, token)
# Encodes domain knowledge about what each field looks like.
# ---------------------------------------------------------------------------
def log_emit(state: str, feat: dict) -> float:
    f = feat
    if state == "START" or state == "END":
        return 0.0

    if state == "MFR":
        score = (
            (0.4  if f["is_allcaps"]           else -0.5) +
            (-0.5 if f["has_percent"]           else  0.0) +
            (-0.5 if f["has_volume_unit"]       else  0.0) +
            (-2.0 if f["starts_gov_warning"]    else  0.0) +
            (0.3  if f["is_short"]              else -0.2) +
            (f["markov_score"] * 0.3) +
            (0.2  if f["position_rank"] < 0.2  else -0.2)
        )
    elif state == "BRAND":
        score = (
            (0.8  if f["is_allcaps"]            else -0.3) +
            (-0.8 if f["has_percent"]           else  0.0) +
            (-0.8 if f["has_volume_unit"]       else  0.0) +
            (-3.0 if f["starts_gov_warning"]    else  0.0) +
            (0.5  if f["is_short"]              else -0.3) +
            (f["markov_score"] * 0.8) +          # Markov chain is most useful here
            (0.4  if f["position_rank"] < 0.3  else -0.1)
        )
    elif state == "TYPE":
        bourbon_keywords = any(kw in feat.get("_upper", "") for kw in
                               ["BOURBON", "WHISKY", "WHISKEY", "VODKA", "GIN", "RUM",
                                "TEQUILA", "BRANDY", "CORDIAL", "STRAIGHT"])
        score = (
            (0.5  if f["is_allcaps"]            else -0.2) +
            (-0.8 if f["has_percent"]           else  0.0) +
            (-0.8 if f["has_volume_unit"]       else  0.0) +
            (-3.0 if f["starts_gov_warning"]    else  0.0) +
            (-0.3 if f["is_short"]              else  0.2) +  # TYPE usually longer
            (0.8  if bourbon_keywords           else -0.4)
        )
    elif state == "ATTR_ABV":
        score = (
            (-0.5 if f["is_allcaps"]            else  0.1) +
            (2.0  if f["has_percent"]           else -1.5) +
            (-0.5 if f["has_volume_unit"]       else  0.2) +
            (-3.0 if f["starts_gov_warning"]    else  0.0) +
            (0.5  if f["has_digits"]            else -1.0)
        )
    elif state == "ATTR_NET":
        score = (
            (-0.5 if f["is_allcaps"]            else  0.1) +
            (-0.5 if f["has_percent"]           else  0.2) +
            (2.0  if f["has_volume_unit"]       else -1.5) +
            (-3.0 if f["starts_gov_warning"]    else  0.0) +
            (0.5  if f["has_digits"]            else -1.0)
        )
    elif state == "WARNING":
        # Continuation lines of the warning don't start with "GOVERNMENT WARNING:"
        # but contain characteristic regulatory vocabulary.
        _upper = feat.get("_upper", "")
        warning_vocab = any(kw in _upper for kw in [
            "SURGEON GENERAL", "PREGNANCY", "BIRTH DEFECTS",
            "ALCOHOLIC BEVERAGES", "OPERATE MACHINERY", "HEALTH PROBLEMS",
            "IMPAIRS", "CONSUMPTION", "ABILITY TO DRIVE",
        ])
        score = (
            (3.0  if f["starts_gov_warning"]    else  0.0) +
            (1.5  if warning_vocab              else -0.5) +
            (0.2  if f["is_allcaps"]            else  0.0) +
            (-0.5 if f["has_percent"]           else  0.0) +
            (-0.5 if f["has_volume_unit"]       else  0.0)
        )
    else:  # OVR — catch-all; mildly prefers tokens that don't fit anything else
        score = -0.3

    return float(np.clip(score, -10, 5))


# ---------------------------------------------------------------------------
# Log-sum-exp helper (numerically stable addition in log-space)
# ---------------------------------------------------------------------------
def _log_sum_exp(log_probs: np.ndarray) -> float:
    """Stable log(sum(exp(x))) — avoids underflow when summing tiny probabilities."""
    m = np.max(log_probs)
    if np.isneginf(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(log_probs - m))))


# ---------------------------------------------------------------------------
# HMM: Viterbi decoder + Forward algorithm
# ---------------------------------------------------------------------------
class LabelHMM:
    def decode_tokens(self, tokens: list[str]) -> list[tuple[str, str]]:
        """
        Run Viterbi over a list of OCR lines.
        Returns [(token, state), ...].
        """
        if not tokens:
            return []

        n = len(tokens)
        ns = len(STATES)

        # viterbi[t][s] = best log-prob of being in state s at position t
        viterbi = np.full((n, ns), -np.inf)
        backptr = np.zeros((n, ns), dtype=int)

        # Initialisation — transition from START
        start_idx = S["START"]
        feats0 = features(tokens[0], 0.0)
        feats0["_upper"] = tokens[0].upper()
        for s in range(ns):
            viterbi[0][s] = A[start_idx][s] + log_emit(STATES[s], feats0)

        # Recursion
        for t in range(1, n):
            rank = t / max(1, n - 1)
            feat = features(tokens[t], rank)
            feat["_upper"] = tokens[t].upper()
            for s in range(ns):
                candidates = viterbi[t-1] + A[:, s]
                best_prev  = int(np.argmax(candidates))
                viterbi[t][s] = candidates[best_prev] + log_emit(STATES[s], feat)
                backptr[t][s] = best_prev

        # Termination — transition to END
        end_idx    = S["END"]
        candidates = viterbi[n-1] + A[:, end_idx]
        last_state = int(np.argmax(candidates))

        # Backtrack
        path = [last_state]
        for t in range(n-1, 0, -1):
            last_state = backptr[t][last_state]
            path.append(last_state)
        path.reverse()

        return [(tokens[i], STATES[path[i]]) for i in range(n)]

    def forward(self, tokens: list[str]) -> float:
        """
        Forward algorithm — computes log P(observations | HMM).

        This is the LIKELIHOOD problem from the HMM trio:
          - Viterbi  → best single state path   (field extraction)
          - Forward  → sum over ALL state paths  (anomaly / fraud score)
          - Baum-Welch → update parameters from data  (learning)

        A very negative return value means the token sequence is
        improbable under the trained model — flag as suspicious.

        The ice-cream analogy: given Jason ate [3, 1, 3] ice creams,
        what is the total probability of that sequence summing over
        every possible HOT/COLD weather combination?  That's this.
        For labels: sum over every possible (BRAND, TYPE, ...) labeling.
        """
        if not tokens:
            return -np.inf

        n  = len(tokens)
        ns = len(STATES)

        # alpha[t][s] = log P(o_1...o_t, q_t=s)
        alpha = np.full((n, ns), -np.inf)

        # Initialisation
        start_idx = S["START"]
        feat0 = features(tokens[0], 0.0)
        feat0["_upper"] = tokens[0].upper()
        for s in range(ns):
            alpha[0][s] = A[start_idx][s] + log_emit(STATES[s], feat0)

        # Recursion — sum (in log-space) over all previous states
        for t in range(1, n):
            rank = t / max(1, n - 1)
            feat = features(tokens[t], rank)
            feat["_upper"] = tokens[t].upper()
            for s in range(ns):
                # log sum_i [ alpha[t-1][i] + A[i][s] ]  + B[s](o_t)
                incoming = alpha[t-1] + A[:, s]
                alpha[t][s] = _log_sum_exp(incoming) + log_emit(STATES[s], feat)

        # Termination — sum into END state
        end_idx = S["END"]
        return _log_sum_exp(alpha[n-1] + A[:, end_idx])

    def likelihood_score(self, ocr_text: str) -> float:
        """
        Normalised log-likelihood per token.
        Near 0  → sequence looks like a legitimate bourbon label.
        Very −  → sequence is anomalous; flag for manual review.
        """
        tokens = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        if not tokens:
            return -np.inf
        return self.forward(tokens) / len(tokens)

    def decode(self, ocr_text: str) -> dict:
        """
        Full pipeline: raw OCR text → field dict.
        Returns keys: brand_name, class_type, abv_percent, net_contents,
                       government_warning, manufacturer.
        """
        tokens = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        labeled = self.decode_tokens(tokens)

        fields: dict[str, list[str]] = {
            "brand_name": [], "class_type": [], "abv_percent": [],
            "net_contents": [], "government_warning": [], "manufacturer": [],
        }
        state_map = {
            "BRAND":    "brand_name",
            "TYPE":     "class_type",
            "ATTR_ABV": "abv_percent",
            "ATTR_NET": "net_contents",
            "WARNING":  "government_warning",
            "MFR":      "manufacturer",
        }
        for token, state in labeled:
            key = state_map.get(state)
            if key:
                fields[key].append(token)

        return {k: " ".join(v).strip() for k, v in fields.items()}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Run Tesseract on this image and tag the output")
    ap.add_argument("--text",  help="Tag this raw OCR text string directly")
    args = ap.parse_args()

    hmm = LabelHMM()

    if args.image:
        import subprocess
        result = subprocess.run(
            ["tesseract", args.image, "stdout", "--psm", "4"],
            capture_output=True, text=True
        )
        ocr_text = result.stdout
    elif args.text:
        ocr_text = args.text
    else:
        ocr_text = sys.stdin.read()

    tokens = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    labeled = hmm.decode_tokens(tokens)

    print("Token-level state assignments (Viterbi):")
    print(f"  {'TOKEN':<45} STATE")
    print(f"  {'-'*45} -----")
    for token, state in labeled:
        print(f"  {token:<45} {state}")

    print()
    fields = hmm.decode(ocr_text)
    print("Extracted fields (Viterbi):")
    for k, v in fields.items():
        if v:
            print(f"  {k:<22} {v[:70]}")

    print()
    score = hmm.likelihood_score(ocr_text)
    flag  = "  ⚠ SUSPICIOUS — not a typical bourbon label" if score < -4.0 else "  ✓ plausible label sequence"
    print(f"Forward likelihood score: {score:+.3f}{flag}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
