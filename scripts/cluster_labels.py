#!/usr/bin/env python3
"""
cluster_labels.py — explore the real bottle dataset via combined text + visual embeddings.

Pipeline per image:
  1. Crop bottle region using Pascal VOC bounding box from XML
  2. Tesseract OCR  → TF-IDF text vector   (512-dim)
  3. MobileNetV3-Small (ONNX) → visual vector (576-dim)
  4. Concatenate + L2-normalise → combined feature vector (1088-dim)
  5. UMAP → 50D for clustering, 2D for scatter plot
  6. K-Means with silhouette-score sweep to pick best k
  7. Output:
       dataset/cluster_assignments.csv   — per-image: cluster, confidence, OCR preview
       dataset/cluster_scatter.png       — 2D UMAP coloured by cluster
       dataset/cluster_representatives.csv — closest image to each centroid

Usage:
    python3 scripts/cluster_labels.py --zip ~/Downloads/archive.zip
    python3 scripts/cluster_labels.py --zip ~/Downloads/archive.zip --k 8
    python3 scripts/cluster_labels.py --zip ~/Downloads/archive.zip --no-visual
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"

# MobileNetV3-Small ONNX model (ONNX Model Zoo — one-time 13 MB download)
MOBILENET_URL  = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MOBILENET_PATH = DATASET_DIR / "mobilenet_v2.onnx"

# ---------------------------------------------------------------------------
# Pascal VOC XML
# ---------------------------------------------------------------------------

def parse_voc_xml(xml_bytes: bytes) -> dict | None:
    try:
        root = ET.fromstring(xml_bytes)
        obj  = root.find("object")
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


def crop_bottle(img_bytes: bytes, bb: dict, pad: int = 20) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x1 = max(0, bb["xmin"] - pad)
    y1 = max(0, bb["ymin"] - pad)
    x2 = min(w, bb["xmax"] + pad)
    y2 = min(h, bb["ymax"] + pad)
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------

def run_tesseract(crop_bgr: np.ndarray) -> str:
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bw   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    up      = cv2.resize(bw, (bw.shape[1]*2, bw.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    _, buf  = cv2.imencode(".jpg", up)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(buf.tobytes())
        tmp = f.name
    try:
        r = subprocess.run(["tesseract", tmp, "stdout", "--psm", "4"],
                           capture_output=True, text=True)
        return r.stdout.strip()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Visual embeddings — MobileNetV2 via ONNX Runtime
# ---------------------------------------------------------------------------

def _download_model():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if MOBILENET_PATH.exists():
        return
    print(f"Downloading MobileNetV2 ONNX weights (~13 MB) …", flush=True)
    urllib.request.urlretrieve(MOBILENET_URL, MOBILENET_PATH)
    print("  done.")


def _get_ort_session():
    import onnxruntime as ort
    _download_model()
    return ort.InferenceSession(str(MOBILENET_PATH),
                                providers=["CPUExecutionProvider"])


_session = None

def visual_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Return 1000-dim softmax logits as a visual feature vector (no classification head needed)."""
    global _session
    if _session is None:
        _session = _get_ort_session()

    # MobileNetV2 ONNX expects [1, 3, 224, 224] float32, ImageNet normalisation
    rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = (resized - mean) / std
    tensor = normed.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 224, 224]

    input_name  = _session.get_inputs()[0].name
    output_name = _session.get_outputs()[0].name
    out = _session.run([output_name], {input_name: tensor})[0]   # [1, 1000]
    return out[0]                                                  # (1000,)


def hog_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """HOG fallback when ONNX model unavailable."""
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 256))
    hog     = cv2.HOGDescriptor((128, 256), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(resized).flatten()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(zip_path: Path, use_visual: bool = True):
    filenames = []
    ocr_texts = []
    visual_vecs = []

    with zipfile.ZipFile(zip_path) as zf:
        xml_names = sorted(n for n in zf.namelist() if n.endswith(".xml"))
        img_index = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        total = len(xml_names)

        for i, xml_name in enumerate(xml_names, 1):
            bb = parse_voc_xml(zf.read(xml_name))
            if not bb:
                continue
            stem = Path(bb["filename"]).stem
            img_key = img_index.get(stem)
            if not img_key:
                continue

            print(f"  [{i:3d}/{total}] {bb['filename']} …", end=" ", flush=True)

            try:
                img_bytes = zf.read(img_key)
                crop      = crop_bottle(img_bytes, bb)
                ocr       = run_tesseract(crop)

                if use_visual:
                    try:
                        vec = visual_embedding(crop)
                    except Exception as e:
                        print(f"(ONNX err: {e}, using HOG)", end=" ")
                        vec = hog_embedding(crop)
                else:
                    vec = hog_embedding(crop)

                filenames.append(bb["filename"])
                ocr_texts.append(ocr or " ")
                visual_vecs.append(vec)
                words = len(ocr.split())
                print(f"{words} OCR words, visual dim={vec.shape[0]}")

            except Exception as e:
                print(f"ERROR: {e}")

    return filenames, ocr_texts, visual_vecs


# ---------------------------------------------------------------------------
# Build combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(ocr_texts, visual_vecs, text_weight=0.4):
    # Text: TF-IDF (512 dims)
    tfidf = TfidfVectorizer(max_features=512, sublinear_tf=True,
                            ngram_range=(1, 2), min_df=1)
    text_mat = tfidf.fit_transform(ocr_texts).toarray().astype(np.float32)
    text_mat = normalize(text_mat)

    # Visual: L2-normalise
    vis_mat = np.array(visual_vecs, dtype=np.float32)
    vis_mat = normalize(vis_mat)

    # Weighted concatenation
    combined = np.hstack([
        text_mat * text_weight,
        vis_mat  * (1.0 - text_weight),
    ])
    return combined, tfidf


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(X, n_components=50, n_components_2d=2):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed — using PCA instead")
        from sklearn.decomposition import PCA
        pca50 = PCA(n_components=min(n_components, X.shape[0]-1, X.shape[1]))
        pca2  = PCA(n_components=2)
        return pca50.fit_transform(X), pca2.fit_transform(X)

    reducer50 = umap.UMAP(n_components=n_components, n_neighbors=10,
                          min_dist=0.1, metric="cosine", random_state=42)
    reducer2  = umap.UMAP(n_components=n_components_2d, n_neighbors=10,
                          min_dist=0.2, metric="cosine", random_state=42)
    return reducer50.fit_transform(X), reducer2.fit_transform(X)


# ---------------------------------------------------------------------------
# K-Means with silhouette sweep
# ---------------------------------------------------------------------------

def best_k(X, k_min=2, k_max=12):
    best_score, best_k = -1, k_min
    print(f"\nSilhouette sweep k={k_min}…{k_max}:")
    for k in range(k_min, min(k_max + 1, len(X))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score  = silhouette_score(X, labels)
        print(f"  k={k:2d}  silhouette={score:.4f}", " ← best" if score > best_score else "")
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster(X, k):
    km = KMeans(n_clusters=k, n_init=15, random_state=42)
    labels = km.fit_predict(X)
    # Distance to assigned centroid (normalised) as a proxy for confidence
    dists  = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    max_d  = dists.max() or 1.0
    confidence = 1.0 - (dists / max_d)
    return labels, confidence, km


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_plot(xy, labels, filenames, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    k      = labels.max() + 1
    colors = cm.tab20(np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(12, 9))

    for c in range(k):
        mask = labels == c
        ax.scatter(xy[mask, 0], xy[mask, 1], color=colors[c],
                   label=f"Cluster {c}", alpha=0.75, s=60)

    for i, fname in enumerate(filenames):
        ax.annotate(Path(fname).stem[-6:], (xy[i, 0], xy[i, 1]),
                    fontsize=5, alpha=0.5)

    ax.set_title("TTB Bottle Label Clusters (UMAP 2D)")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Output CSVs
# ---------------------------------------------------------------------------

def write_outputs(filenames, ocr_texts, labels, confidence, km, X50, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    k = km.n_clusters

    # Per-image assignments
    assign_path = out_dir / "cluster_assignments.csv"
    with open(assign_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "cluster", "confidence", "ocr_preview"])
        for fname, lbl, conf, ocr in zip(filenames, labels, confidence, ocr_texts):
            w.writerow([fname, int(lbl), f"{conf:.2f}", ocr[:120].replace("\n", " | ")])
    print(f"Cluster assignments → {assign_path}")

    # Cluster representatives (image closest to centroid)
    rep_path = out_dir / "cluster_representatives.csv"
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "representative_filename", "confidence", "ocr_preview"])
        for c in range(k):
            mask  = np.where(labels == c)[0]
            dists = np.linalg.norm(X50[mask] - km.cluster_centers_[c], axis=1)
            best  = mask[dists.argmin()]
            w.writerow([c, filenames[best], f"{confidence[best]:.2f}",
                        ocr_texts[best][:120].replace("\n", " | ")])
    print(f"Cluster representatives → {rep_path}")

    # Summary
    print(f"\nCluster sizes:")
    for c in range(k):
        n = (labels == c).sum()
        print(f"  Cluster {c:2d}: {n:3d} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip",       required=True, help="Path to bottle photo ZIP")
    ap.add_argument("--k",         type=int, default=0,
                    help="Number of clusters (0 = auto via silhouette sweep)")
    ap.add_argument("--k-min",     type=int, default=2)
    ap.add_argument("--k-max",     type=int, default=12)
    ap.add_argument("--no-visual", action="store_true",
                    help="Skip visual embeddings, use TF-IDF text only")
    ap.add_argument("--no-umap",   action="store_true",
                    help="Skip UMAP, cluster in raw feature space")
    ap.add_argument("--text-weight", type=float, default=0.4,
                    help="Weight for text features vs visual (default 0.4/0.6)")
    ap.add_argument("--out",       default=str(DATASET_DIR),
                    help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)

    print("=== TTB Label Cluster Explorer ===")
    print(f"ZIP: {args.zip}")
    print(f"Visual embeddings: {'off' if args.no_visual else 'MobileNetV2-ONNX (HOG fallback)'}")
    print(f"UMAP: {'off' if args.no_umap else 'on'}")
    print()

    # 1. Extract features
    print("Step 1: Feature extraction")
    filenames, ocr_texts, visual_vecs = extract_features(
        Path(args.zip), use_visual=not args.no_visual
    )

    if not filenames:
        print("No images processed — check ZIP path and XML annotations.")
        sys.exit(1)

    print(f"\nExtracted features for {len(filenames)} images")

    # 2. Build combined feature matrix
    print("\nStep 2: Building combined text + visual feature matrix")
    X, tfidf = build_feature_matrix(
        ocr_texts, visual_vecs,
        text_weight=args.text_weight,
    )
    print(f"  Combined feature matrix: {X.shape}")

    # 3. UMAP
    if not args.no_umap and X.shape[0] > 5:
        print("\nStep 3: UMAP dimensionality reduction")
        n_comp = min(50, X.shape[0] - 2)
        X50, X2 = run_umap(X, n_components=n_comp)
        print(f"  UMAP 50D shape: {X50.shape}")
    else:
        from sklearn.decomposition import PCA
        n = min(50, X.shape[0] - 1, X.shape[1])
        X50 = PCA(n_components=n).fit_transform(X)
        X2  = PCA(n_components=2).fit_transform(X)
        print(f"\nStep 3: PCA {n}D (UMAP skipped)")

    # 4. Cluster
    print("\nStep 4: K-Means clustering")
    if args.k > 0:
        k = args.k
        print(f"  Using k={k} (user-specified)")
    else:
        k = best_k(X50, k_min=args.k_min, k_max=min(args.k_max, len(filenames) - 1))
        print(f"\n  Best k={k} by silhouette score")

    labels, confidence, km = cluster(X50, k)

    # 5. Outputs
    print("\nStep 5: Writing outputs")
    write_outputs(filenames, ocr_texts, labels, confidence, km, X50, out_dir)
    scatter_plot(X2, labels, filenames, out_dir / "cluster_scatter.png")

    print(f"\nDone. Next step: review cluster_representatives.csv ({k} rows) and")
    print(f"annotate one representative per cluster to seed the training corpus.")
    print(f"Then run: python3 scripts/annotate_dataset.py --zip {args.zip} --out dataset/")


if __name__ == "__main__":
    main()
