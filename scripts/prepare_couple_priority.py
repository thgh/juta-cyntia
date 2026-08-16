#!/usr/bin/env python3
"""Bias a photo so fogleman/primitive spends more shapes on the married couple.

Strategy:
1. Detect faces (Haar). Expand to a soft couple ROI covering heads + torsos.
2. Build a soft importance mask (1.0 on couple, ~0.15 elsewhere).
3. Create a guided target: keep the couple sharp; blur + flatten the rest so
   residual error concentrates on the couple during primitive optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASCADE = ROOT / "assets" / "cascades" / "haarcascade_frontalface_default.xml"


def detect_faces(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(str(CASCADE))
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {CASCADE}")

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(max(24, bgr.shape[1] // 40), max(24, bgr.shape[0] // 40)),
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def couple_roi_from_faces(
    faces: list[tuple[int, int, int, int]], width: int, height: int
) -> tuple[int, int, int, int]:
    """Return expanded bounding box covering faces + upper bodies."""
    if not faces:
        # Fallback: central portrait framing (common wedding composition).
        x0 = int(width * 0.18)
        x1 = int(width * 0.82)
        y0 = int(height * 0.08)
        y1 = int(height * 0.78)
        return x0, y0, x1, y1

    # Prefer the two largest faces (married couple); ignore tiny extras.
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    chosen = faces_sorted[:2]

    xs, ys, xe, ye = [], [], [], []
    for x, y, w, h in chosen:
        # Expand downward for torsos / dress / suit, and laterally a bit.
        xs.append(int(x - 0.55 * w))
        ys.append(int(y - 0.45 * h))
        xe.append(int(x + 1.55 * w))
        ye.append(int(y + 3.2 * h))

    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(width, max(xe))
    y1 = min(height, max(ye))
    return x0, y0, x1, y1


def soft_mask(width: int, height: int, roi: tuple[int, int, int, int], feather: float) -> np.ndarray:
    x0, y0, x1, y1 = roi
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    # Feather proportional to image size so edges don't hard-cut shapes.
    k = max(3, int(min(width, height) * feather) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # Remap: background stays slightly important so the scene doesn't vanish.
    return 0.12 + 0.88 * mask


def make_guided_target(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    # Strong blur for non-couple areas — cheap for primitive to approximate.
    k = max(21, (min(h, w) // 8) | 1)
    blurred = cv2.GaussianBlur(bgr, (k, k), 0)
    mean = bgr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    flat = (0.65 * blurred.astype(np.float32) + 0.35 * mean).astype(np.uint8)

    m = mask[..., None]
    guided = (bgr.astype(np.float32) * m + flat.astype(np.float32) * (1.0 - m)).astype(
        np.uint8
    )
    return guided


def overlay_debug(bgr: np.ndarray, faces, roi, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 80), 3)
    for x, y, w, h in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 120, 255), 2)
    heat = (mask * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
    return cv2.addWeighted(out, 0.72, heat_color, 0.28, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Source photo path")
    parser.add_argument(
        "-o",
        "--outdir",
        default=str(ROOT / "output"),
        help="Directory for guided target + debug assets",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=0.08,
        help="Soft-mask feather as fraction of min(image side)",
    )
    args = parser.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {src}")

    h, w = bgr.shape[:2]
    faces = detect_faces(bgr)
    roi = couple_roi_from_faces(faces, w, h)
    mask = soft_mask(w, h, roi, args.feather)
    guided = make_guided_target(bgr, mask)
    debug = overlay_debug(bgr, faces, roi, mask)

    guided_path = outdir / "guided_target.png"
    mask_path = outdir / "importance_mask.png"
    debug_path = outdir / "couple_roi_debug.png"
    original_path = outdir / "original.png"

    cv2.imwrite(str(guided_path), guided)
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(debug_path), debug)
    # Keep a normalized PNG copy of the original for compositing / comparison.
    Image.open(src).convert("RGB").save(original_path)

    print(f"faces_detected={len(faces)}")
    print(f"roi={roi}")
    print(f"wrote {guided_path}")
    print(f"wrote {mask_path}")
    print(f"wrote {debug_path}")
    print(f"wrote {original_path}")


if __name__ == "__main__":
    main()
