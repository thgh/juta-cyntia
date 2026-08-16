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


def _face_center(face: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = face
    return x + w / 2.0, y + h / 2.0


def pick_couple_faces(
    faces: list[tuple[int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Pick the married couple faces, preferring a central side-by-side pair."""
    if not faces:
        return []

    cx, cy = width / 2.0, height / 2.0

    def score(face: tuple[int, int, int, int]) -> float:
        x, y, w, h = face
        fx, fy = _face_center(face)
        # Prefer reasonably large faces near the photo center (ignore blur/tree hits).
        size = (w * h) / float(width * height)
        dx = abs(fx - cx) / width
        dy = abs(fy - cy) / height
        return size * 4.0 - (dx * 1.6 + dy * 1.1)

    ranked = sorted(faces, key=score, reverse=True)

    # Search for a plausible couple: two similarly sized faces, side by side,
    # both near the middle of the frame.
    best_pair = None
    best_pair_score = -1e9
    top = ranked[: min(6, len(ranked))]
    for i, a in enumerate(top):
        ax, ay = _face_center(a)
        aw, ah = a[2], a[3]
        for b in top[i + 1 :]:
            bx, by = _face_center(b)
            bw, bh = b[2], b[3]
            size_ratio = max(aw * ah, bw * bh) / max(1.0, min(aw * ah, bw * bh))
            if size_ratio > 2.8:
                continue
            # Couple stands roughly shoulder-to-shoulder.
            if abs(ay - by) > height * 0.12:
                continue
            horiz = abs(ax - bx) / width
            if horiz < 0.04 or horiz > 0.45:
                continue
            mid_x = (ax + bx) / 2.0
            mid_y = (ay + by) / 2.0
            pair_score = (
                score(a)
                + score(b)
                - abs(mid_x - cx) / width
                - abs(mid_y - cy) / height * 0.5
                - abs(size_ratio - 1.0)
            )
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = [a, b]

    if best_pair:
        return best_pair
    return ranked[:1]


def couple_roi_from_faces(
    faces: list[tuple[int, int, int, int]], width: int, height: int
) -> tuple[int, int, int, int]:
    """Return expanded bounding box covering faces + full wedding attire."""
    chosen = pick_couple_faces(faces, width, height)
    if not chosen:
        # Fallback: central full-length portrait framing.
        x0 = int(width * 0.22)
        x1 = int(width * 0.78)
        y0 = int(height * 0.12)
        y1 = int(height * 0.92)
        return x0, y0, x1, y1

    xs, ys, xe, ye = [], [], [], []
    for x, y, w, h in chosen:
        # Expand for hair, torsos, full dress/suit length, and ground underfoot.
        xs.append(int(x - 1.25 * w))
        ys.append(int(y - 0.85 * h))
        xe.append(int(x + 2.25 * w))
        ye.append(int(y + 14.0 * h))

    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(width, max(xe))
    y1 = min(height, max(ye))

    # Ensure the ROI stays centered enough to cover both people fully.
    min_width = int(width * 0.34)
    if x1 - x0 < min_width:
        mid = (x0 + x1) // 2
        x0 = max(0, mid - min_width // 2)
        x1 = min(width, x0 + min_width)
    return x0, y0, x1, y1


def soft_mask(width: int, height: int, roi: tuple[int, int, int, int], feather: float) -> np.ndarray:
    x0, y0, x1, y1 = roi
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    # Feather proportional to image size so edges don't hard-cut shapes.
    k = max(3, int(min(width, height) * feather) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # Keep surroundings somewhat important (tree + teal motion blur), but
    # still bias residual error toward the couple.
    return 0.35 + 0.65 * mask


def make_guided_target(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    # Mild blur outside the couple — preserves color/structure of bridesmaids + tree.
    k = max(15, (min(h, w) // 14) | 1)
    blurred = cv2.GaussianBlur(bgr, (k, k), 0)

    m = mask[..., None]
    guided = (bgr.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)).astype(
        np.uint8
    )
    return guided


def overlay_debug(
    bgr: np.ndarray,
    faces,
    chosen,
    roi,
    mask: np.ndarray,
) -> np.ndarray:
    out = bgr.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 80), 3)
    for x, y, w, h in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 120, 255), 1)
    for x, y, w, h in chosen:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 3)
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
    chosen = pick_couple_faces(faces, w, h)
    roi = couple_roi_from_faces(faces, w, h)
    mask = soft_mask(w, h, roi, args.feather)
    guided = make_guided_target(bgr, mask)
    debug = overlay_debug(bgr, faces, chosen, roi, mask)

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
    print(f"couple_faces={chosen}")
    print(f"roi={roi}")
    print(f"wrote {guided_path}")
    print(f"wrote {mask_path}")
    print(f"wrote {debug_path}")
    print(f"wrote {original_path}")


if __name__ == "__main__":
    main()
