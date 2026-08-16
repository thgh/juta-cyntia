#!/usr/bin/env python3
"""Build a smooth distance-based importance map for weighted primitive.

Weights (relative to background = 1):
  - faces  ≈ 100
  - bodies ≈ 10
  - background = 1

Membership uses a distance falloff so importance declines gradually:
  weight = 1 + 9 * body_m + 90 * face_m
where body_m / face_m are smooth [0,1] fields from distance transforms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_couple_priority as couple  # noqa: E402


def face_core_mask(
    height: int, width: int, faces: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Elliptical face cores (slightly larger than Haar boxes)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in faces:
        cx, cy = int(x + w / 2), int(y + h / 2)
        axes = (max(1, int(w * 0.62)), max(1, int(h * 0.72)))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def body_core_mask(height: int, width: int, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    # Rounded body capsule inside the couple ROI.
    pad_x = int((x1 - x0) * 0.06)
    pad_y = int((y1 - y0) * 0.02)
    rr = max(8, int(min(x1 - x0, y1 - y0) * 0.08))
    cv2.rectangle(
        mask,
        (x0 + pad_x, y0 + pad_y),
        (x1 - pad_x, y1 - pad_y),
        255,
        -1,
    )
    # Soften corners via morphological close-ish blur later; radius hint:
    _ = rr
    return mask


def soft_membership(core: np.ndarray, sigma: float) -> np.ndarray:
    """1 inside core, smoothly decaying with distance outside (Gaussian of distance)."""
    # Distance to nearest core pixel; 0 inside.
    inv = np.where(core > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    if sigma <= 1e-6:
        return (core > 0).astype(np.float32)
    m = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    m = np.maximum(m, (core > 0).astype(np.float32))
    return np.clip(m, 0.0, 1.0)


def build_weight_map(
    height: int,
    width: int,
    faces: list[tuple[int, int, int, int]],
    body_roi: tuple[int, int, int, int],
    face_weight: float = 100.0,
    body_weight: float = 10.0,
    bg_weight: float = 1.0,
    face_sigma_scale: float = 0.55,
    body_sigma_scale: float = 0.12,
) -> np.ndarray:
    face_core = face_core_mask(height, width, faces)
    body_core = body_core_mask(height, width, body_roi)

    # Sigma from geometry so falloff is gradual but localized.
    if faces:
        mean_face = float(np.mean([max(f[2], f[3]) for f in faces]))
    else:
        mean_face = min(width, height) * 0.05
    face_sigma = max(4.0, mean_face * face_sigma_scale)

    bw = body_roi[2] - body_roi[0]
    bh = body_roi[3] - body_roi[1]
    body_sigma = max(8.0, min(bw, bh) * body_sigma_scale)

    face_m = soft_membership(face_core, face_sigma)
    body_m = soft_membership(body_core, body_sigma)

    # Ensure faces sit on top of body membership.
    body_m = np.maximum(body_m, face_m)

    # weight = bg + (body-bg)*body_m + (face-body)*face_m
    weights = (
        bg_weight
        + (body_weight - bg_weight) * body_m
        + (face_weight - body_weight) * face_m
    )
    return weights.astype(np.float32)


def encode_weight_png(weights: np.ndarray) -> np.ndarray:
    """Encode weight 1..100 as grayscale 0..255 for wprimitive -w."""
    # inverse of: w = 1 + (g/255)*99  => g = (w-1)/99*255
    g = np.clip((weights - 1.0) / 99.0 * 255.0, 0, 255).astype(np.uint8)
    return g


def weight_overlay(bgr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Visualize on log-ish scale so face/body/bg are distinguishable.
    norm = np.log1p(weights)
    norm = (norm / norm.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--outdir", default=str(ROOT / "output"))
    parser.add_argument("--face-weight", type=float, default=100.0)
    parser.add_argument("--body-weight", type=float, default=10.0)
    parser.add_argument("--bg-weight", type=float, default=1.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read {args.input}")
    h, w = bgr.shape[:2]

    faces_all = couple.detect_faces(bgr)
    chosen = couple.pick_couple_faces(faces_all, w, h)
    body_roi = couple.couple_roi_from_faces(faces_all, w, h)

    weights = build_weight_map(
        h,
        w,
        chosen,
        body_roi,
        face_weight=args.face_weight,
        body_weight=args.body_weight,
        bg_weight=args.bg_weight,
    )

    gray = encode_weight_png(weights)
    overlay = weight_overlay(bgr, weights)

    weight_path = outdir / "importance_weights.png"
    overlay_path = outdir / "importance_weights_overlay.png"
    meta_path = outdir / "importance_weights.json"
    original_path = outdir / "original.png"

    cv2.imwrite(str(weight_path), gray)
    cv2.imwrite(str(overlay_path), overlay)
    Image.open(args.input).convert("RGB").save(original_path)

    meta = {
        "faces": chosen,
        "body_roi": body_roi,
        "face_weight": args.face_weight,
        "body_weight": args.body_weight,
        "bg_weight": args.bg_weight,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"wrote {weight_path}")
    print(f"wrote {overlay_path}")


if __name__ == "__main__":
    main()
