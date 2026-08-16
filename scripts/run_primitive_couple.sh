#!/usr/bin/env bash
# Two-pass fogleman/primitive render:
# 1) full scene with rotated ellipses
# 2) high-detail couple crop composited back with a soft mask
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/ubuntu/go/bin:${HOME}/go/bin"

INPUT="${1:-${ROOT}/input/couple.jpg}"
OUTDIR="${ROOT}/output"
mkdir -p "${OUTDIR}/frames" "${OUTDIR}/couple_pass"

SCENE_N="${SCENE_N:-420}"
COUPLE_N="${COUPLE_N:-380}"
R="${R:-520}"
S="${S:-1600}"
A="${A:-128}"

echo "==> Preparing couple-priority assets"
python3 "${ROOT}/scripts/prepare_couple_priority.py" -i "${INPUT}" -o "${OUTDIR}"

# Read ROI from a tiny sidecar written by prepare (fallback parse from debug isn't needed:
# recompute ROI values via python for the crop)
python3 - <<PY
import cv2, json
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}/scripts")
from prepare_couple_priority import detect_faces, couple_roi_from_faces

bgr = cv2.imread("${OUTDIR}/original.png")
h, w = bgr.shape[:2]
faces = detect_faces(bgr)
roi = couple_roi_from_faces(faces, w, h)
Path("${OUTDIR}/couple_roi.json").write_text(json.dumps({"roi": roi, "size": [w, h]}))
x0,y0,x1,y1 = roi
crop = bgr[y0:y1, x0:x1]
cv2.imwrite("${OUTDIR}/couple_pass/couple_crop.png", crop)
print("roi", roi, "crop", crop.shape)
PY

echo "==> Pass 1: full scene (${SCENE_N} rotated ellipses)"
primitive \
  -i "${OUTDIR}/guided_target.png" \
  -o "${OUTDIR}/pass1_scene.png" \
  -o "${OUTDIR}/primitive_rotated_ellipses.svg" \
  -n "${SCENE_N}" \
  -m 7 \
  -r "${R}" \
  -s "${S}" \
  -a "${A}" \
  -v

echo "==> Pass 2: couple crop detail (${COUPLE_N} rotated ellipses)"
# Higher working resolution relative to crop size for facial accuracy
primitive \
  -i "${OUTDIR}/couple_pass/couple_crop.png" \
  -o "${OUTDIR}/couple_pass/couple_primitive.png" \
  -n "${COUPLE_N}" \
  -m 7 \
  -r 480 \
  -s 1200 \
  -a "${A}" \
  -v

echo "==> Compositing couple detail onto scene"
python3 - <<'PY'
import json
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

out = Path('/workspace/output')
meta = json.loads((out / 'couple_roi.json').read_text())
x0, y0, x1, y1 = meta['roi']
orig_w, orig_h = meta['size']

scene = Image.open(out / 'pass1_scene.png').convert('RGBA')
couple = Image.open(out / 'couple_pass' / 'couple_primitive.png').convert('RGBA')
# Map ROI from original coordinates into output scene coordinates
sx = scene.width / orig_w
sy = scene.height / orig_h
dx0, dy0 = int(x0 * sx), int(y0 * sy)
dx1, dy1 = int(x1 * sx), int(y1 * sy)
target_w, target_h = max(1, dx1 - dx0), max(1, dy1 - dy0)
couple_r = couple.resize((target_w, target_h), Image.Resampling.LANCZOS)

# Soft alpha mask so the refinement blends cleanly
mask = Image.new('L', (target_w, target_h), 0)
draw = ImageDraw.Draw(mask)
pad = int(min(target_w, target_h) * 0.06)
draw.rounded_rectangle(
    (pad, pad, target_w - pad - 1, target_h - pad - 1),
    radius=max(8, pad * 2),
    fill=255,
)
mask = mask.filter(ImageFilter.GaussianBlur(radius=max(4, pad)))

base = scene.copy()
base.paste(couple_r, (dx0, dy0), mask)
final = base.convert('RGB')
final.save(out / 'primitive_rotated_ellipses.png', quality=95)

# Comparison strip
orig = Image.open(out / 'original.png').convert('RGB')
h = 900
def fit(im):
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.Resampling.LANCZOS)
a, b = fit(orig), fit(final)
canvas = Image.new('RGB', (a.width + b.width + 24, h + 24), (24, 24, 24))
canvas.paste(a, (12, 12))
canvas.paste(b, (a.width + 24, 12))
canvas.save(out / 'comparison.png')
print('wrote', out / 'primitive_rotated_ellipses.png')
print('wrote', out / 'comparison.png')
PY

# SVG from the scene pass (couple detail pass is raster-composited)
cp "${OUTDIR}/pass1_scene.png" "${OUTDIR}/primitive_scene_only.png"

ls -lh "${OUTDIR}/primitive_rotated_ellipses.png" "${OUTDIR}/comparison.png" "${OUTDIR}/couple_roi_debug.png" "${OUTDIR}/primitive_rotated_ellipses.svg"
echo "==> Done"
