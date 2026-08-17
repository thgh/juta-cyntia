#!/usr/bin/env bash
# Single-pass extension of fogleman/primitive:
# the original algorithm + a distance weight overlay for scoring/proposals.
# Rotated ellipses only. No crop compositing (avoids grey fog).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/ubuntu/go/bin:${HOME}/go/bin:${ROOT}/tools/wprimitive"

INPUT="${1:-${ROOT}/input/couple.jpg}"
OUTDIR="${ROOT}/output"
mkdir -p "${OUTDIR}"

N="${N:-1800}"
WARMUP="${WARMUP:-500}"
R="${R:-800}"
S="${S:-1600}"
# a=0 lets primitive pick alpha per shape (better coverage, less foggy wash)
A="${A:-0}"

WPRIM="${ROOT}/tools/wprimitive/wprimitive"
echo "Building wprimitive..."
(cd "${ROOT}/tools/wprimitive" && GOTOOLCHAIN=local go build -o wprimitive .)

echo "==> Distance importance map (faces≈100x, bodies≈10x, bg=1x)"
python3 "${ROOT}/scripts/build_importance_weights.py" -i "${INPUT}" -o "${OUTDIR}"

echo "==> Single-pass extended primitive (warmup=${WARMUP}, then weighted, n=${N})"
"${WPRIM}" \
  -i "${INPUT}" \
  -w "${OUTDIR}/importance_weights.png" \
  -warmup "${WARMUP}" \
  -o "${OUTDIR}/primitive_rotated_ellipses.png" \
  -o "${OUTDIR}/primitive_rotated_ellipses.svg" \
  -n "${N}" \
  -m 7 \
  -r "${R}" \
  -s "${S}" \
  -a "${A}" \
  -v

echo "==> Comparison strip"
python3 - <<'PY'
from PIL import Image
from pathlib import Path
out = Path('/workspace/output')
orig = Image.open(out / 'original.png').convert('RGB')
result = Image.open(out / 'primitive_rotated_ellipses.png').convert('RGB')
h = 900
def fit(im):
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.Resampling.LANCZOS)
a, b = fit(orig), fit(result)
canvas = Image.new('RGB', (a.width + b.width + 24, h + 24), (24, 24, 24))
canvas.paste(a, (12, 12))
canvas.paste(b, (a.width + 24, 12))
canvas.save(out / 'comparison.png')
print('wrote', out / 'comparison.png')
PY

ls -lh \
  "${OUTDIR}/primitive_rotated_ellipses.png" \
  "${OUTDIR}/comparison.png" \
  "${OUTDIR}/importance_weights_overlay.png"
echo "==> Done"
