#!/usr/bin/env bash
# Run fogleman/primitive with rotated ellipses, prioritizing the married couple.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/ubuntu/go/bin:${HOME}/go/bin"

INPUT="${1:-}"
if [[ -z "${INPUT}" ]]; then
  # Auto-pick first image under input/
  INPUT="$(find "${ROOT}/input" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
    | sort | head -n 1 || true)"
fi

if [[ -z "${INPUT}" || ! -f "${INPUT}" ]]; then
  echo "ERROR: No source photo found."
  echo "Place the married-couple photo at input/couple.jpg (or pass a path)."
  exit 1
fi

N="${N:-500}"          # number of rotated ellipses
R="${R:-512}"          # working/input resize (higher = more detail, slower)
S="${S:-1600}"         # output size
A="${A:-128}"          # alpha

OUTDIR="${ROOT}/output"
mkdir -p "${OUTDIR}" "${ROOT}/input"

echo "==> Preparing couple-priority guided target from: ${INPUT}"
python3 "${ROOT}/scripts/prepare_couple_priority.py" -i "${INPUT}" -o "${OUTDIR}"

GUIDED="${OUTDIR}/guided_target.png"
FINAL_PNG="${OUTDIR}/primitive_rotated_ellipses.png"
FINAL_SVG="${OUTDIR}/primitive_rotated_ellipses.svg"
FINAL_GIF="${OUTDIR}/primitive_rotated_ellipses.gif"

echo "==> Running primitive (rotated ellipses only, n=${N}, r=${R}, s=${S})"
# Mode 7 = rotated ellipse
primitive \
  -i "${GUIDED}" \
  -o "${FINAL_PNG}" \
  -o "${FINAL_SVG}" \
  -o "${OUTDIR}/frames/frame_%03d.png" \
  -n "${N}" \
  -m 7 \
  -r "${R}" \
  -s "${S}" \
  -a "${A}" \
  -nth 25 \
  -v

# Optional animated GIF of progress (if enough frames)
if compgen -G "${OUTDIR}/frames/frame_*.png" > /dev/null; then
  if command -v convert >/dev/null 2>&1; then
    convert -delay 12 -loop 0 "${OUTDIR}/frames/frame_*.png" -delay 80 "${FINAL_PNG}" "${FINAL_GIF}" || true
  fi
fi

# Side-by-side comparison strip
python3 - <<PY
from PIL import Image
from pathlib import Path
out = Path("${OUTDIR}")
orig = Image.open(out / "original.png").convert("RGB")
result = Image.open(out / "primitive_rotated_ellipses.png").convert("RGB")
h = 800
def fit(im):
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.Resampling.LANCZOS)
a, b = fit(orig), fit(result)
canvas = Image.new("RGB", (a.width + b.width + 24, h + 24), (24, 24, 24))
canvas.paste(a, (12, 12))
canvas.paste(b, (a.width + 24, 12))
canvas.save(out / "comparison.png")
print("wrote", out / "comparison.png")
PY

echo "==> Done"
echo "    ${FINAL_PNG}"
echo "    ${FINAL_SVG}"
ls -lh "${FINAL_PNG}" "${FINAL_SVG}" "${OUTDIR}/comparison.png" "${OUTDIR}/couple_roi_debug.png"
