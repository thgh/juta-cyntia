#!/usr/bin/env bash
# Distance-weighted primitive (rotated ellipses):
#   faces ≈ 100× background, bodies ≈ 10× background
# Scene pass keeps background quality; refinement passes use the weight map.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/ubuntu/go/bin:${HOME}/go/bin:${ROOT}/tools/wprimitive"

INPUT="${1:-${ROOT}/input/couple.jpg}"
OUTDIR="${ROOT}/output"
mkdir -p "${OUTDIR}/body" "${OUTDIR}/faces"

SCENE_N="${SCENE_N:-480}"
BODY_N="${BODY_N:-600}"
FACE_N="${FACE_N:-1000}"
FACE_A="${FACE_A:-64}"
R="${R:-750}"
S="${S:-1600}"
A="${A:-128}"

WPRIM="${ROOT}/tools/wprimitive/wprimitive"
echo "Building wprimitive..."
(cd "${ROOT}/tools/wprimitive" && GOTOOLCHAIN=local go build -o wprimitive .)

echo "==> Distance importance map (faces=100x, bodies=10x, bg=1x)"
python3 "${ROOT}/scripts/build_importance_weights.py" -i "${INPUT}" -o "${OUTDIR}"

echo "==> Pass 1: scene (unweighted — keep background quality), n=${SCENE_N}"
"${WPRIM}" \
  -i "${INPUT}" \
  -o "${OUTDIR}/pass_scene.png" \
  -o "${OUTDIR}/primitive_rotated_ellipses.svg" \
  -n "${SCENE_N}" \
  -m 7 \
  -r "${R}" \
  -s "${S}" \
  -a "${A}" \
  -v

echo "==> Preparing body + face crops with distance weights"
python3 - <<'PY'
import json, cv2, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/scripts')
from build_importance_weights import build_weight_map, encode_weight_png
import prepare_couple_priority as couple

out = Path('/workspace/output')
meta = json.loads((out/'importance_weights.json').read_text())
bgr = cv2.imread(str(out/'original.png'))
h, w = bgr.shape[:2]
faces = meta['faces']
body_roi = meta['body_roi']
x0,y0,x1,y1 = body_roi

# Body crop with a little padding
pad = 12
bx0, by0 = max(0, x0-pad), max(0, y0-pad)
bx1, by1 = min(w, x1+pad), min(h, y1+pad)
body = bgr[by0:by1, bx0:bx1]
cv2.imwrite(str(out/'body'/'body_crop.png'), body)

# Weights in full-image space, then crop
weights = build_weight_map(h, w, faces, body_roi, face_weight=100, body_weight=10, bg_weight=1)
body_w = weights[by0:by1, bx0:bx1]
cv2.imwrite(str(out/'body'/'body_weights.png'), encode_weight_png(body_w))
(out/'body'/'body_roi.json').write_text(json.dumps({'roi':[bx0,by0,bx1,by1],'size':[w,h]}))

# Face band
pads=[]
for x,y,fw,fh in faces:
    pads.append((max(0,int(x-1.5*fw)), max(0,int(y-1.3*fh)),
                 min(w,int(x+2.5*fw)), min(h,int(y+2.8*fh))))
fx0=min(p[0] for p in pads); fy0=min(p[1] for p in pads)
fx1=max(p[2] for p in pads); fy1=max(p[3] for p in pads)
faces_img = bgr[fy0:fy1, fx0:fx1]
cv2.imwrite(str(out/'faces'/'faces_crop.png'), faces_img)

# Face weights: pure distance-to-face peaks at 100
ch, cw = faces_img.shape[:2]
yy, xx = np.mgrid[0:ch, 0:cw]
fwmap = np.ones((ch, cw), np.float32)
for x,y,fw,fh in faces:
    cx = (x + fw/2) - fx0
    cy = (y + fh/2) - fy0
    sigma = max(fw, fh) * 0.65
    m = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / (sigma**2))
    fwmap = np.maximum(fwmap, 1.0 + 99.0 * m)
cv2.imwrite(str(out/'faces'/'faces_weights.png'), encode_weight_png(fwmap))
(out/'faces'/'faces_roi.json').write_text(json.dumps({'roi':[fx0,fy0,fx1,fy1],'size':[w,h]}))
print('body', body.shape, 'faces', faces_img.shape)
PY

echo "==> Pass 2: body crop with distance weights, n=${BODY_N}"
"${WPRIM}" \
  -i "${OUTDIR}/body/body_crop.png" \
  -w "${OUTDIR}/body/body_weights.png" \
  -o "${OUTDIR}/body/body_primitive.png" \
  -n "${BODY_N}" \
  -m 7 \
  -r 640 \
  -s 1200 \
  -a "${A}" \
  -v

echo "==> Pass 3: faces crop with distance weights, n=${FACE_N}, a=${FACE_A}"
"${WPRIM}" \
  -i "${OUTDIR}/faces/faces_crop.png" \
  -w "${OUTDIR}/faces/faces_weights.png" \
  -o "${OUTDIR}/faces/faces_primitive.png" \
  -n "${FACE_N}" \
  -m 7 \
  -r 640 \
  -s 1200 \
  -a "${FACE_A}" \
  -v

echo "==> Composite body then faces onto scene"
python3 - <<'PY'
import json
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

out = Path('/workspace/output')

def composite(base_path, overlay_path, roi_json, soft='rect'):
    meta = json.loads(Path(roi_json).read_text())
    x0,y0,x1,y1 = meta['roi']
    ow, oh = meta['size']
    base = Image.open(base_path).convert('RGBA')
    overlay = Image.open(overlay_path).convert('RGBA')
    sx, sy = base.width/ow, base.height/oh
    dx0, dy0 = int(x0*sx), int(y0*sy)
    dx1, dy1 = int(x1*sx), int(y1*sy)
    tw, th = max(1, dx1-dx0), max(1, dy1-dy0)
    ov = overlay.resize((tw, th), Image.Resampling.LANCZOS)
    mask = Image.new('L', (tw, th), 0)
    d = ImageDraw.Draw(mask)
    pad = int(min(tw, th) * (0.08 if soft=='rect' else 0.10))
    if soft == 'ellipse':
        d.ellipse((pad, pad, tw-pad-1, th-pad-1), fill=255)
        blur = max(8, pad)
    else:
        d.rounded_rectangle((pad, pad, tw-pad-1, th-pad-1), radius=max(10, pad*2), fill=255)
        blur = max(6, pad)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    out_im = base.copy()
    out_im.paste(ov, (dx0, dy0), mask)
    return out_im

scene = out/'pass_scene.png'
mid = composite(scene, out/'body'/'body_primitive.png', out/'body'/'body_roi.json', soft='rect')
mid_path = out/'pass_scene_body.png'
mid.convert('RGB').save(mid_path)
final = composite(mid_path, out/'faces'/'faces_primitive.png', out/'faces'/'faces_roi.json', soft='ellipse')
final.convert('RGB').save(out/'primitive_rotated_ellipses.png', quality=95)

orig = Image.open(out/'original.png').convert('RGB')
h=900
def fit(im):
    w=int(im.width*h/im.height)
    return im.resize((w,h), Image.Resampling.LANCZOS)
a,b = fit(orig), fit(final.convert('RGB'))
canvas = Image.new('RGB', (a.width+b.width+24, h+24), (24,24,24))
canvas.paste(a,(12,12)); canvas.paste(b,(a.width+24,12))
canvas.save(out/'comparison.png')
print('wrote primitive_rotated_ellipses.png + comparison.png')
PY

ls -lh "${OUTDIR}/primitive_rotated_ellipses.png" "${OUTDIR}/comparison.png" "${OUTDIR}/importance_weights_overlay.png"
echo "==> Done"
