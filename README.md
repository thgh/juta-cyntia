# juta-cyntia

Geometric portrait using an **extension of [fogleman/primitive](https://github.com/fogleman/primitive)** —
same algorithm, plus a distance weight overlay. **Rotated ellipses only.**

## Result

| File | Description |
| --- | --- |
| [`output/primitive_rotated_ellipses.png`](output/primitive_rotated_ellipses.png) | Final artwork |
| [`output/primitive_rotated_ellipses.svg`](output/primitive_rotated_ellipses.svg) | Vector output |
| [`output/comparison.png`](output/comparison.png) | Original vs result |
| [`output/importance_weights_overlay.png`](output/importance_weights_overlay.png) | Weight map visualization |

## How the original algorithm is extended

`tools/wprimitive` is stock primitive with three additions:

1. **`-w weights.png`** — grayscale map encoded as importance **1…100**
2. **Weighted scoring** — pixel error is multiplied by the weight (faces cost more to get wrong)
3. **`-warmup N`** — first N shapes use **uniform** weights (normal coverage), then the map turns on for detail

Shape sizes stay in the original range (~1–32+); importance only *softly* skews proposals toward smaller ellipses on faces. No crop compositing (that caused the grey fog).

Distance map:

```text
weight = 1 + 9·body_m + 90·face_m   # ≈ bg 1×, body 10×, face 100×
```

`body_m` / `face_m` fall off smoothly with distance from body ROI / face cores.

## Run

```bash
./scripts/run_primitive_couple.sh input/couple.jpg
# or:
tools/wprimitive/wprimitive -i input/couple.jpg -w output/importance_weights.png \
  -warmup 400 -n 1500 -m 7 -r 720 -s 1600 -o output/out.png
```
