# juta-cyntia

Geometric portrait of the married couple using a **distance-weighted** fork of
[fogleman/primitive](https://github.com/fogleman/primitive) — **rotated ellipses only**.

## Result

| File | Description |
| --- | --- |
| [`output/primitive_rotated_ellipses.png`](output/primitive_rotated_ellipses.png) | Final artwork |
| [`output/primitive_rotated_ellipses.svg`](output/primitive_rotated_ellipses.svg) | Vector scene pass |
| [`output/comparison.png`](output/comparison.png) | Original vs result |
| [`output/importance_weights_overlay.png`](output/importance_weights_overlay.png) | Distance importance visualization |
| [`input/couple.jpg`](input/couple.jpg) | Source photo |

## Distance importance

Per-pixel weight uses smooth distance falloff from face cores and the body ROI:

```
weight = 1 + 9·body_m + 90·face_m
```

So approximately:

| Region | Weight vs background |
| --- | --- |
| Faces | **~100×** |
| Bodies | **~10×** |
| Background | **1×** |

`body_m` / `face_m` are soft memberships from a distance transform
(`exp(-½ (d/σ)²)`), so importance declines gradually instead of hard cuts.

These weights drive:

1. **Error scoring** in `tools/wprimitive` (face mistakes cost ~100× more)
2. **Shape proposals** (centers sampled toward important pixels; radii shrink there)

## Pipeline

1. Build the distance weight map
2. Unweighted scene pass (keeps background looking good)
3. Body crop pass with the weight map (~10×)
4. Face crop pass with the weight map (~100×, lower alpha)
5. Soft-mask composite back onto the scene

```bash
./scripts/run_primitive_couple.sh input/couple.jpg
```
