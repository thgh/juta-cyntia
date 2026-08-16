# juta-cyntia

Geometric portrait of the married couple using [fogleman/primitive](https://github.com/fogleman/primitive).

## Result

| File | Description |
| --- | --- |
| [`output/primitive_rotated_ellipses.png`](output/primitive_rotated_ellipses.png) | Final artwork (rotated ellipses only) |
| [`output/primitive_rotated_ellipses.svg`](output/primitive_rotated_ellipses.svg) | Vector version of the scene pass |
| [`output/comparison.png`](output/comparison.png) | Original vs result |
| [`input/couple.jpg`](input/couple.jpg) | Source wedding photo |

## Approach

- **Shapes:** rotated ellipses only (`primitive -m 7`)
- **Couple priority:**
  1. Detect the couple’s faces and build a soft full-body ROI
  2. Mildly guide the full scene so residual error favors the couple
  3. Pass 1: full-scene primitives
  4. Pass 2: high-detail couple crop composited back
  5. Pass 3: face-band refinement composited with a soft elliptical mask

## Re-run

```bash
./scripts/run_primitive_couple.sh input/couple.jpg
```

Optional knobs: `SCENE_N`, `COUPLE_N`, `R`, `S`.
