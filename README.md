# juta-cyntia

Geometric portrait of the married couple using [fogleman/primitive](https://github.com/fogleman/primitive).

## Goal

- Shape mode: **rotated ellipses only** (`-m 7`)
- Optimize for **accuracy of the couple** over background / surroundings

## How it works

1. `scripts/prepare_couple_priority.py` detects faces, builds a soft couple ROI, and creates a *guided target* (couple stays sharp; the rest is blurred/flattened so residual error concentrates on the couple).
2. `scripts/run_primitive_couple.sh` runs `primitive` with hundreds of rotated ellipses on that guided target.

## Usage

1. Put the source photo at `input/couple.jpg` (or `.png`).
2. Run:

```bash
./scripts/run_primitive_couple.sh input/couple.jpg
```

Outputs land in `output/`:

| File | Description |
| --- | --- |
| `primitive_rotated_ellipses.png` | Final artwork |
| `primitive_rotated_ellipses.svg` | Vector version |
| `couple_roi_debug.png` | Detected couple ROI / importance overlay |
| `comparison.png` | Original vs result |

Optional env knobs: `N` (shape count, default 500), `R` (working size, default 512), `S` (output size, default 1600).

```bash
N=800 R=640 S=2048 ./scripts/run_primitive_couple.sh input/couple.jpg
```
