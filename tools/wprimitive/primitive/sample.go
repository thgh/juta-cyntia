package primitive

import (
	"math"
	"math/rand"
)

// SampleWeightedPosition picks a pixel with probability proportional to weight^power.
// power>1 further concentrates proposals on faces.
func SampleWeightedPosition(rnd *rand.Rand, w, h int, power float64) (x, y, localWeight float64) {
	if PixelWeights == nil || WeightCDF == nil || WeightSum <= 0 {
		xf := rnd.Float64() * float64(w)
		yf := rnd.Float64() * float64(h)
		return xf, yf, 1
	}

	if power < 1 {
		power = 1
	}
	for tries := 0; tries < 64; tries++ {
		xf := rnd.Float64() * float64(w)
		yf := rnd.Float64() * float64(h)
		xi := int(xf)
		yi := int(yf)
		if xi >= w {
			xi = w - 1
		}
		if yi >= h {
			yi = h - 1
		}
		wt := pixelWeight(xi, yi, w)
		p := math.Pow(wt/WeightMax, power)
		if rnd.Float64() <= p {
			return xf, yf, wt
		}
	}
	// Fallback: CDF binary search (linear weights)
	r := rnd.Float64() * WeightSum
	lo, hi := 0, len(WeightCDF)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if WeightCDF[mid] < r {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	yi := lo / w
	xi := lo % w
	return float64(xi) + 0.5, float64(yi) + 0.5, pixelWeight(xi, yi, w)
}

// SampleEllipseRadii returns rx, ry in roughly the original algorithm's range
// (about 1..32+), with only a soft skew toward smaller shapes in high-weight
// areas. Large shapes remain possible everywhere so the canvas stays covered.
func SampleEllipseRadii(rnd *rand.Rand, localWeight float64, imageMinDim int) (rx, ry float64) {
	base := 32.0
	if imageMinDim < 96 {
		base = float64(imageMinDim) / 3
	}
	// Power > 1 skews rand toward 0 (smaller). Faces get a stronger skew,
	// background stays close to uniform like stock primitive.
	power := 1.0
	if WeightMax > 1 && localWeight > 1 {
		t := math.Log(localWeight) / math.Log(WeightMax) // 0..1
		power = 1.0 + 1.6*t                             // bg~1, body~1.7, face~2.6
	}
	rx = math.Pow(rnd.Float64(), power)*base + 1
	ry = math.Pow(rnd.Float64(), power)*base + 1
	return rx, ry
}
