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

	// Rejection sampling with accept prob (weight/WeightMax)^power.
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

// MaxRadiusForWeight returns a max ellipse radius that shrinks in important areas.
func MaxRadiusForWeight(localWeight float64, imageMinDim int) float64 {
	base := float64(imageMinDim) * 0.08
	if base < 18 {
		base = 18
	}
	if base > 40 {
		base = 40
	}
	if WeightMax <= 1 || localWeight <= 1 {
		return base
	}
	// faces (~100): ~2-4px, bodies (~10): ~8-12px, bg (~1): base
	t := math.Log(localWeight) / math.Log(WeightMax) // 0..1
	minR := 2.0
	return minR + (base-minR)*math.Pow(1.0-t, 2.2)
}
