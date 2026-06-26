package agent

import (
	"bytes"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// SpiritClass is a recognized beverage category.
type SpiritClass struct {
	Category   string  // "BOURBON WHISKEY", "WINE", "VODKA/GIN", etc.
	Confidence float64 // 0–1
}

// ClassifySpirit makes a pre-OCR guess at spirit category using three visual
// signals: bottle/glass color, label region darkness, and aspect ratio.
//
// These signals are surprisingly reliable:
//   - Clear glass → vodka or gin (almost always)
//   - Amber/brown glass pixels → whiskey/bourbon
//   - Dark green glass + tall shape → wine
//   - Maroon/burgundy label region → red wine
//   - Tall slender aspect ratio (h/w > 2.0) → wine bottle shape
func ClassifySpirit(imgBytes []byte) SpiritClass {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return SpiritClass{"UNKNOWN", 0}
	}

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w == 0 || h == 0 {
		return SpiritClass{"UNKNOWN", 0}
	}
	aspectRatio := float64(h) / float64(w)

	// Sample every 8th pixel for speed (~30k samples on a 1500px image).
	const stride = 8
	var (
		totalSamples int
		clearCount   int // S < 0.12 — clear/transparent glass
		amberCount   int // hue 18–50°, S > 0.25 — amber/brown (whiskey)
		greenCount   int // hue 90–170°, S > 0.25 — green glass (wine)
		redCount     int // hue < 18° or > 340°, S > 0.25 — maroon/red (wine label)
	)

	// Label region = center 40% of image width and middle 60% of height.
	lx0 := b.Min.X + w*3/10
	lx1 := b.Min.X + w*7/10
	ly0 := b.Min.Y + h*2/10
	ly1 := b.Min.Y + h*8/10
	var labelDarkCount, labelTotal int

	for y := b.Min.Y; y < b.Max.Y; y += stride {
		for x := b.Min.X; x < b.Max.X; x += stride {
			r32, g32, bl32, _ := img.At(x, y).RGBA()
			r := uint8(r32 >> 8)
			g := uint8(g32 >> 8)
			bv := uint8(bl32 >> 8)

			hue, sat, val := rgbToHSV(r, g, bv)
			totalSamples++

			if sat < 0.12 {
				clearCount++
			} else if hue >= 18 && hue <= 50 && sat > 0.25 {
				amberCount++
			} else if hue >= 90 && hue <= 170 && sat > 0.25 {
				greenCount++
			} else if (hue < 18 || hue > 340) && sat > 0.25 {
				redCount++
			}

			if x >= lx0 && x < lx1 && y >= ly0 && y < ly1 {
				labelTotal++
				if val < 0.30 {
					labelDarkCount++
				}
			}
		}
	}

	if totalSamples == 0 {
		return SpiritClass{"UNKNOWN", 0}
	}

	clearRatio := float64(clearCount) / float64(totalSamples)
	amberRatio := float64(amberCount) / float64(totalSamples)
	greenRatio := float64(greenCount) / float64(totalSamples)
	redRatio := float64(redCount) / float64(totalSamples)
	labelDarkRatio := 0.0
	if labelTotal > 0 {
		labelDarkRatio = float64(labelDarkCount) / float64(labelTotal)
	}

	// Classification rules in priority order.
	switch {
	case clearRatio > 0.40:
		return SpiritClass{"VODKA / GIN", clamp(0.65 + clearRatio*0.4)}

	case amberRatio > 0.20:
		conf := clamp(0.60 + amberRatio*0.8)
		if labelDarkRatio > 0.50 {
			conf = clamp(conf + 0.10)
		}
		return SpiritClass{"BOURBON / WHISKEY", conf}

	case greenRatio > 0.15 && aspectRatio > 1.8:
		if redRatio > 0.08 {
			return SpiritClass{"RED WINE", clamp(0.60 + greenRatio)}
		}
		return SpiritClass{"WINE", clamp(0.55 + greenRatio)}

	case redRatio > 0.15 && aspectRatio > 1.8:
		return SpiritClass{"RED WINE", clamp(0.55 + redRatio)}

	case aspectRatio > 2.0:
		return SpiritClass{"WINE", 0.50}

	case labelDarkRatio > 0.55:
		return SpiritClass{"DARK SPIRITS", 0.45}

	default:
		return SpiritClass{"SPIRITS", 0.30}
	}
}

func clamp(v float64) float64 {
	return math.Min(1.0, math.Max(0.0, math.Round(v*100)/100))
}

func rgbToHSV(r, g, b uint8) (h, s, v float64) {
	rf := float64(r) / 255
	gf := float64(g) / 255
	bf := float64(b) / 255

	mx := math.Max(math.Max(rf, gf), bf)
	mn := math.Min(math.Min(rf, gf), bf)
	delta := mx - mn

	v = mx
	if mx == 0 {
		return 0, 0, 0
	}
	s = delta / mx
	if delta == 0 {
		return 0, s, v
	}

	switch mx {
	case rf:
		h = 60 * math.Mod((gf-bf)/delta, 6)
	case gf:
		h = 60 * ((bf-rf)/delta + 2)
	default:
		h = 60 * ((rf-gf)/delta + 4)
	}
	if h < 0 {
		h += 360
	}
	return
}
