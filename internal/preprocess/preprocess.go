package preprocess

import (
	"fmt"
	"image"
	"log"

	"gocv.io/x/gocv"
)

// ProcessLabel runs the full preprocessing chain:
// MSER region crop → grayscale → Otsu binarize → Gaussian denoise → 2× upscale
func ProcessLabel(imgBytes []byte) ([]byte, error) {
	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if mat.Empty() {
		return nil, fmt.Errorf("image is empty after decode")
	}
	defer mat.Close()

	cropped := cropMSER(&mat)
	defer cropped.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(cropped, &gray, gocv.ColorBGRToGray)

	// Denoise before binarizing — blurring a binary image creates gray fringe
	// pixels at character edges that confuse Tesseract on small text.
	denoised := gocv.NewMat()
	defer denoised.Close()
	gocv.GaussianBlur(gray, &denoised, image.Pt(3, 3), 0, 0, gocv.BorderDefault)

	binary := gocv.NewMat()
	defer binary.Close()
	gocv.Threshold(denoised, &binary, 0, 255, gocv.ThresholdBinary+gocv.ThresholdOtsu)

	h, w := binary.Rows(), binary.Cols()
	upscaled := gocv.NewMat()
	defer upscaled.Close()
	gocv.Resize(binary, &upscaled, image.Pt(w*2, h*2), 0, 0, gocv.InterpolationCubic)

	buf, err := gocv.IMEncodeWithParams(".jpg", upscaled, []int{gocv.IMWriteJpegQuality, 95})
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	return buf.GetBytes(), nil
}

// cropMSER uses MSER keypoints to find text regions and returns the bounding crop.
// Falls back to the full image if no keypoints are detected.
// GoCV's MSER only exposes the Feature2D interface (Detect), not DetectRegions,
// so we derive the bounding rect from keypoint center positions.
func cropMSER(mat *gocv.Mat) gocv.Mat {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(*mat, &gray, gocv.ColorBGRToGray)

	mser := gocv.NewMSER()
	defer mser.Close()

	kps := mser.Detect(gray)
	if len(kps) == 0 {
		log.Println("preprocess: MSER found no keypoints, using full image")
		return mat.Clone()
	}

	minX, minY := int(kps[0].X), int(kps[0].Y)
	maxX, maxY := minX, minY
	for _, kp := range kps[1:] {
		x, y := int(kp.X), int(kp.Y)
		if x < minX {
			minX = x
		}
		if y < minY {
			minY = y
		}
		if x > maxX {
			maxX = x
		}
		if y > maxY {
			maxY = y
		}
	}

	const pad = 20
	h, w := mat.Rows(), mat.Cols()
	x1 := max(0, minX-pad)
	y1 := max(0, minY-pad)
	x2 := min(w, maxX+pad)
	y2 := min(h, maxY+pad)

	log.Printf("preprocess: MSER crop (%d,%d)→(%d,%d)", x1, y1, x2, y2)
	return mat.Region(image.Rect(x1, y1, x2, y2))
}
