package agent

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"ttb-label-verifier/internal/models"
)

const ollamaURL = "http://localhost:11434/api/chat"

var (
	systemPrompt string
	jsonFence    = regexp.MustCompile(`(?s)` + "```" + `(?:json)?\s*(.*?)` + "```")
)

func init() {
	data, err := os.ReadFile("prompts/system.txt")
	if err != nil {
		log.Printf("warning: cannot load prompts/system.txt: %v — agent will use minimal prompt", err)
		systemPrompt = "Extract alcohol label fields and return JSON only."
		return
	}
	systemPrompt = string(data)
}

type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  map[string]any  `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ollamaResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

// UseOllama enables the Ollama vision-model path. Off by default because moondream
// requires a GPU to meet the sub-5s latency target; on CPU it takes 60+ seconds.
// Enable with -ollama flag on the server for GPU deployments.
var UseOllama bool

// ExtractFields runs the extraction pipeline. When UseOllama is true it tries
// moondream then llava:7b (preprocessed image) before falling back to Tesseract.
// When UseOllama is false it goes straight to Tesseract (~1.5s on CPU).
// Raw bytes are used for Tesseract because GoCV binarization degrades small text.
func ExtractFields(ctx context.Context, preprocessed, raw []byte) (*models.LabelFields, error) {
	if UseOllama {
		b64 := base64.StdEncoding.EncodeToString(preprocessed)
		available := availableOllamaModels()
		for _, model := range []string{"moondream", "llava:7b"} {
			if !available[model] {
				log.Printf("agent: %s not available locally — skipping", model)
				continue
			}
			fields, err := tryOllama(ctx, model, b64)
			if err == nil {
				return fields, nil
			}
			log.Printf("agent: %s failed: %v", model, err)
		}
		log.Println("agent: all Ollama models failed — falling back to Tesseract")
	}
	return tesseractExtract(raw)
}

// availableOllamaModels returns a set of model name prefixes available locally.
func availableOllamaModels() map[string]bool {
	type tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get("http://localhost:11434/api/tags")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var tags tagsResp
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return nil
	}
	set := make(map[string]bool)
	for _, m := range tags.Models {
		// Match by prefix so "moondream:latest" matches "moondream"
		name := strings.SplitN(m.Name, ":", 2)[0]
		set[name] = true
		set[m.Name] = true
	}
	return set
}

func tryOllama(ctx context.Context, model, b64Image string) (*models.LabelFields, error) {
	req := ollamaRequest{
		Model: model,
		Messages: []ollamaMessage{
			{Role: "system", Content: systemPrompt},
			{
				Role:    "user",
				Content: "Extract all label fields and return valid JSON only.",
				Images:  []string{b64Image},
			},
		},
		Stream:  false,
		Options: map[string]any{"temperature": 0},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 8s hard limit — GPU inference is fast, CPU inference falls back to Tesseract.
	client := &http.Client{Timeout: 8 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned %d", resp.StatusCode)
	}

	var chatResp ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	raw := strings.TrimSpace(chatResp.Message.Content)
	if m := jsonFence.FindStringSubmatch(raw); m != nil {
		raw = strings.TrimSpace(m[1])
	}

	var fields models.LabelFields
	if err := json.Unmarshal([]byte(raw), &fields); err != nil {
		return nil, fmt.Errorf("invalid JSON from model: %w", err)
	}

	return &fields, nil
}

func tesseractExtract(imgBytes []byte) (*models.LabelFields, error) {
	// Resize to max 1500px on the longest side before OCR.
	// Full phone photos (12 MP) take 3–5 s on any PSM mode; 1500 px takes ~0.4 s
	// and Tesseract accuracy does not improve beyond ~300 DPI for label text.
	resized, err := resizeForOCR(imgBytes, 1500)
	if err != nil {
		log.Printf("agent: resize failed (%v), using original", err)
		resized = imgBytes
	}

	tmp, err := os.CreateTemp("", "ttb-*.jpg")
	if err != nil {
		return nil, fmt.Errorf("tmp file: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(resized); err != nil {
		return nil, fmt.Errorf("write tmp: %w", err)
	}
	tmp.Close()

	// PSM 11 (sparse text) finds text anywhere in the image regardless of
	// layout — best for real bottle photos where brand, class type, and ABV
	// appear in different font sizes and are not column-aligned.
	// Fall back to PSM 6 (uniform block) only if PSM 11 yields very little.
	bestText := ""
	bestPSM := ""
	for _, psm := range []string{"11", "6"} {
		out, err := exec.Command("tesseract", tmp.Name(), "stdout", "--psm", psm).Output()
		if err != nil {
			continue
		}
		text := string(out)
		if len(strings.Fields(text)) > len(strings.Fields(bestText)) {
			bestText = text
			bestPSM = psm
		}
		if len(strings.Fields(bestText)) >= 10 {
			break // enough text found; skip remaining modes
		}
	}

	if bestText == "" {
		return nil, fmt.Errorf("tesseract: no text found")
	}

	wordCount := len(strings.Fields(bestText))
	log.Printf("agent: Tesseract PSM=%s yielded %d words", bestPSM, wordCount)

	// CRF sequence tagger — better field extraction than regex on real labels.
	// Falls back to regex heuristics if the model or script is unavailable.
	if fields, err := crfExtract(bestText); err == nil {
		// Confidence reflects actual OCR quality: more words = more confident.
		fields.Confidence = ocrConfidence(wordCount, fields)
		return fields, nil
	} else {
		log.Printf("agent: CRF unavailable (%v) — using regex fallback", err)
	}

	fields := extractFromText(bestText)
	fields.Confidence = ocrConfidence(wordCount, fields)
	return fields, nil
}

// ocrConfidence returns a 0–1 confidence score based on how much text was
// extracted and how many mandatory fields were populated.
func ocrConfidence(wordCount int, f *models.LabelFields) float64 {
	textScore := math.Min(1.0, float64(wordCount)/40.0) // 40 words = full confidence
	fieldsFound := 0
	if f.BrandName != "" { fieldsFound++ }
	if f.ClassType != "" { fieldsFound++ }
	if f.ABVPercent > 0  { fieldsFound++ }
	if f.NetContents != "" { fieldsFound++ }
	fieldScore := float64(fieldsFound) / 4.0
	return math.Round((textScore*0.4+fieldScore*0.6)*100) / 100
}

// resizeForOCR downsizes imgBytes so the longest edge is at most maxPx.
// Returns the original bytes unchanged if the image is already smaller.
func resizeForOCR(imgBytes []byte, maxPx int) ([]byte, error) {
	src, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, err
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= maxPx && h <= maxPx {
		return imgBytes, nil
	}
	scale := float64(maxPx) / float64(max(w, h))
	nw := int(float64(w) * scale)
	nh := int(float64(h) * scale)
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	// Nearest-neighbour resize — fast, zero dependencies, fine for OCR input.
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			sx := b.Min.X + int(float64(x)/scale)
			sy := b.Min.Y + int(float64(y)/scale)
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 90}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// crfExtract pipes raw OCR text through the Python CRF tagger and returns
// extracted label fields as a LabelFields struct.
func crfExtract(ocrText string) (*models.LabelFields, error) {
	cmd := exec.Command("python3", "/app/scripts/crf_tagger.py", "--extract")
	cmd.Stdin = strings.NewReader(ocrText)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("crf exec: %w", err)
	}

	var raw struct {
		BrandName         string `json:"brand_name"`
		ClassType         string `json:"class_type"`
		ABVPercent        string `json:"abv_percent"`
		NetContents       string `json:"net_contents"`
		GovernmentWarning string `json:"government_warning"`
		Error             string `json:"error"`
	}
	if err := json.Unmarshal(out, &raw); err != nil {
		return nil, fmt.Errorf("crf json: %w", err)
	}
	if raw.Error != "" {
		return nil, fmt.Errorf("crf: %s", raw.Error)
	}
	if raw.BrandName == "" && raw.ClassType == "" {
		return nil, fmt.Errorf("crf: no fields extracted")
	}

	// The CRF returns the full ABV line ("43.0% Alc./Vol.") — extract the number.
	abv := 0.0
	for _, re := range reABV {
		if m := re.FindStringSubmatch(raw.ABVPercent); m != nil {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				abv = v
				break
			}
		}
	}

	return &models.LabelFields{
		BrandName:         raw.BrandName,
		ClassType:         raw.ClassType,
		ABVPercent:        abv,
		NetContents:       raw.NetContents,
		GovernmentWarning: raw.GovernmentWarning,
		Confidence:        0.75,
	}, nil
}
