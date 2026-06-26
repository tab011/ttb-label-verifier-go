package agent

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
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

// ExtractFields runs the three-tier extraction pipeline:
//
//	Tier 1 — Tesseract OCR + CRF sequence tagger (~0.5s, free)
//	Tier 2 — if confidence < ClaudeFallbackThreshold, escalate to Claude Haiku
//	          vision (~3s, ~$0.001/image); skipped if ANTHROPIC_API_KEY unset.
//
// Enterprise note: set ANTHROPIC_BASE_URL to route Tier 2 calls through an
// internal proxy instead of api.anthropic.com (firewall blocks public endpoint;
// VPN-connected officers reach the internal web app which calls the proxy).
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

	fields, err := tesseractExtract(raw)
	if err != nil {
		// Tesseract failed entirely — try Claude immediately if available.
		if os.Getenv("ANTHROPIC_API_KEY") != "" {
			log.Printf("agent: Tesseract failed (%v) — escalating to Claude Haiku", err)
			return claudeExtract(ctx, raw)
		}
		return nil, err
	}

	// Tier 2: escalate to Claude Haiku if OCR confidence is low.
	if fields.Confidence < ClaudeFallbackThreshold && os.Getenv("ANTHROPIC_API_KEY") != "" {
		log.Printf("agent: Tesseract confidence %.2f < %.2f — escalating to Claude Haiku",
			fields.Confidence, ClaudeFallbackThreshold)
		if claudeFields, err := claudeExtract(ctx, raw); err == nil {
			return claudeFields, nil
		} else {
			log.Printf("agent: Claude Haiku fallback failed (%v) — using Tesseract result", err)
		}
	}

	return fields, nil
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

// labelKeywords are terms that appear on genuine alcohol labels.
// Scoring by keyword count is more reliable than raw word count because
// PSM 11 on a busy bottle photo picks up noise/reflections as "words."
var labelKeywords = []string{
	"WHISKEY", "WHISKY", "BOURBON", "DISTILLERY", "DISTILLED", "BOTTLED",
	"STRAIGHT", "KENTUCKY", "TENNESSEE", "PROOF", "ALCOHOL", "GOVERNMENT",
	"WARNING", "ALC", "VOL", "750", "1.75", "LITER", "LITRE",
}

// labelScore counts how many domain keywords appear in the OCR output.
func labelScore(text string) int {
	upper := strings.ToUpper(text)
	n := 0
	for _, kw := range labelKeywords {
		if strings.Contains(upper, kw) {
			n++
		}
	}
	return n
}

// runTesseract calls tesseract on a temp file path with --oem 1 (LSTM only)
// and the given PSM mode. Returns empty string on error.
func runTesseract(path, psm string) string {
	out, err := exec.Command("tesseract", path, "stdout", "--oem", "1", "--psm", psm).Output()
	if err != nil {
		return ""
	}
	return string(out)
}

func tesseractExtract(imgBytes []byte) (*models.LabelFields, error) {
	// Decode and resize to max 1500px. Returns the decoded image for variant generation.
	resized, src, err := resizeForOCR(imgBytes, 1500)
	if err != nil {
		log.Printf("agent: resize failed (%v), using original", err)
		resized = imgBytes
		src = nil
	}

	// Generate image variants: original, grayscale, inverted-grayscale.
	// Many bourbon bottles have dark labels with light text; inverting the
	// grayscale turns white-on-dark into black-on-white which Tesseract reads
	// far better than the original dark background.
	type variant struct {
		data []byte
		label string
	}
	variants := []variant{{nil, "orig"}, {nil, "gray"}, {nil, "inv"}}
	variants[0].data = resized
	if src != nil {
		variants[1].data = toGray(src)
		variants[2].data = invertGray(src)
	}

	// Try PSM 3 (auto layout) then PSM 11 (sparse) on each variant;
	// keep the result with the highest domain-keyword score.
	bestText, bestScore, bestDesc := "", -1, ""
	for _, v := range variants {
		if len(v.data) == 0 {
			continue
		}
		tmp, err := os.CreateTemp("", "ttb-*.jpg")
		if err != nil {
			continue
		}
		tmpName := tmp.Name()
		tmp.Write(v.data)
		tmp.Close()

		for _, psm := range []string{"3", "11"} {
			text := runTesseract(tmpName, psm)
			if score := labelScore(text); score > bestScore {
				bestText, bestScore, bestDesc = text, score, v.label+"/psm"+psm
			}
		}
		os.Remove(tmpName)
	}

	if bestText == "" {
		return nil, fmt.Errorf("tesseract: no text found in any variant")
	}

	wordCount := len(strings.Fields(bestText))
	log.Printf("agent: best OCR variant=%s keywords=%d words=%d", bestDesc, bestScore, wordCount)

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
// Returns (jpegBytes, decodedImage, error). The decoded image is returned so
// callers can generate grayscale/inverted variants without decoding twice.
func resizeForOCR(imgBytes []byte, maxPx int) ([]byte, image.Image, error) {
	src, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, nil, err
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	var dst image.Image
	if w <= maxPx && h <= maxPx {
		dst = src
	} else {
		scale := float64(maxPx) / float64(max(w, h))
		nw := int(float64(w) * scale)
		nh := int(float64(h) * scale)
		d := image.NewRGBA(image.Rect(0, 0, nw, nh))
		for y := 0; y < nh; y++ {
			for x := 0; x < nw; x++ {
				sx := b.Min.X + int(float64(x)/scale)
				sy := b.Min.Y + int(float64(y)/scale)
				d.Set(x, y, src.At(sx, sy))
			}
		}
		dst = d
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 90}); err != nil {
		return nil, nil, err
	}
	return buf.Bytes(), dst, nil
}

// toGray returns a JPEG-encoded grayscale version of the image.
func toGray(src image.Image) []byte {
	b := src.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			g.Set(x, y, src.At(x, y))
		}
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, g, &jpeg.Options{Quality: 90})
	return buf.Bytes()
}

// invertGray returns a JPEG-encoded inverted-grayscale image.
// Bourbon labels often have white/gold text on dark backgrounds; inverting
// converts that to black-on-white which Tesseract handles much better.
func invertGray(src image.Image) []byte {
	b := src.Bounds()
	g := image.NewGray(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, _, _, _ := src.At(x, y).RGBA()
			lum := uint8(r >> 8)
			g.SetGray(x, y, color.Gray{Y: 255 - lum})
		}
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, g, &jpeg.Options{Quality: 90})
	return buf.Bytes()
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
