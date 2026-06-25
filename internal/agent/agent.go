package agent

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"regexp"
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
	// Use the tesseract CLI rather than Gosseract's C API because Gosseract's
	// SetImageFromBytes uses a fixed 70-DPI assumption and clips the top of
	// the image (missing brand names). The CLI reads JPEG metadata directly.
	tmp, err := os.CreateTemp("", "ttb-*.jpg")
	if err != nil {
		return nil, fmt.Errorf("tmp file: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(imgBytes); err != nil {
		return nil, fmt.Errorf("write tmp: %w", err)
	}
	tmp.Close()

	out, err := exec.Command("tesseract", tmp.Name(), "stdout", "--psm", "4").Output()
	if err != nil {
		// Try without PSM override on older Tesseract versions
		out, err = exec.Command("tesseract", tmp.Name(), "stdout").Output()
		if err != nil {
			return nil, fmt.Errorf("tesseract exec: %w", err)
		}
	}

	return extractFromText(string(out)), nil
}
