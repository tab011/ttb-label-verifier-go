package agent

// Claude Haiku vision fallback — Tier 3 extraction.
//
// Triggered when Tesseract + CRF confidence is below ClaudeFallbackThreshold.
// Calls the Anthropic Messages API directly via net/http so there is no Python
// dependency in the runtime image.
//
// Enterprise / TTB in-house deployment — four environment variables control
// all outbound Anthropic traffic. Set these on the server; block
// api.anthropic.com:443 at the firewall. VPN-connected officers reach the
// internal web app, which calls the internal gateway — no direct public
// internet access from field devices required.
//
//   ANTHROPIC_API_KEY          — required; enterprise API key from Anthropic
//
//   ANTHROPIC_BASE_URL         — redirect all /v1/* calls to an internal proxy
//                                e.g. https://anthropic-gw.internal.ttb.gov
//                                The go http client uses this as the base URL
//                                in place of https://api.anthropic.com
//
//   ANTHROPIC_DEFAULT_HEADERS  — comma-separated "Key: Value" pairs injected
//                                into every request; used for org-routing or
//                                gateway authentication headers the proxy needs
//                                e.g. "X-Org-ID: ttb,X-Gateway-Token: ..."
//
//   HTTPS_PROXY                — standard env var; go's http.DefaultTransport
//                                reads this automatically; routes all outbound
//                                HTTPS through the corporate proxy server
//                                e.g. https://proxy.ttb.gov:8080
//
// Cost: ~$0.001 per image (Haiku input pricing). Only invoked on low-confidence
// labels — known labels that OCR well cost nothing.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"ttb-label-verifier/internal/models"
)

// ClaudeFallbackThreshold is the OCR confidence below which we escalate to
// Claude Haiku vision. Adjust upward if API costs are a concern, downward
// if Tesseract is performing well enough.
const ClaudeFallbackThreshold = 0.40

const anthropicVersion = "2023-06-01"

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system"`
	Messages  []anthropicMessage `json:"messages"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type   string           `json:"type"`
	Text   string           `json:"text,omitempty"`
	Source *anthropicSource `json:"source,omitempty"`
}

type anthropicSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// claudeExtract sends the image to Claude Haiku and returns extracted label
// fields. The image is resized to 1000px before sending to reduce token cost.
func claudeExtract(ctx context.Context, imgBytes []byte) (*models.LabelFields, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY not set — Claude fallback unavailable")
	}

	baseURL := strings.TrimRight(os.Getenv("ANTHROPIC_BASE_URL"), "/")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}

	// Resize to 1000px to minimise input tokens (~$0.001 vs ~$0.004 at full res).
	resizedBytes, _, resizeErr := resizeForOCR(imgBytes, 1000)
	if resizeErr != nil {
		resizedBytes = imgBytes
	}

	// Re-encode as JPEG to ensure correct media type.
	img, jpegErr := jpeg.Decode(bytes.NewReader(resizedBytes))
	if jpegErr == nil {
		var buf bytes.Buffer
		if jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}) == nil {
			resizedBytes = buf.Bytes()
		}
	}

	b64 := base64.StdEncoding.EncodeToString(resizedBytes)

	req := anthropicRequest{
		Model:     "claude-haiku-4-5-20251001",
		MaxTokens: 512,
		System: `You are a TTB alcohol label reader. Extract mandatory label fields and return
valid JSON only — no markdown fences, no explanation. If a field is not visible
on the label, use an empty string or 0 for numbers.`,
		Messages: []anthropicMessage{{
			Role: "user",
			Content: []anthropicContent{
				{
					Type: "image",
					Source: &anthropicSource{
						Type:      "base64",
						MediaType: "image/jpeg",
						Data:      b64,
					},
				},
				{
					Type: "text",
					Text: `Read this alcohol bottle label and return JSON with these exact keys:
{
  "brand_name": "...",
  "class_type": "...",
  "abv_percent": 0.0,
  "net_contents": "...",
  "government_warning": "..."
}
brand_name: the distillery/brand name in uppercase as it appears on the label.
class_type: e.g. STRAIGHT BOURBON WHISKY, KENTUCKY STRAIGHT BOURBON WHISKEY.
abv_percent: numeric ABV, e.g. 40.0.
net_contents: e.g. 750 mL.
government_warning: full GOVERNMENT WARNING text if present, else empty string.`,
				},
			},
		}},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	// ANTHROPIC_DEFAULT_HEADERS: "Key: Value,Key2: Value2"
	// Injected for enterprise proxy auth or org-routing requirements.
	if extra := os.Getenv("ANTHROPIC_DEFAULT_HEADERS"); extra != "" {
		for _, pair := range strings.Split(extra, ",") {
			parts := strings.SplitN(strings.TrimSpace(pair), ":", 2)
			if len(parts) == 2 {
				httpReq.Header.Set(strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]))
			}
		}
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic API unreachable: %w", err)
	}
	defer resp.Body.Close()

	var ar anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if ar.Error != nil {
		return nil, fmt.Errorf("anthropic error: %s", ar.Error.Message)
	}
	if len(ar.Content) == 0 {
		return nil, fmt.Errorf("empty response from Claude")
	}

	raw := strings.TrimSpace(ar.Content[0].Text)
	// Strip markdown fences if the model added them despite instructions.
	if i := strings.Index(raw, "{"); i > 0 {
		raw = raw[i:]
	}
	if i := strings.LastIndex(raw, "}"); i >= 0 {
		raw = raw[:i+1]
	}

	var parsed struct {
		BrandName         string  `json:"brand_name"`
		ClassType         string  `json:"class_type"`
		ABVPercent        float64 `json:"abv_percent"`
		NetContents       string  `json:"net_contents"`
		GovernmentWarning string  `json:"government_warning"`
	}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("parse Claude JSON: %w — raw: %s", err, raw)
	}

	fields := &models.LabelFields{
		BrandName:         strings.TrimSpace(parsed.BrandName),
		ClassType:         strings.TrimSpace(parsed.ClassType),
		ABVPercent:        parsed.ABVPercent,
		NetContents:       strings.TrimSpace(parsed.NetContents),
		GovernmentWarning: strings.TrimSpace(parsed.GovernmentWarning),
		Confidence:        0.85, // Claude Haiku vision is high-confidence on clear labels
	}

	log.Printf("agent: Claude Haiku extracted brand=%q class=%q abv=%.1f%%",
		fields.BrandName, fields.ClassType, fields.ABVPercent)
	return fields, nil
}

