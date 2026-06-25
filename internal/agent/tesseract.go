package agent

import (
	"regexp"
	"strconv"
	"strings"

	"ttb-label-verifier/internal/models"
)

var (
	reABV = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(\d+\.?\d*)\s*%`),
		regexp.MustCompile(`(?i)(\d+\.?\d*)\s*percent`),
		regexp.MustCompile(`(?i)(\d+\.?\d*)\s*[oO0]/[oO0]`),
	}
	// Matches 750ml, 1750ml, 1L, 375mL, 1.75 L, 50 fl oz, 1 liter
	reNet = regexp.MustCompile(`(?i)(\d+\.?\d*\s*(?:mL|L\b|fl\.?\s*oz|liters?))`)
)

func extractFromText(text string) *models.LabelFields {
	upper := strings.ToUpper(text)

	abv := 0.0
	for _, re := range reABV {
		if m := re.FindStringSubmatch(text); m != nil {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				abv = v
				break
			}
		}
	}

	net := ""
	if m := reNet.FindStringSubmatch(text); m != nil {
		net = strings.TrimSpace(m[1])
	}

	warning := ""
	if i := strings.Index(upper, "GOVERNMENT WARNING"); i >= 0 {
		warning = strings.TrimSpace(text[i:])
	}

	// Layout convention: brand is the first all-caps line, class type is the second.
	// The old heuristic picked the LONGEST all-caps line, which matched the class type
	// text on bourbon labels (e.g. "STRAIGHT BOURBON WHISKY" > most brand names).
	lines := strings.Split(text, "\n")
	brand := ""
	classType := ""
	for _, line := range lines {
		stripped := strings.TrimSpace(line)
		if stripped == "" {
			continue
		}
		if stripped == strings.ToUpper(stripped) {
			if brand == "" {
				brand = stripped
			} else if classType == "" {
				classType = stripped
				break
			}
		}
	}

	return &models.LabelFields{
		BrandName:         brand,
		ClassType:         classType,
		ABVPercent:        abv,
		NetContents:       net,
		GovernmentWarning: warning,
		Confidence:        0.4, // flagged low so UI can warn user
	}
}
