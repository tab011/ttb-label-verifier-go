package compare

import (
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"ttb-label-verifier/internal/models"
)

// Exact text required by 27 CFR § 16.21, normalized to uppercase for comparison.
const governmentWarning = "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK " +
	"ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. " +
	"(2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR " +
	"OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."

var abvPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(\d+\.?\d*)\s*%`),
	regexp.MustCompile(`(?i)(\d+\.?\d*)\s*percent`),
	regexp.MustCompile(`(?i)(\d+\.?\d*)\s*[oO0]/[oO0]`),
}

// Verify checks extracted label fields against the form-submitted expected values.
// expected["class_type"] is used to apply 27 CFR class-type-aware ABV range rules.
func Verify(extracted *models.LabelFields, expected map[string]string) *models.ComplianceResult {
	verdicts := map[string]models.FieldVerdict{
		"brand_name":         fuzzyCheck(extracted.BrandName, expected["brand_name"], 90),
		"class_type":         fuzzyCheck(extracted.ClassType, expected["class_type"], 85),
		"abv_percent":        abvCheckTyped(extracted.ABVPercent, expected["abv_percent"], expected["class_type"]),
		"net_contents":       fuzzyCheck(extracted.NetContents, expected["net_contents"], 85),
		"government_warning": warningCheck(extracted.GovernmentWarning),
	}

	overall := "PASS"
	var failures []string
	for k, v := range verdicts {
		if v.Status != "PASS" {
			overall = "FAIL"
			failures = append(failures, k)
		}
	}

	notes := "All fields verified."
	if len(failures) > 0 {
		notes = "Failed: " + strings.Join(failures, ", ") + "."
	}

	return &models.ComplianceResult{
		Verdict:    overall,
		Fields:     verdicts,
		Notes:      notes,
		Confidence: extracted.Confidence,
	}
}

func normalizeWarning(s string) string {
	return strings.Join(strings.Fields(strings.ToUpper(s)), " ")
}

// warningCheck verifies the government warning per 27 CFR § 16.21.
// Two conditions must both hold:
//  1. Content match: the words match the required text (case-insensitive, whitespace-normalized)
//  2. Case compliance: the extracted text is substantially uppercase (≥70% of alpha chars)
//     because 27 CFR § 16.21 requires ALL CAPS presentation on the label.
func warningCheck(extracted string) models.FieldVerdict {
	// Content check: fuzzy match (≥88%) tolerates minor OCR noise in the long warning text
	contentScore := similarity(normalizeWarning(extracted), governmentWarning)
	if contentScore < 88 {
		return models.FieldVerdict{Status: "FAIL", Extracted: extracted, Expected: governmentWarning}
	}

	// Count uppercase ratio to detect wrong-case violations
	var letters, upper int
	for _, c := range extracted {
		if unicode.IsLetter(c) {
			letters++
			if unicode.IsUpper(c) {
				upper++
			}
		}
	}
	if letters > 20 && float64(upper)/float64(letters) < 0.70 {
		return models.FieldVerdict{
			Status:   "FAIL",
			Extracted: extracted,
			Expected:  governmentWarning + " [ALL CAPS required per 27 CFR § 16.21]",
		}
	}

	return models.FieldVerdict{Status: "PASS", Extracted: extracted, Expected: governmentWarning}
}

func fuzzyCheck(extracted, expected string, threshold float64) models.FieldVerdict {
	if expected == "" {
		return models.FieldVerdict{Status: "UNKNOWN", Extracted: extracted, Expected: expected}
	}
	score := similarity(strings.ToUpper(extracted), strings.ToUpper(expected))
	status := "FAIL"
	if score >= threshold {
		status = "PASS"
	}
	return models.FieldVerdict{Status: status, Extracted: extracted, Expected: expected, Score: &score}
}

// abvCheckTyped validates ABV with 27 CFR class-type-aware rules:
//   - Bottled-in-Bond (BIB): must be exactly 50.0% per 27 CFR § 5.121
//   - All distilled spirits: must be 40–80% per 27 CFR § 5.36
//   - If expected form value provided, also checks against it (±0.1 tolerance)
func abvCheckTyped(extractedVal float64, expectedStr, classType string) models.FieldVerdict {
	extractedStr := strconv.FormatFloat(extractedVal, 'f', 1, 64)
	ct := strings.ToUpper(classType)

	// Regulatory range check first
	if strings.Contains(ct, "BOTTLED IN BOND") || strings.Contains(ct, "BIB") {
		if math.Abs(extractedVal-50.0) > 0.05 {
			return models.FieldVerdict{
				Status: "FAIL", Extracted: extractedStr,
				Expected: "50.0 (BIB requires exactly 50% per 27 CFR § 5.121)",
			}
		}
	} else if extractedVal > 0 && (extractedVal < 40.0 || extractedVal > 80.0) {
		return models.FieldVerdict{
			Status: "FAIL", Extracted: extractedStr,
			Expected: "40.0–80.0 (spirits must be 40–80% ABV per 27 CFR § 5.36)",
		}
	}

	// Check against form-submitted expected value if provided
	expectedVal := parseABV(expectedStr)
	if expectedVal == nil {
		return models.FieldVerdict{Status: "UNKNOWN", Extracted: extractedStr, Expected: expectedStr}
	}
	diff := math.Abs(extractedVal - *expectedVal)
	status := "PASS"
	if diff > 0.1 {
		status = "FAIL"
	}
	return models.FieldVerdict{Status: status, Extracted: extractedStr, Expected: expectedStr, Score: &diff}
}

func parseABV(s string) *float64 {
	for _, re := range abvPatterns {
		if m := re.FindStringSubmatch(s); m != nil {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				return &v
			}
		}
	}
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "%"))
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return &v
	}
	return nil
}

// similarity returns 0–100 based on normalized Levenshtein distance.
func similarity(a, b string) float64 {
	if a == b {
		return 100
	}
	maxLen := max(len([]rune(a)), len([]rune(b)))
	if maxLen == 0 {
		return 100
	}
	dist := levenshtein([]rune(a), []rune(b))
	return (1.0 - float64(dist)/float64(maxLen)) * 100
}

func levenshtein(a, b []rune) int {
	la, lb := len(a), len(b)
	dp := make([][]int, la+1)
	for i := range dp {
		dp[i] = make([]int, lb+1)
		dp[i][0] = i
	}
	for j := 0; j <= lb; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= la; i++ {
		for j := 1; j <= lb; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = 1 + min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1]))
			}
		}
	}
	return dp[la][lb]
}
