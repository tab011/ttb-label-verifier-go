package models

type LabelFields struct {
	BrandName         string  `json:"brand_name"`
	ClassType         string  `json:"class_type"`
	ABVPercent        float64 `json:"abv_percent"`
	NetContents       string  `json:"net_contents"`
	GovernmentWarning string  `json:"government_warning"`
	Confidence        float64 `json:"confidence"`
}

type FieldVerdict struct {
	Status    string   `json:"status"` // PASS, FAIL, UNKNOWN
	Extracted string   `json:"extracted"`
	Expected  string   `json:"expected"`
	Score     *float64 `json:"score,omitempty"`
}

type ComplianceResult struct {
	Verdict          string                  `json:"verdict"` // PASS or FAIL
	Fields           map[string]FieldVerdict `json:"fields"`
	Notes            string                  `json:"notes"`
	Confidence       float64                 `json:"confidence"`
	SpiritCategory   string                  `json:"spirit_category,omitempty"`
	SpiritConfidence float64                 `json:"spirit_confidence,omitempty"`
}
