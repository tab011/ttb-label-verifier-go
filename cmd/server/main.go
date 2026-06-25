package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"ttb-label-verifier/internal/agent"
	"ttb-label-verifier/internal/compare"
	"ttb-label-verifier/internal/preprocess"
	"ttb-label-verifier/internal/store"
)

var (
	tmpl  *template.Template
	colaDB = store.New()
)

func main() {
	addr       := flag.String("addr", ":8080", "listen address")
	csvPath    := flag.String("cola-csv", "testdata/labels.csv", "COLA lookup CSV to pre-load")
	useOllama  := flag.Bool("ollama", false, "enable Ollama vision model (requires GPU for sub-5s latency)")
	flag.Parse()
	agent.UseOllama = *useOllama

	var err error
	tmpl, err = template.ParseGlob("web/templates/*.html")
	if err != nil {
		log.Fatal("parse templates:", err)
	}

	// Pre-load COLA lookup store
	if *csvPath != "" {
		if f, err := os.Open(*csvPath); err == nil {
			n, err := colaDB.LoadCSV(f)
			f.Close()
			if err != nil {
				log.Printf("COLA CSV load warning: %v", err)
			} else {
				log.Printf("COLA store: %d records loaded from %s", n, *csvPath)
			}
		} else {
			log.Printf("COLA CSV not found at %s (continuing without pre-load)", *csvPath)
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", handleIndex)
	mux.HandleFunc("POST /verify", handleVerify)
	mux.HandleFunc("POST /batch", handleBatch)
	mux.HandleFunc("POST /import", handleImport)
	mux.HandleFunc("GET /api/lookup", handleLookup)
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir("web/static"))))

	log.Printf("TTB Label Verifier running on http://localhost%s  (COLA store: %d records)", *addr, colaDB.Len())
	log.Fatal(http.ListenAndServe(*addr, mux))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if err := tmpl.ExecuteTemplate(w, "index.html", nil); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

// handleLookup lets the JS autocomplete fill form fields from the COLA store.
// GET /api/lookup?brand=MAKER%27S+MARK
func handleLookup(w http.ResponseWriter, r *http.Request) {
	brand := r.URL.Query().Get("brand")
	rec, ok := colaDB.Lookup(brand)
	if !ok {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rec)
}

// handleImport replaces the COLA store from an uploaded CSV.
// POST /import  multipart field: "csv"
func handleImport(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		jsonError(w, "invalid form", http.StatusBadRequest)
		return
	}
	f, _, err := r.FormFile("csv")
	if err != nil {
		jsonError(w, "no csv file", http.StatusBadRequest)
		return
	}
	defer f.Close()
	data, _ := io.ReadAll(f)
	n, err := colaDB.LoadCSV(bytes.NewReader(data))
	if err != nil {
		jsonError(w, fmt.Sprintf("CSV parse error: %v", err), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"imported": n})
}

func handleVerify(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		jsonError(w, "invalid form data", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		jsonError(w, "no image uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		jsonError(w, "could not read image", http.StatusInternalServerError)
		return
	}

	expected := map[string]string{
		"brand_name":   r.FormValue("brand_name"),
		"class_type":   r.FormValue("class_type"),
		"abv_percent":  r.FormValue("abv_percent"),
		"net_contents": r.FormValue("net_contents"),
	}

	// Auto-fill from COLA store when form fields are blank
	if rec, ok := colaDB.Lookup(expected["brand_name"]); ok {
		if expected["class_type"] == "" {
			expected["class_type"] = rec.ClassType
		}
		if expected["abv_percent"] == "" {
			expected["abv_percent"] = rec.ABVPercent
		}
		if expected["net_contents"] == "" {
			expected["net_contents"] = rec.NetContents
		}
	}

	preprocessed, err := preprocess.ProcessLabel(imgBytes)
	if err != nil {
		log.Printf("preprocess warning: %v — using raw image", err)
		preprocessed = imgBytes
	}

	extracted, err := agent.ExtractFields(r.Context(), preprocessed, imgBytes)
	if err != nil {
		jsonError(w, fmt.Sprintf("extraction failed: %v", err), http.StatusInternalServerError)
		return
	}

	result := compare.Verify(extracted, expected)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func handleBatch(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(128 << 20); err != nil {
		jsonError(w, "invalid form data", http.StatusBadRequest)
		return
	}

	csvFile, _, err := r.FormFile("csv")
	if err != nil {
		jsonError(w, "no CSV uploaded", http.StatusBadRequest)
		return
	}
	defer csvFile.Close()

	csvBytes, _ := io.ReadAll(csvFile)
	rows := parseCSV(string(csvBytes))

	imageFiles := r.MultipartForm.File["images"]
	imageMap := make(map[string][]byte, len(imageFiles))
	for _, fh := range imageFiles {
		f, err := fh.Open()
		if err != nil {
			continue
		}
		data, _ := io.ReadAll(f)
		f.Close()
		imageMap[fh.Filename] = data
	}

	type batchResult struct {
		Filename string `json:"filename"`
		Verdict  string `json:"verdict"`
		Notes    string `json:"notes"`
	}

	var results []batchResult
	for _, row := range rows {
		fname := row["filename"]
		imgBytes, ok := imageMap[fname]
		if !ok {
			results = append(results, batchResult{Filename: fname, Verdict: "ERROR", Notes: "image not found"})
			continue
		}

		// Auto-fill missing expected values from COLA store
		if rec, found := colaDB.Lookup(row["brand_name"]); found {
			if row["class_type"] == "" {
				row["class_type"] = rec.ClassType
			}
			if row["abv_percent"] == "" {
				row["abv_percent"] = rec.ABVPercent
			}
			if row["net_contents"] == "" {
				row["net_contents"] = rec.NetContents
			}
		}

		preprocessed, err := preprocess.ProcessLabel(imgBytes)
		if err != nil {
			log.Printf("preprocess warning for %s: %v — using raw", fname, err)
			preprocessed = imgBytes
		}

		extracted, err := agent.ExtractFields(r.Context(), preprocessed, imgBytes)
		if err != nil {
			results = append(results, batchResult{Filename: fname, Verdict: "ERROR", Notes: err.Error()})
			continue
		}

		result := compare.Verify(extracted, row)
		results = append(results, batchResult{Filename: fname, Verdict: result.Verdict, Notes: result.Notes})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func parseCSV(s string) []map[string]string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) < 2 {
		return nil
	}
	headers := splitCSVLine(lines[0])
	var rows []map[string]string
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == "" {
			continue
		}
		vals := splitCSVLine(line)
		row := make(map[string]string, len(headers))
		for i, h := range headers {
			if i < len(vals) {
				row[strings.TrimSpace(h)] = strings.TrimSpace(vals[i])
			}
		}
		rows = append(rows, row)
	}
	return rows
}

func splitCSVLine(line string) []string {
	return strings.Split(line, ",")
}
