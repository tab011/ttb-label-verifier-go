// Package store holds an in-memory lookup table of COLA records loaded from CSV.
// The server loads the CSV once at startup; the /import endpoint can hot-reload it.
package store

import (
	"encoding/csv"
	"fmt"
	"io"
	"strings"
	"sync"
)

// COLARecord mirrors the columns in labels.csv / the TTB bulk export.
type COLARecord struct {
	BrandName   string
	ClassType   string
	ABVPercent  string
	NetContents string
}

// Store is a thread-safe in-memory COLA lookup keyed by normalized brand name.
type Store struct {
	mu      sync.RWMutex
	records map[string]COLARecord // key = strings.ToUpper(brand_name)
}

func New() *Store {
	return &Store{records: make(map[string]COLARecord)}
}

// LoadCSV replaces the store contents from a CSV reader.
// Accepted column names (case-insensitive): brand_name, class_type, abv_percent, net_contents.
// Rows with an empty brand_name are skipped.
func (s *Store) LoadCSV(r io.Reader) (int, error) {
	cr := csv.NewReader(r)
	header, err := cr.Read()
	if err != nil {
		return 0, fmt.Errorf("read header: %w", err)
	}
	idx := make(map[string]int)
	for i, h := range header {
		idx[strings.ToLower(strings.TrimSpace(h))] = i
	}
	col := func(row []string, name string) string {
		i, ok := idx[name]
		if !ok || i >= len(row) {
			return ""
		}
		return strings.TrimSpace(row[i])
	}

	fresh := make(map[string]COLARecord)
	for {
		row, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue
		}
		brand := col(row, "brand_name")
		if brand == "" {
			continue
		}
		key := strings.ToUpper(brand)
		fresh[key] = COLARecord{
			BrandName:   brand,
			ClassType:   col(row, "class_type"),
			ABVPercent:  col(row, "abv_percent"),
			NetContents: col(row, "net_contents"),
		}
	}

	s.mu.Lock()
	s.records = fresh
	s.mu.Unlock()
	return len(fresh), nil
}

// Lookup returns the best-matching COLA record for the given brand name.
// Exact match (case-insensitive) first; falls back to prefix match.
func (s *Store) Lookup(brandName string) (COLARecord, bool) {
	key := strings.ToUpper(strings.TrimSpace(brandName))
	s.mu.RLock()
	defer s.mu.RUnlock()

	if r, ok := s.records[key]; ok {
		return r, true
	}
	// Prefix fallback: first record whose key starts with the query (or vice-versa)
	for k, r := range s.records {
		if strings.HasPrefix(k, key) || strings.HasPrefix(key, k) {
			return r, true
		}
	}
	return COLARecord{}, false
}

// Len returns the number of loaded records.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.records)
}
