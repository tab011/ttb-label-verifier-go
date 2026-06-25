BINARY   := ttb-label-verifier
PORT     := 8181
TESTDATA := testdata/images

.PHONY: all build run test train-crf regen-testdata clean

all: build

## Build the Go server binary
build:
	go build -o $(BINARY) ./cmd/server/main.go

## Run the server (builds first)
run: build
	./$(BINARY) -port $(PORT)

## Run with Ollama vision model enabled (requires GPU for sub-5s latency)
run-ollama: build
	./$(BINARY) -port $(PORT) -ollama

## Install Python dependencies
pip:
	pip3 install -r requirements.txt

## Regenerate synthetic test labels (30 images + labels.csv)
regen-testdata:
	python3 scripts/build_testdata.py

## Train the CRF tagger from testdata labels (run after regen-testdata)
train-crf:
	python3 scripts/crf_tagger.py --train

## Build the brand Markov chain matrix (run after regen-testdata)
train-markov:
	python3 scripts/brand_markov.py --mode matrix

## Full offline setup: regenerate data, train both models
setup: pip regen-testdata train-markov train-crf
	@echo ""
	@echo "Setup complete. Run 'make run' to start the server on :$(PORT)."

## Run the Go test suite
test:
	go test ./...

## Run the CRF vs HMM benchmark on testdata
benchmark:
	python3 scripts/crf_tagger.py --benchmark

## Run the full Python pipeline integration test
test-pipeline:
	python3 scripts/test_pipeline.py

clean:
	rm -f $(BINARY)
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
