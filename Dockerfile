# ── Stage 1: build the Go binary ────────────────────────────────────────────
FROM gocv/opencv:4.8.0 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    golang-go \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o /ttb-label-verifier ./cmd/server/main.go

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM gocv/opencv:4.8.0

# Tesseract OCR engine + English language data
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python ML dependencies for CRF/Markov scripts called at startup
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY --from=builder /ttb-label-verifier /app/ttb-label-verifier

# Copy prompts, testdata models, and web assets
COPY prompts/       /app/prompts/
COPY web/           /app/web/
COPY testdata/brand_markov.json /app/testdata/brand_markov.json
COPY testdata/crf_model.pkl     /app/testdata/crf_model.pkl

EXPOSE 8181
CMD ["/app/ttb-label-verifier", "-port", "8181"]
