# ── Stage 1: build the Go binary ────────────────────────────────────────────
FROM gocv/opencv:4.8.0 AS builder

# gocv/opencv:4.8.0 already has a Go installation at /usr/local/go (circa 1.21).
# Overwriting with tar -C /usr/local merges rather than replaces, leaving old
# runtime files (msize.go) alongside new ones (msize_allocheaders.go) → redecl.
# Remove first, then install Go 1.22.4 cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/go \
    && wget -q https://go.dev/dl/go1.22.4.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz \
    && rm go1.22.4.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GOTOOLCHAIN=local

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o /ttb-label-verifier ./cmd/server/main.go

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM gocv/opencv:4.8.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY --from=builder /ttb-label-verifier /app/ttb-label-verifier
COPY prompts/       /app/prompts/
COPY web/           /app/web/
COPY testdata/brand_markov.json /app/testdata/brand_markov.json
COPY testdata/crf_model.pkl     /app/testdata/crf_model.pkl

EXPOSE 8181
CMD ["/app/ttb-label-verifier", "-addr", ":8181"]
