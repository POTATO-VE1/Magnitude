# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM golang:1.25-alpine AS builder

WORKDIR /build

ENV CGO_CFLAGS_ALLOW="-mavx2|-mfma|-O3"

# CGO is needed for SIMD distance kernels (AVX2/NEON); install the C toolchain.
RUN apk add --no-cache gcc musl-dev

# Cache dependencies first (layer changes rarely)
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
# Note: CGO is needed for SIMD distance kernels on amd64/arm64
COPY . .
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /build/magnitude ./cmd/server/main.go

# ── Stage 2: Production ──────────────────────────────────────────────────────
FROM alpine:3.20

LABEL org.opencontainers.image.title="Magnitude"
LABEL org.opencontainers.image.description="Fast, self-hosted vector database"
LABEL org.opencontainers.image.url="https://github.com/POTATO-VE1/Magnitude"
LABEL org.opencontainers.image.source="https://github.com/POTATO-VE1/Magnitude"
LABEL org.opencontainers.image.licenses="MIT"

RUN apk add --no-cache ca-certificates curl

# Create non-root user
RUN addgroup -S magnitude && adduser -S magnitude -G magnitude

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/magnitude /app/magnitude

# Create directories for runtime data
RUN mkdir -p /app/data /app/certs && \
    chown -R magnitude:magnitude /app

# Copy default config (Docker variant has authentication enabled)
COPY config.docker.yaml /app/config.yaml

USER magnitude

# REST API + gRPC + metrics
EXPOSE 8080 9090 9091

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/v1/health || exit 1

ENTRYPOINT ["/app/magnitude"]
