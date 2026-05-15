# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM golang:1.25-alpine AS builder

WORKDIR /build

# Cache dependencies first (layer changes rarely)
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build (pure Go — no cgo needed, modernc.org/sqlite is native)
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /build/magnitude ./cmd/server/main.go

# ── Stage 2: Production ──────────────────────────────────────────────────────
FROM alpine:3.20

RUN apk add --no-cache ca-certificates curl

# Create non-root user
RUN addgroup -S magnitude && adduser -S magnitude -G magnitude

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/magnitude /app/magnitude

# Create directories for runtime data (will be overridden by volume mounts)
RUN mkdir -p /app/data /app/certs && \
    chown -R magnitude:magnitude /app

# Copy default config
COPY config.yaml /app/config.yaml

USER magnitude

EXPOSE 8443 9090

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9090/health || exit 1

ENTRYPOINT ["/app/magnitude"]
CMD ["--config", "/app/config.yaml"]
