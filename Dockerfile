# Build stage
FROM golang:1.25-alpine AS builder

RUN apk --no-cache add gcc musl-dev

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -o magnitude ./cmd/server

# Run stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /app/magnitude .
COPY config.yaml .

EXPOSE 8443

CMD ["./magnitude"]
