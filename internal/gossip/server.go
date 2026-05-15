package gossip

import (
	"fmt"
	"log/slog"
	"net"
	"time"
)

// StartUDP binds a UDP listener and starts the background goroutines for receiving
// and disseminating gossip packets.
func (g *Protocol) StartUDP(secretKey string) error {
	g.mu.Lock()
	if g.running {
		g.mu.Unlock()
		return fmt.Errorf("gossip protocol is already running")
	}
	g.running = true
	g.mu.Unlock()

	addr := fmt.Sprintf(":%d", g.config.Port)
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return err
	}

	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return err
	}

	slog.Info("gossip udp server started", "port", g.config.Port)

	// Start receiver
	go g.receiveLoop(conn, secretKey)

	// Start disseminator
	go g.disseminationLoop(conn, secretKey)

	return nil
}

// StopUDP closes the UDP listener and background goroutines.
// Safe to call multiple times — uses sync.Once internally.
func (g *Protocol) StopUDP() {
	g.stopOnce.Do(func() {
		g.mu.Lock()
		g.running = false
		g.mu.Unlock()
		close(g.stopCh)
	})
}

func (g *Protocol) receiveLoop(conn *net.UDPConn, secretKey string) {
	defer conn.Close()

	buf := make([]byte, 65535) // Max UDP packet size
	for {
		// Check for shutdown
		select {
		case <-g.stopCh:
			return
		default:
		}

		conn.SetReadDeadline(time.Now().Add(1 * time.Second))
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			slog.Error("gossip read error", "error", err)
			continue
		}

		msg, err := VerifyAndUnmarshal(buf[:n], secretKey)
		if err != nil {
			slog.Warn("gossip decode/verify failed", "error", err)
			continue
		}

		isNew := g.HandleMessage(*msg)
		if isNew {
			// If it's a new event, we buffer it for dissemination so the rest
			// of the cluster learns about it.
			g.bufferForDissemination(*msg)
		}
	}
}
