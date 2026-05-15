package gossip

import (
	"log/slog"
	"math/rand"
	"net"
	"time"
)

// bufferForDissemination adds a new message to the outgoing buffer.
func (g *Protocol) bufferForDissemination(msg Message) {
	g.bufferMu.Lock()
	defer g.bufferMu.Unlock()

	// Keep buffer bounded (e.g. last 100 messages)
	if len(g.buffer) >= 100 {
		g.buffer = g.buffer[1:]
	}
	g.buffer = append(g.buffer, msg)
}

// disseminationLoop periodically transmits buffered messages to random peers.
func (g *Protocol) disseminationLoop(conn *net.UDPConn, secretKey string) {
	ticker := time.NewTicker(g.config.ProbeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-g.stopCh:
			return
		case <-ticker.C:
			g.disseminate(conn, secretKey)
		}
	}
}

func (g *Protocol) disseminate(conn *net.UDPConn, secretKey string) {
	g.bufferMu.Lock()
	if len(g.buffer) == 0 {
		g.bufferMu.Unlock()
		return
	}
	// Copy buffer and drain it to avoid resending stale messages every tick.
	msgs := make([]Message, len(g.buffer))
	copy(msgs, g.buffer)
	g.buffer = g.buffer[:0] // drain — messages are now the disseminator's responsibility
	g.bufferMu.Unlock()

	g.mu.RLock()
	peers := make([]string, len(g.peers))
	copy(peers, g.peers)
	fanout := g.config.Fanout
	g.mu.RUnlock()

	if len(peers) == 0 {
		return
	}

	// Shuffle peers for random selection
	rand.Shuffle(len(peers), func(i, j int) {
		peers[i], peers[j] = peers[j], peers[i]
	})

	// Select top N peers based on Fanout
	if fanout > len(peers) {
		fanout = len(peers)
	}
	selectedPeers := peers[:fanout]

	for _, msg := range msgs {
		data, err := SignMessage(&msg, secretKey)
		if err != nil {
			slog.Warn("gossip: failed to sign message", "error", err)
			continue
		}

		for _, p := range selectedPeers {
			addr, err := net.ResolveUDPAddr("udp", p)
			if err != nil {
				continue
			}
			_, err = conn.WriteToUDP(data, addr)
			if err != nil {
				slog.Debug("gossip: failed to send to peer", "peer", p, "error", err)
			}
		}
	}
}

// Broadcast injects a new locally-generated event into the network.
func (g *Protocol) Broadcast(event EventKind, payload []byte) {
	msg := g.CreateMessage(event, payload)
	// Process it locally (which adds it to seenSet and triggers callbacks)
	g.HandleMessage(msg)
	// Buffer it so the dissemination loop picks it up
	g.bufferForDissemination(msg)
}
