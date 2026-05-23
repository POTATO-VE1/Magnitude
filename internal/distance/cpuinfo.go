package distance

import (
	"log/slog"
	"runtime"

	"golang.org/x/sys/cpu"
)

// CPUFeatures reports available SIMD/acceleration features for distance computation.
type CPUFeatures struct {
	Arch      string
	HasAVX2   bool
	HasAVX512 bool
	HasNEON   bool
	HasFMA    bool
	BatchImpl string // "asm-avx2", "asm-neon", or "pure-go"
}

// DetectCPUFeatures returns the detected CPU features for the current platform.
func DetectCPUFeatures() CPUFeatures {
	f := CPUFeatures{
		Arch: runtime.GOARCH,
	}

	switch runtime.GOARCH {
	case "amd64":
		f.HasAVX2 = cpu.X86.HasAVX2
		f.HasAVX512 = cpu.X86.HasAVX512F
		f.HasFMA = cpu.X86.HasFMA
		if f.HasAVX2 && f.HasFMA {
			f.BatchImpl = "asm-avx2"
		} else {
			f.BatchImpl = "pure-go"
		}
	case "arm64":
		f.HasNEON = cpu.ARM64.HasASIMD
		if f.HasNEON {
			f.BatchImpl = "asm-neon"
		} else {
			f.BatchImpl = "pure-go"
		}
	default:
		f.BatchImpl = "pure-go"
	}

	return f
}

// LogCPUFeatures logs the detected CPU features at startup.
func LogCPUFeatures() {
	f := DetectCPUFeatures()
	slog.Info("distance computation",
		"arch", f.Arch,
		"implementation", f.BatchImpl,
		"avx2", f.HasAVX2,
		"avx512", f.HasAVX512,
		"neon", f.HasNEON,
		"fma", f.HasFMA,
	)
}
