package neat

import (
	"fmt"
	"math"

	"github.com/TylerLeite/neuro-q/ma"
)

var NextInnovationNumber uint = 0

type EdgeGene struct {
	InNode  uint
	OutNode uint
	Enabled bool
	Weight  float64

	Origin           ma.MutationType
	InnovationNumber uint
}

func (e *EdgeGene) Copy() *EdgeGene {
	newEdge := EdgeGene{
		InNode:  e.InNode,
		OutNode: e.OutNode,
		Enabled: e.Enabled,
		Weight:  e.Weight,

		Origin:           e.Origin,
		InnovationNumber: e.InnovationNumber,
	}

	return &newEdge
}

func (e *EdgeGene) ToString() string {
	enabledStr := "|"
	if !e.Enabled {
		enabledStr = "x"
	}
	return fmt.Sprintf("{%d %s %d,%d (%d) @ %.2f}", e.InnovationNumber, enabledStr, e.InNode, e.OutNode, e.Origin, e.Weight)
}

var b64 = []string{
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	"-", "_",
}

func uintToBase64(n uint, padding int) string {
	digits := ""
	for n > 0 {
		digits = b64[n%64] + digits
		n = n / 64
	}

	for len(digits) < padding {
		digits = "0" + digits
	}

	return digits
}

func (e *EdgeGene) ToRep(padTo int) string {
	if !e.Enabled {
		return ""
	}

	return fmt.Sprintf("%s%s%s",
		uintToBase64(e.InNode, padTo),
		uintToBase64(e.OutNode, padTo),
		uintToBase64(uint(math.Float64bits(e.Weight)), 11),
	)
}

func (e *EdgeGene) InnovationKey() string {
	return fmt.Sprintf("%d|%d->%d", e.Origin, e.InNode, e.OutNode)
}

var (
	InnovationHistory = make(map[string]uint)
)

func ResetInnovationHistory() {
	InnovationHistory = make(map[string]uint)
	NextInnovationNumber = 0
}

func NewEdgeGene(in, out uint, weight float64, origin ma.MutationType) *EdgeGene {
	e := EdgeGene{
		InNode:  in,
		OutNode: out,
		Enabled: true,
		Weight:  weight,

		Origin:           origin,
		InnovationNumber: 0,
	}

	// Track innovations at the source of new connection genes, this way the check is never missed + changes are localized here
	if _, ok := InnovationHistory[e.InnovationKey()]; !ok {
		InnovationHistory[e.InnovationKey()] = NextInnovationNumber
		NextInnovationNumber += 1
	}

	e.InnovationNumber = InnovationHistory[e.InnovationKey()]
	return &e
}
