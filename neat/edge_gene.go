package neat

import (
	"fmt"

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
	return fmt.Sprintf("{%d | %d,%d @ %.2f}", e.InnovationNumber, e.InNode, e.OutNode, e.Weight)
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
