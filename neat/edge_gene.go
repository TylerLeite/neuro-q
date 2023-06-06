package neat

import (
	"fmt"

	"github.com/TylerLeite/neuro-q/ma"
)

var NextInnovationNumber uint

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
	return fmt.Sprintf("{%d,%d}", e.InNode, e.OutNode)
}

var InnovationHistory map[string]uint

func NewEdgeGene(in, out uint, weight float64) *EdgeGene {
	e := EdgeGene{
		InNode:           in,
		OutNode:          out,
		Enabled:          true,
		Weight:           weight,
		InnovationNumber: 0,
	}

	// Track innovations at the source of new connection genes, this way the check is never missed + changes are localized here
	if innovationNumber, ok := InnovationHistory[e.ToString()]; ok {
		e.InnovationNumber = innovationNumber
	} else {
		e.InnovationNumber = NextInnovationNumber
		InnovationHistory[e.ToString()] = NextInnovationNumber
		NextInnovationNumber += 1
	}

	return &e
}
