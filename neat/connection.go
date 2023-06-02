package neat

import "fmt"

var NextInnovationNumber uint

type MutationType uint8

const (
	MutationAddNode MutationType = iota
	MutationAddEdge
)

type Connection struct {
	InNode  uint
	OutNode uint
	Enabled bool
	Weight  float64

	Origin           MutationType
	InnovationNumber uint
}

func (c *Connection) Repr() string {
	return fmt.Sprintf("%d.%d", c.InNode, c.OutNode)
}

func NewConnection(in, out uint, weight float64, mutations map[string]uint) *Connection {
	c := Connection{
		InNode:           in,
		OutNode:          out,
		Enabled:          true,
		Weight:           weight,
		InnovationNumber: 0,
	}

	// Track innovations at the source of new connection genes, this way the check is never missed + changes are localized here
	if innovationNumber, ok := mutations[c.Repr()]; ok {
		c.InnovationNumber = innovationNumber
	} else {
		c.InnovationNumber = NextInnovationNumber
		mutations[c.Repr()] = NextInnovationNumber
		NextInnovationNumber += 1
	}

	return &c
}
