package neat

import "fmt"

var NextInnovationNumber uint

type Connection struct {
	InNode           uint
	OutNode          uint
	Enabled          bool
	Weight           float64
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

	if innovationNumber, ok := mutations[c.Repr()]; ok {
		c.InnovationNumber = innovationNumber
	} else {
		c.InnovationNumber = NextInnovationNumber
		mutations[c.Repr()] = NextInnovationNumber
		NextInnovationNumber += 1
	}

	return &c
}
