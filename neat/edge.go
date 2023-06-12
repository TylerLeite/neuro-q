package neat

import "fmt"

type Edge struct {
	In      *Node
	Out     *Node
	Weight  float64
	Enabled bool

	Label string
}

func NewEdge(n, c *Node) *Edge {
	e := new(Edge)
	e.In = n
	e.Out = c
	e.Weight = 1.0
	e.Enabled = true
	return e
}

func (e *Edge) ToString() string {
	return fmt.Sprintf("{%s %s @ %.2g}", e.In.Label, e.Out.Label, e.Weight)
}

func (e *Edge) ForwardPropogate() {
	e.Out.ForwardPropogate()
}

func (e *Edge) Reset() {
	if e.In.state != Unactivated {
		e.In.Reset()
	}
}
