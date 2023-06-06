package neat

type Edge struct {
	In      *Node
	Out     *Node
	Weight  float64
	Enabled bool
}

func NewEdge(n, c *Node) *Edge {
	e := new(Edge)
	e.In = n
	e.Out = c
	e.Weight = 1.0
	e.Enabled = true
	return e
}

func (e *Edge) CalculateValue() float64 {
	return e.Weight * e.In.CalculateValue()
}

func (e *Edge) Reset() {
	e.In.Reset()
}
