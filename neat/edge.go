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

func (e *Edge) CalculateValue(sumChan chan float64) float64 {
	chainChan := make(chan float64)
	go e.In.CalculateValue(chainChan)
	value := e.Weight * <-chainChan
	if sumChan != nil {
		sumChan <- value
	}
	return value
}

func (e *Edge) Reset() {
	e.In.Reset()
}
