package neat

import (
	"fmt"
	"math"
)

type ActivationState int8
type ActivationFunction func(float64) float64

func SigmoidFunc(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-4.9*x))
}

const (
	Unactivated ActivationState = iota
	InActivation
	Activated
)

type Node struct {
	In  []*Edge
	Out []*Edge

	Label string

	state ActivationState

	fn    ActivationFunction
	value float64
}

func NewNode(activation ActivationFunction) *Node {
	n := new(Node)
	n.In = make([]*Edge, 0)
	n.Out = make([]*Edge, 0)

	n.state = Unactivated

	n.fn = activation
	n.value = math.NaN()

	return n
}

func (n *Node) ToString() string {
	inRepr := ""
	for _, edge := range n.In {
		inRepr += edge.ToString() + ", "
	}

	outRepr := ""
	for _, edge := range n.Out {
		outRepr += edge.ToString() + ", "
	}

	return fmt.Sprintf("%s:\nin: %s\nout: %s\n", n.Label, inRepr, outRepr)
}

func (n *Node) Value() float64 {
	return n.value
}

func (n *Node) SetDefaultValue(m float64) *Node {
	n.value = m
	return n
}

func (n *Node) AddChild(c *Node) *Edge {
	e := NewEdge(n, c)

	n.Out = append(n.Out, e)
	c.In = append(c.In, e)
	return e
}

func (n *Node) ForwardPropogate() {
	if n.state != Activated {
		if len(n.In) == 0 {
			n.state = Activated
			n.value = n.fn(n.value)
		} else {
			// Get sum of inputs
			sum := 0.0
			for _, p := range n.In {
				if p.In.state != Activated {
					return
				} else {
					sum += p.In.value
				}
			}
			n.state = Activated
			n.value = n.fn(sum)
		}
	}

	// Pay it forward
	for _, p := range n.Out {
		p.ForwardPropogate()
	}
}

// For now, assume reset is only called after a given calculate is done running
func (n *Node) Reset() {
	if n.state == Unactivated {
		return
	}

	n.state = Unactivated
	n.value = math.NaN()
	for _, edge := range n.In {
		edge.Reset()
	}
}
