package neat

import (
	"fmt"
	"math"
)

type ActivationState int8
type ActivationFunction func(float64) float64

func IdentityFunc(x float64) float64 {
	return x
}

func SigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-4.9*x))
}

const (
	Unactivated ActivationState = iota
	InActivation
	Activated
)

type NodeType uint8

const (
	BiasNode NodeType = iota
	SensorNode
	HiddenNode
	OutputNode
)

type Node struct {
	In  []*Edge
	Out []*Edge

	Label string

	Type NodeType

	state ActivationState

	fn    ActivationFunction
	value float64

	visited bool // only used during activation
}

func NewNode(activation ActivationFunction, typ NodeType) *Node {
	n := Node{
		In:  make([]*Edge, 0),
		Out: make([]*Edge, 0),

		Type: typ,

		state: Unactivated,

		fn:    activation,
		value: math.NaN(),
	}

	return &n
}

// TODO: rename this to ToPretty, more compact ToString
func (n *Node) ToString() string {
	inRepr := "< No inputs >"
	if len(n.In) > 0 {
		inRepr = "in: "
		for _, edge := range n.In {
			inRepr += edge.ToString() + ", "
		}
	}

	outRepr := "< No outputs >"
	if len(n.Out) > 0 {
		outRepr = "out: "
		for _, edge := range n.Out {
			outRepr += edge.ToString() + ", "
		}
	}

	return fmt.Sprintf("Node #%s:\n\t%s\n\t%s\n", n.Label, inRepr, outRepr)
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
	n.visited = true
	if n.state != Activated {
		Log(fmt.Sprintf("%s not activated, trying activation\n", n.Label), DEBUG, DEBUG_PROPAGATION)
		if len(n.In) == 0 {
			Log(fmt.Sprintf("%s is an input node\n", n.Label), DEBUG, DEBUG_PROPAGATION)
			n.state = Activated
			n.value = n.fn(n.value)
		} else {
			n.state = InActivation
			// Get sum of inputs
			sum := 0.0
			bias := 0.0
			// anyActivated := false
			allActivated := true
			for _, p := range n.In {
				if math.IsNaN(p.In.value) {
					Log(fmt.Sprintf("%s has an input which is not yet activated (%s)\n", n.Label, p.In.Label), DEBUG, DEBUG_PROPAGATION)
					allActivated = false
				} else if p.In.Type != BiasNode {
					sum += p.In.value * p.Weight
					Log(fmt.Sprintf("Adding #%s (value = %.2g*%.2g) to sum\n", p.In.Label, p.In.value, p.Weight), DEBUG, DEBUG_PROPAGATION)
					// anyActivated = true
				} else {
					bias = p.In.value * p.Weight
					Log(fmt.Sprintf("Adding #%s (value = %.2g*%.2g) to bias\n", p.In.Label, p.In.value, p.Weight), DEBUG, DEBUG_PROPAGATION)
					// anyActivated = true
				}
			}

			// if !anyActivated || (!allActivated && len(n.Out) == 0) {
			// 	// Don't set an output value unless all its inputs are activated
			// 	Log(fmt.Sprintf("%s activation unsucessful\n", n.Label), DEBUG, DEBUG_PROPAGATION)
			// 	return
			// }
			if !allActivated {
				// Don't set an output value unless all its inputs are activated
				Log(fmt.Sprintf("%s activation unsucessful\n", n.Label), DEBUG, DEBUG_PROPAGATION)
				return
			}

			if allActivated {
				n.state = Activated
			}

			Log(fmt.Sprintf("Sum is %.2g, bias is %.2g\n", sum, bias), DEBUG, DEBUG_PROPAGATION)
			n.value = n.fn(sum) + bias
			Log(fmt.Sprintf("%s activation successful! Value = %.2g\n", n.Label, n.value), DEBUG, DEBUG_PROPAGATION)
		}
	}

	// Pay it forward
	for _, p := range n.Out {
		if !p.Out.visited {
			Log(fmt.Sprintf("Propagating to %s\n", p.Out.Label), DEBUG, DEBUG_PROPAGATION)
			p.ForwardPropogate()
		}
	}
}

// For now, assume reset is only called after a given calculate is done running
func (n *Node) Reset() {
	Log(fmt.Sprintf("Resetting %s\n", n.Label), DEBUG, DEBUG_RESET)
	n.state = Unactivated
	n.value = math.NaN()
	n.visited = false
}

func (n *Node) Deactivate() {
	Log(fmt.Sprintf("Deactivating %s\n", n.Label), DEBUG, DEBUG_RESET)
	n.visited = false
}
