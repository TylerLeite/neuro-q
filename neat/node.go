package neat

import (
	"fmt"
	"math"

	"github.com/TylerLeite/neuro-q/log"
)

type ActivationState int8

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

	fn    ActivationFunction
	value float64

	visitedBy map[*Edge]bool // only used during activation
}

func NewNode(activation ActivationFunction, typ NodeType) *Node {
	n := Node{
		In:  make([]*Edge, 0),
		Out: make([]*Edge, 0),

		Type: typ,

		fn:    activation,
		value: math.NaN(),

		visitedBy: make(map[*Edge]bool),
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

func (n *Node) State() ActivationState {
	timesActivated := len(n.visitedBy)
	maxActivations := len(n.In)

	if timesActivated == 0 {
		return Unactivated
	} else if timesActivated < maxActivations {
		return InActivation
	} else {
		return Activated
	}
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
	log.Book(fmt.Sprintf("%s not activated, trying activation\n", n.Label), log.DEBUG, log.DEBUG_PROPAGATION)
	if len(n.In) == 0 {
		log.Book(fmt.Sprintf("%s is an input node\n", n.Label), log.DEBUG, log.DEBUG_PROPAGATION)
		n.value = n.fn(n.value)
	} else {
		// Get sum of inputs
		sum := 0.0
		bias := 0.0
		allActivated := true

		for _, p := range n.In {
			if math.IsNaN(p.In.value) {
				log.Book(fmt.Sprintf("%s has an input which is not yet activated (%s)\n", n.Label, p.In.Label), log.DEBUG, log.DEBUG_PROPAGATION)
				allActivated = false
			} else if p.In.Type != BiasNode {
				sum += p.In.value * p.Weight
				log.Book(fmt.Sprintf("Adding #%s (value = %.2g*%.2g) to sum\n", p.In.Label, p.In.value, p.Weight), log.DEBUG, log.DEBUG_PROPAGATION)
			} else {
				bias = p.In.value * p.Weight
				log.Book(fmt.Sprintf("Adding #%s (value = %.2g*%.2g) to bias\n", p.In.Label, p.In.value, p.Weight), log.DEBUG, log.DEBUG_PROPAGATION)
			}
		}

		log.Book(fmt.Sprintf("Sum is %.2g, bias is %.2g\n", sum, bias), log.DEBUG, log.DEBUG_PROPAGATION)
		n.value = n.fn(sum) + bias

		if allActivated {
			log.Book(fmt.Sprintf("%s activation finished! Value = %.2g\n", n.Label, n.value), log.DEBUG, log.DEBUG_PROPAGATION)

		} else {
			// State stays "in activation"
			log.Book(fmt.Sprintf("%s activation unfinished, partial value = %.2g\n", n.Label, n.value), log.DEBUG, log.DEBUG_PROPAGATION)
		}

	}

	// Pay it forward
	for _, edge := range n.Out {
		log.Book(fmt.Sprintf("Checking if %s has propagated to %s yet\n", edge.In.Label, edge.Out.Label), log.DEBUG, log.DEBUG_PROPAGATION)

		if _, ok := edge.Out.visitedBy[edge]; !ok {
			log.Book(fmt.Sprintf("Propagating to %s\n", edge.Out.Label), log.DEBUG, log.DEBUG_PROPAGATION)
			edge.ForwardPropogate()
		} else {
			log.Book(fmt.Sprintf("Not propagating to %s\n", edge.Out.Label), log.DEBUG, log.DEBUG_PROPAGATION)
		}
	}
}

// For now, assume reset is only called after a given calculate is done running
func (n *Node) Reset() {
	log.Book(fmt.Sprintf("Resetting %s\n", n.Label), log.DEBUG, log.DEBUG_RESET)
	n.value = math.NaN()
	n.visitedBy = make(map[*Edge]bool)
}

func (n *Node) Deactivate() {
	log.Book(fmt.Sprintf("Deactivating %s\n", n.Label), log.DEBUG, log.DEBUG_RESET)
	n.visitedBy = make(map[*Edge]bool)
}
