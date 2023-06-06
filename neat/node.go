package neat

import (
	"math"
	"sync"
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

	state ActivationState
	lock  *sync.Cond

	fn    ActivationFunction // activation function
	value float64
}

func NewNode(activation ActivationFunction) *Node {
	n := new(Node)
	n.In = make([]*Edge, 0)
	n.Out = make([]*Edge, 0)

	n.state = Unactivated

	m := sync.Mutex{}
	n.lock = sync.NewCond(&m)

	n.fn = activation
	n.value = math.NaN()

	return n
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

func (n *Node) CalculateValue() float64 {
	for n.state == InActivation {
		n.lock.Wait()
	}

	if n.state == Unactivated {
		n.lock.L.Lock()
		n.state = InActivation

		if len(n.In) > 0 {
			// this is not an input node

			sum := 0.0
			// need to sum all parent inputs
			for _, p := range n.In {
				if p == nil {
					continue
				}
				sum += p.CalculateValue() // make sure u send input upstream
			}

			// then run that through the activation function
			n.value = n.fn(sum)
		}

		n.state = Activated
		n.lock.L.Unlock()
		n.lock.Broadcast()
	}

	//fmt.Println("Activating: " + n.FnDesc)
	return n.value
}

// for now, assume reset is only called after a given calculate is done running
func (n *Node) Reset() {
	if n.state == Unactivated {
		return
	}

	n.state = Unactivated
	for _, edge := range n.In {
		edge.Reset()
	}
}
