package neat

import (
	"fmt"
	"math/rand"

	"github.com/TylerLeite/neuro-q/ma"
)

// var nextId uint64 = 0

// const fnLength = 2

type GraphPart interface {
	CalculateValue() float64
	Reset()
}

type Network struct {
	Nodes []*Node
	Edges []*Edge

	DNA        *Genome
	isCompiled bool

	// Because fitness function is defined on populations and crossover is defined on organisms, need a reference here
	// TODO: maybe add crossover as a function member of population like fitness is?
	Population *ma.Population
}

func NilNetwork() *Network {
	n := Network{
		Nodes: make([]*Node, 0),
		Edges: make([]*Edge, 0),
		DNA:   nil,
	}

	return &n
}

func NewNetwork(dna *Genome, p *ma.Population) *Network {
	n := Network{
		Nodes:      make([]*Node, 0),
		Edges:      make([]*Edge, 0),
		DNA:        dna,
		Population: p,
	}

	n.Compile()
	return &n
}

func (n *Network) ToString() string {
	nodesRepr := ""
	for _, node := range n.Nodes {
		nodesRepr += node.ToString()
	}
	return nodesRepr
}

func (n *Network) Copy() ma.Organism {
	copyDna := n.DNA.Copy().(*Genome)
	out := NewNetwork(copyDna, n.Population)
	return ma.Organism(out)
}

func (n *Network) RandomNeighbor() ma.Organism {
	neighbor := n.Copy()

	// TODO: get from config
	args := MutateArgs{
		FeedForward: true,
	}

	r := rand.Float64()
	mutation := MutationAddConnection
	if r < 0.1 {
		mutation = MutationMutateWeight
	} else if r < 0.3 {
		mutation = MutationAddNode
	}

	neighbor.GeneticCode().Mutate(mutation, args)
	return neighbor
}

func (n *Network) NewFromGeneticCode(geneticCode ma.GeneticCode) ma.Organism {
	dna := geneticCode.(*Genome)
	out := NewNetwork(dna, n.Population)
	return ma.Organism(out)
}

func (n *Network) Crossover(others []ma.Organism) ma.Organism {
	// TODO: either check to make sure others is only 1 element long or support N >= 1 parents
	n2 := others[0].(*Network)

	// Need to know which parent is more fit for inheriting excess and disjoint genes
	moreFitParent := n
	if n.Population.FitnessOf(n) < n2.Population.FitnessOf(n2) {
		moreFitParent = n2
	}

	g1 := n.GeneticCode().(*Genome)
	g2 := n2.GeneticCode().(*Genome)

	// TODO: keep these sorted so this doesn't need to be run more than once per genome
	g1.SortConnections()
	g2.SortConnections()

	g := Genome{
		Connections: make([]*EdgeGene, 0),
		SensorNodes: make([]uint, 0),
		HiddenNodes: make([]uint, 0),
		OutputNodes: make([]uint, 0),
	}

	// Line up genes by innovation number
	var i1, i2 int
	// Need to sort connections slices by innovation number
	for {
		// Check if we are in excess node territory
		if i1 >= len(g1.Connections) {
			// Inherit excess genes from the longer genome if it is the more fit parent
			if n2 == moreFitParent {
				// This loop will be empty if i2 >= len(g2.Connection)
				for ; i2 < len(g2.Connections); i2 += 1 {
					g.Connections = append(g.Connections, g2.Connections[i2])
				}
			}
			break
		} else if i2 >= len(g2.Connections) {
			if n == moreFitParent {
				for ; i1 < len(g1.Connections); i1 += 1 {
					g.Connections = append(g.Connections, g1.Connections[i1])
				}
			}
			break
		}

		if g1.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Inherit a gene randomly when there is an innovation number  match
			if rand.Intn(2) == 0 {
				g.Connections = append(g.Connections, g1.Connections[i1])
			} else {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}

			// TODO: check gene disable safety
			// If either gene is disabled, there is a 75% chance the inherited gene is disabled as well
			// (as long as it's safe to do so)
			// if !g1.Connections[i1].Enabled || !g2.Connections[i2].Enabled && rand.Intn(4) == 0 {
			// 	g.Connections[len(g.Connections)-1].Enabled = false
			// }

			i1 += 1
			i2 += 1
		} else if g1.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			// Inherit disjoint genes from the more fit parent
			if n == moreFitParent {
				g.Connections = append(g.Connections, g1.Connections[i1])
			}

			i1 += 1
		} else if g1.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			if n2 == moreFitParent {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}

			i2 += 1
		}
	}

	// Make an organism out of this genome
	return n.NewFromGeneticCode(ma.GeneticCode(&g))
}

func (n *Network) GeneticCode() ma.GeneticCode {
	return ma.GeneticCode(n.DNA)
}

func (n *Network) LoadGeneticCode(dna ma.GeneticCode) {
	n.DNA = dna.(*Genome)
}

// TODO: check for errors e.g. loops in feed-forward networks, connections between nonexistant nodes, etc.
func (n *Network) Compile() error {
	// No need to recompile, genome should never change
	if n.isCompiled {
		return nil
	}

	// TODO: figure out where this isn't being called but should be (maybe crossover?)
	n.DNA.PopulateNodeSlices()

	// Create all nodes
	nNodes := len(n.DNA.SensorNodes) + len(n.DNA.HiddenNodes) + len(n.DNA.OutputNodes)
	n.Nodes = make([]*Node, nNodes)
	nodeMap := make([]*Node, nNodes)

	// TODO: support for other activation functions
	// Also there is probably a slightly cleaner way of doing this than 3 nearly identical loops but oh well
	for _, v := range n.DNA.SensorNodes {
		nodeMap[v] = NewNode(IdentityFunc)
		nodeMap[v].Label = fmt.Sprintf("%d", v)
		n.Nodes[v] = nodeMap[v]
	}

	for _, v := range n.DNA.HiddenNodes {
		nodeMap[v] = NewNode(SigmoidFunc)
		nodeMap[v].Label = fmt.Sprintf("%d", v)
		n.Nodes[v] = nodeMap[v]
	}

	for _, v := range n.DNA.OutputNodes {
		nodeMap[v] = NewNode(SigmoidFunc) // TODO: is this the best activation function for output?
		nodeMap[v].Label = fmt.Sprintf("%d", v)
		n.Nodes[v] = nodeMap[v]
	}

	// Create all edges
	for _, v := range n.DNA.Connections {
		newEdge := nodeMap[v.InNode].AddChild(nodeMap[v.OutNode])
		newEdge.Label = fmt.Sprintf("%d", v.InnovationNumber)
		newEdge.Weight = v.Weight
		n.Edges = append(n.Edges, newEdge)
	}

	n.isCompiled = true
	return nil
}

func (n *Network) IsCompiled() bool {
	return n.isCompiled
}
