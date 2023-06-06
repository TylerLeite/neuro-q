package neat

import (
	_ "fmt"
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

func (n *Network) Copy() ma.Organism {
	copyDna := n.DNA.Copy().(*Genome)
	out := NewNetwork(copyDna, n.Population)
	return ma.Organism(out)
}

func (n *Network) RandomNeighbor() ma.Organism {
	return nil
}

func (n *Network) NewFromGeneticCode(ma.GeneticCode) ma.Organism {
	return nil
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

	g := Genome{
		Connections: make([]*EdgeGene, 0),
		SensorNodes: make([]uint, 0),
		HiddenNodes: make([]uint, 0),
		OutputNodes: make([]uint, 0),
	}

	// Line up genes by innovation number
	var i1, i2 int
	for {
		// Check if we are in excess node territory
		if i1 >= len(g1.Connections) {
			if i2 >= len(g2.Connections) {
				break
			} else if n == moreFitParent {
				// Inherit excess genes from the more fit parent
				for ; i1 < len(g1.Connections); i1 += 1 {
					g.Connections = append(g.Connections, g1.Connections[i1])
				}
			}
		} else if i2 >= len(g2.Connections) && n2 == moreFitParent {
			// Inherit excess genes from the more fit parent
			for ; i2 < len(g2.Connections); i2 += 1 {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}
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
	// Assume dna has populated node slices. This should always be the case

	// Create all nodes
	nNodes := len(n.DNA.SensorNodes) + len(n.DNA.HiddenNodes) + len(n.DNA.OutputNodes)
	nodeMap := make([]*Node, nNodes)

	// TODO: support for other activation functions
	// Also there is probably a slightly cleaner way of doing this than 3 nearly identical loops but oh well
	for _, v := range n.DNA.SensorNodes {
		nodeMap[v] = NewNode(SigmoidFunc)
		n.Nodes = append(n.Nodes, nodeMap[v])
	}

	for _, v := range n.DNA.HiddenNodes {
		nodeMap[v] = NewNode(SigmoidFunc)
		n.Nodes = append(n.Nodes, nodeMap[v])
	}

	for _, v := range n.DNA.OutputNodes {
		nodeMap[v] = NewNode(SigmoidFunc)
		n.Nodes = append(n.Nodes, nodeMap[v])
	}

	// Create all edges
	for _, v := range n.DNA.Connections {
		newEdge := nodeMap[v.InNode].AddChild(nodeMap[v.OutNode])
		n.Edges = append(n.Edges, newEdge)
	}

	n.isCompiled = true
	return nil
}

func (n *Network) IsCompiled() bool {
	return n.isCompiled
}
