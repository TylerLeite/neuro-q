package neat

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/TylerLeite/neuro-q/log"
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
	n.isCompiled = false

	// TODO: get from config
	args := MutateArgs{
		FeedForward: true,
	}

	r := rand.Float64()
	mutation := MutationAddNode

	for k, v := range n.DNA.MutationOdds() {
		r -= v
		if r <= 0 {
			mutation = k
		}
	}

	neighbor.GeneticCode().Mutate(mutation, args)

	// Check validity
	if log.DEBUG_MUTATION {
		neighbor.Compile()
		network := neighbor.(*Network)

		// Check for disconnected nodes
		foundErrors := false
		for _, node := range network.Nodes {
			if len(node.In) == 0 && len(node.Out) == 0 {
				foundErrors = true
				break
			}
		}

		for _, edge := range network.Edges {
			if edge.In == edge.Out {
				foundErrors = true
			}
		}

		if foundErrors {
			log.Book(fmt.Sprintf("vvvvvvvvvv\n%s\n %s\n", network.ToString(), neighbor.GeneticCode().ToString()), log.DEBUG_MUTATION)
			panic("Found errors in RandomNeighbor()")
		}
	}

	return neighbor
}

func (n *Network) NewFromGeneticCode(geneticCode ma.GeneticCode) ma.Organism {
	dna := geneticCode.(*Genome)
	out := NewNetwork(dna, n.Population)
	return ma.Organism(out)
}

func insertActivation(activationFunctions map[uint]ActivationFunction, g *Genome, i int) {
	if activationFunctions != nil {
		if _, ok := activationFunctions[g.Connections[i].InNode]; !ok {
			activationFunctions[g.Connections[i].InNode] = g.ActivationFunctionOf(g.Connections[i].InNode)
		}
		if _, ok := activationFunctions[g.Connections[i].OutNode]; !ok {
			activationFunctions[g.Connections[i].OutNode] = g.ActivationFunctionOf(g.Connections[i].OutNode)
		}
	}
}

func (n *Network) Crossover(others []ma.Organism) ma.Organism {
	// TODO: either check to make sure others is only 1 element long or support N >= 1 parents
	n2 := others[0].(*Network)

	// Need to know which parent is more fit for inheriting excess and disjoint genes
	moreFitParent := n

	nFitness := n.Population.FitnessOf(n)
	n2Fitness := n2.Population.FitnessOf(n2)
	if nFitness < n2Fitness {
		moreFitParent = n2
	} else if nFitness == n2Fitness && len(n2.DNA.Connections) < len(n.DNA.Connections) {
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
		UsesBias:    g1.UsesBias, // if g1 uses bias, g2 sure ought to as well
	}

	// Also need to crossover activation functions, if parents use this feature
	var activationFunctions map[uint]ActivationFunction
	if g1.ActivationFunctions != nil && g2.ActivationFunctions != nil { // If one is nil, both should be
		activationFunctions = make(map[uint]ActivationFunction)
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
					g.Connections = append(g.Connections, g2.Connections[i2].Copy())
					insertActivation(activationFunctions, g2, i2)
				}
			}
			break
		} else if i2 >= len(g2.Connections) {
			if n == moreFitParent {
				for ; i1 < len(g1.Connections); i1 += 1 {
					g.Connections = append(g.Connections, g1.Connections[i1].Copy())
					insertActivation(activationFunctions, g1, i1)
				}
			}
			break
		}

		if g1.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Inherit a gene randomly when there is an innovation number  match
			if rand.Intn(2) == 0 {
				g.Connections = append(g.Connections, g1.Connections[i1].Copy())
				insertActivation(activationFunctions, g1, i1)
			} else {
				g.Connections = append(g.Connections, g2.Connections[i2].Copy())
				insertActivation(activationFunctions, g2, i2)
			}

			// TODO: check gene disable safety
			// If either gene is disabled, there is a 75% chance the inherited gene is disabled as well
			// (as long as it's safe to do so)
			// if !g1.Connections[i1].Enabled || !g2.Connections[i2].Enabled && rand.Intn(4) == 0 {
			// 	g.Connections[len(g.Connections)-1].Enabled = false
			// }

			// Unless both genes are disabled, enable this one
			// TODO: figure out why this is necessary
			if !g1.Connections[i1].Enabled && !g2.Connections[i2].Enabled {
				g.Connections[len(g.Connections)-1].Enabled = false
			} else {
				g.Connections[len(g.Connections)-1].Enabled = true
			}

			i1 += 1
			i2 += 1
		} else if g1.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			// Inherit disjoint genes from the more fit parent
			if n == moreFitParent {
				g.Connections = append(g.Connections, g1.Connections[i1].Copy())
				insertActivation(activationFunctions, g1, i1)
			}

			i1 += 1
		} else if g1.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			if n2 == moreFitParent {
				g.Connections = append(g.Connections, g2.Connections[i2].Copy())
				insertActivation(activationFunctions, g2, i2)

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
// TODO: potential bug if all connections are removed out of a node (e.g. through gene disable). should think if this is possible. may be the reason to have node genes
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

	log.Book(fmt.Sprintf("Allocating space for %d + %d + %d = %d nodes.\n", len(n.DNA.SensorNodes), len(n.DNA.HiddenNodes), len(n.DNA.OutputNodes), len(n.Nodes)), log.DEBUG, log.DEBUG_COMPILE)

	// TODO: support for other activation functions
	// Also there is probably a slightly cleaner way of doing this than 3 nearly identical loops but oh well
	for _, v := range n.DNA.SensorNodes {
		if v == 0 && n.DNA.UsesBias {
			n.Nodes[v] = NewNode(n.DNA.ActivationFunctionOf(v), BiasNode)
		} else {
			n.Nodes[v] = NewNode(n.DNA.ActivationFunctionOf(v), SensorNode)
		}
		n.Nodes[v].Label = fmt.Sprintf("%d", v)
	}

	for _, v := range n.DNA.HiddenNodes {
		n.Nodes[v] = NewNode(n.DNA.ActivationFunctionOf(v), HiddenNode)
		n.Nodes[v].Label = fmt.Sprintf("%d", v)
	}

	for _, v := range n.DNA.OutputNodes {
		n.Nodes[v] = NewNode(n.DNA.ActivationFunctionOf(v), OutputNode) // TODO: is this the best activation function for output?
		n.Nodes[v].Label = fmt.Sprintf("%d", v)
	}

	if log.DEBUG_COMPILE {
		for i, n := range n.Nodes {
			if n == nil || fmt.Sprintf("%d", i) != n.Label {
				log.Book(fmt.Sprintf("%s\n", n.ToString()), log.DEBUG, log.DEBUG_COMPILE)
				panic("Nodes not in order while compiling!")
			}
		}
	}

	// Create all edges
	for _, v := range n.DNA.Connections {
		if !v.Enabled {
			log.Book(fmt.Sprintf("Skipping disabled connection from %d to %d\n", v.InNode, v.OutNode), log.DEBUG, log.DEBUG_COMPILE)
			continue
		}
		log.Book(fmt.Sprintf("Adding connection from %d to %d\n", v.InNode, v.OutNode), log.DEBUG, log.DEBUG_COMPILE)
		newEdge := n.Nodes[v.InNode].AddChild(n.Nodes[v.OutNode])
		newEdge.Label = fmt.Sprintf("%d (%s)", v.InnovationNumber, MutationTypeToString[v.Origin])
		newEdge.Weight = v.Weight
		n.Edges = append(n.Edges, newEdge)
	}

	// TODO: sort edges?

	n.isCompiled = true
	return nil
}

func (n *Network) ForceCompile() error {
	n.isCompiled = false
	return n.Compile()
}

func (n *Network) IsCompiled() bool {
	return n.isCompiled
}

// TODO: This is ActivateRecurrent, also write ActivateFeedForward
func (n *Network) Activate(inputs []float64, sensors, outputs []*Node) error {
	for _, node := range n.Nodes {
		node.Reset()
	}

	done := false
	sanity := 100
	for !done && sanity > 0 {
		for i, in := range sensors {
			in.SetDefaultValue(inputs[i])
			log.Book(fmt.Sprintf("Sensor %s prop\n", in.Label), log.DEBUG, log.DEBUG_PROPAGATION)
			in.ForwardPropogate()
		}

		done = true
		for _, out := range outputs {
			if math.IsNaN(out.Value()) {
				done = false
			}
		}
		sanity -= 1
	}

	log.Book(fmt.Sprintf("\nProp trace\n%s\nGenome:\n\t%s\n%s\n", n.ToString(), n.DNA.NodesToString(), n.DNA.ToString()), log.DEBUG, log.DEBUG_PROPAGATION)

	if sanity <= 0 {
		return errors.New("canceling activation, too many loops in the network")
	}

	return nil
}
