package neat

import (
	"math"
	"math/rand"

	"github.com/TylerLeite/neuro-q/ma"
)

// ma.GeneticCode is an interface into genomes
// neat.Genome is a network definition, it implements ma.GeneticCode
// cppn.Organism is an actual, usable network. It is compiled from a neat.Genome

// TODO: node genes, will they always just work out if they go sensor -> output -> hidden?
// TODO: bias flag for nodes
type Genome struct {
	Connections []*EdgeGene
	SensorNodes []uint
	HiddenNodes []uint
	OutputNodes []uint
}

func (g *Genome) Copy() ma.GeneticCode {
	newGenome := &Genome{
		Connections: make([]*EdgeGene, len(g.Connections)),
		SensorNodes: make([]uint, len(g.SensorNodes)),
		HiddenNodes: make([]uint, len(g.HiddenNodes)),
		OutputNodes: make([]uint, len(g.OutputNodes)),
	}

	for i, v := range g.Connections {
		newGenome.Connections[i] = v.Copy()
	}

	copy(newGenome.SensorNodes, g.SensorNodes)
	copy(newGenome.HiddenNodes, g.HiddenNodes)
	copy(newGenome.OutputNodes, g.OutputNodes)

	return ma.GeneticCode(newGenome)
}

func (g *Genome) Randomize() ma.GeneticCode {
	newGenome := g.Copy().(*Genome)
	newGenome.Connections = make([]*EdgeGene, 0)
	newGenome.HiddenNodes = make([]uint, 0)
	newGenome.SensorNodes = make([]uint, 0)
	newGenome.OutputNodes = make([]uint, 0)

	inNodes := len(g.SensorNodes)
	outNodes := len(g.OutputNodes)

	// Make sure all nodes are connected at least once
	nodesConnected := make([]bool, outNodes)
	for s := 0; s < inNodes; s += 1 {
		outNode := uint(rand.Intn(int(outNodes)))
		nodesConnected[outNode] = true

		c := NewEdgeGene(uint(s), outNode, rand.Float64()-0.5)
		newGenome.Connections = append(newGenome.Connections, c)
	}

	for o := 0; o < outNodes; o += 1 {
		// No need to double up
		if nodesConnected[o] {
			continue
		}

		inNode := uint(rand.Intn(int(inNodes)))
		c := NewEdgeGene(inNode, uint(o), rand.Float64()-0.5)
		newGenome.Connections = append(newGenome.Connections, c)
	}

	newGenome.PopulateNodeSlices()

	return ma.GeneticCode(newGenome)
}

func (g *Genome) ToString() string {
	edges := "["
	for _, v := range g.Connections {
		edges += v.ToString() + " "
	}
	edges += "]"

	// Shouldn't need to include nodes in the representation. That would be redundant considering nodes are generated based on edges
	return edges
}

// NOTE: need to map innovation numbers within a generation to specific mutations
// -> keep a list of innovations that occurred in this generation

const (
	MutationAddConnection ma.MutationType = iota
	MutationAddNode
)

func (g *Genome) ListMutations() map[string]ma.MutationType {
	m := make(map[string]ma.MutationType)
	m["Add Connection"] = MutationAddConnection
	m["Add Node"] = MutationAddNode
	return m
}

// Neat genome mutation requires arguments to be passed, define them in a struct
type MutateArgs struct {
	FeedForward bool
}

func (g *Genome) Mutate(typ ma.MutationType, args interface{}) ma.GeneticCode {
	switch typ {
	case MutationAddConnection:
		g.AddConnection(args.(MutateArgs).FeedForward)
	case MutationAddNode:
		g.AddNode()
	default:
		return nil
	}
	return ma.GeneticCode(g)
}

func (g *Genome) AddConnection(feedForward bool) {
	nIn := len(g.SensorNodes) + len(g.HiddenNodes)
	nOut := len(g.HiddenNodes) + len(g.OutputNodes)

	// Are both nodes hidden? then order the edge small -> low
	isHidden := false

	r1 := rand.Intn(nIn)
	if r1 >= len(g.SensorNodes) {
		r1 -= len(g.SensorNodes)
		r1 = int(g.HiddenNodes[r1])
		isHidden = true
	} else {
		r1 = int(g.SensorNodes[r1])
	}

	r2 := r1

	for r2 == r1 {
		r2 = rand.Intn(nOut)
		if r2 >= len(g.HiddenNodes) {
			r2 -= len(g.HiddenNodes)
			r2 = int(g.OutputNodes[r2])
			isHidden = false
		} else {
			r2 = int(g.HiddenNodes[r2])
		}
	}

	if feedForward && isHidden && r2 < r1 {
		r1, r2 = r2, r1
	}

	connection := NewEdgeGene(uint(r1), uint(r2), rand.Float64())
	g.Connections = append(g.Connections, connection)
}

func (g *Genome) AddNode() {
	// Randomly pick a connection to bifurcate
	randomGene := (g.Connections)[rand.Intn(len(g.Connections))]

	nextNode := g.HiddenNodes[len(g.HiddenNodes)-1] + 1

	// Create two new connection genes to fit this node into the network
	// -> new is weight 1, new -> is the old edge's weight
	new1 := NewEdgeGene(randomGene.InNode, nextNode, 1)
	new2 := NewEdgeGene(nextNode, randomGene.OutNode, randomGene.Weight)

	// Disable the old connection
	randomGene.Enabled = false

	g.Connections = append(g.Connections, []*EdgeGene{new1, new2}...)
	g.HiddenNodes = append(g.HiddenNodes, nextNode)
}

type boolpair []bool

func (g *Genome) PopulateNodeSlices() {
	nodeClassifications := make(map[uint]boolpair)

	for _, edge := range g.Connections {
		if _, ok := nodeClassifications[edge.InNode]; !ok {
			nodeClassifications[edge.InNode] = boolpair{false, false}
		}
		if _, ok := nodeClassifications[edge.OutNode]; !ok {
			nodeClassifications[edge.OutNode] = boolpair{false, false}
		}

		// Gotta do it this way or face the red squiggly underlines of doom
		nodeClassifications[edge.InNode][0] = true
		nodeClassifications[edge.OutNode][1] = true
	}

	// TODO: finish this
}

func (g *Genome) DistanceFrom(gc ma.GeneticCode, c1, c2, c3 float64) float64 {
	g2 := gc.(*Genome)

	var (
		nExcess   float64 // excess genes, how many are at the end of each genome after the last matching gene
		nDisjoint float64 // disjoint genes, how many are misaligned before the last matching gene
		nShared   float64 // shared genes, how many match in innovation number
		N         float64 // N is the number of genes in the larger genome
		W         float64 // average weight difference between matching genes
	)

	// It's the same iteration as during crossover
	var i1, i2 int
	for {
		// Check if we are in excess node territory
		if i1 >= len(g.Connections) {
			if i2 < len(g2.Connections)-1 {
				nExcess = float64(len(g2.Connections) - i2 - 1)
			}

			break
		} else if i2 >= len(g2.Connections) {
			if i1 < len(g.Connections)-1 {
				nExcess = float64(len(g.Connections) - i1 - 1)
			}
		}

		if g.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Check difference in weights for shared genes
			nShared += 1
			W += math.Abs(g.Connections[i1].Weight - g2.Connections[i2].Weight)
		} else if g.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i1 += 1
		} else if g.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i2 += 1
		}
	}

	W /= nShared

	// Should never be negative, but take absolute value just in case
	return math.Abs(c1*nExcess/N + c2*nDisjoint/N + c3*W)
}
