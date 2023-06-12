package neat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

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

func NewGenome(inNodes, outNodes int) *Genome {
	g := &Genome{
		Connections: make([]*EdgeGene, 0),
		SensorNodes: make([]uint, inNodes),
		HiddenNodes: make([]uint, 0),
		OutputNodes: make([]uint, outNodes),
	}

	g.Randomize()
	return g
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

func (g *Genome) Randomize() {
	inNodes := len(g.SensorNodes)
	outNodes := len(g.OutputNodes)

	g.Connections = make([]*EdgeGene, 0)
	g.HiddenNodes = make([]uint, 0)
	g.SensorNodes = make([]uint, 0)
	g.OutputNodes = make([]uint, 0)

	// Make sure all nodes are connected at least once
	nodesConnected := make([]bool, outNodes)
	for s := 0; s < inNodes; s += 1 {
		outNode := rand.Intn(outNodes)
		nodesConnected[outNode] = true
		outNode += inNodes

		c := NewEdgeGene(uint(s), uint(outNode), rand.Float64()-0.5, NoMutation)
		g.Connections = append(g.Connections, c)
	}

	for o := 0; o < outNodes; o += 1 {
		// No need to double up
		if nodesConnected[o] {
			continue
		}

		inNode := uint(rand.Intn(inNodes))
		c := NewEdgeGene(inNode, uint(o+inNodes), rand.Float64()-0.5, NoMutation)
		g.Connections = append(g.Connections, c)
	}

	g.PopulateNodeSlices()
}

func (g *Genome) ToString() string {
	edges := "["
	for _, v := range g.Connections {
		edges += v.ToString() + " "
	}
	edges = edges[:len(edges)-1] + "]"

	// Shouldn't need to include nodes in the representation. That would be redundant considering nodes are generated based on edges
	return edges
}

// NOTE: need to map innovation numbers within a generation to specific mutations
// -> keep a list of innovations that occurred in this generation

const (
	NoMutation ma.MutationType = iota
	MutationAddConnection
	MutationAddNode
	MutationMutateWeight
)

func (g *Genome) ListMutations() map[string]ma.MutationType {
	m := make(map[string]ma.MutationType)
	m["Add Connection"] = MutationAddConnection
	m["Add Node"] = MutationAddNode
	m["Mutate Weight"] = MutationMutateWeight
	return m
}

// Neat genome mutation requires arguments to be passed, define them in a struct
type MutateArgs struct {
	FeedForward bool
}

func (g *Genome) Mutate(typ ma.MutationType, args interface{}) {
	switch typ {
	case MutationAddConnection:
		err := g.AddConnection(args.(MutateArgs).FeedForward)
		if err != nil {
			g.AddNode()
		}
	case MutationAddNode:
		g.AddNode()
	case MutationMutateWeight:
		g.MutateWeight()
	default:
		// TODO: unknown mutation type error
		fmt.Printf("ERROR: Unknown mutation type: %d", typ)
	}
}

type AddConnectionError struct {
	Msg string
}

func NewAddConnectionError(msg string) *AddConnectionError {
	return &AddConnectionError{
		Msg: msg,
	}
}
func (e *AddConnectionError) Error() string {
	return e.Msg
}

func (g *Genome) AddConnection(feedForward bool) error {
	nIn := len(g.SensorNodes) + len(g.HiddenNodes)
	nOut := len(g.HiddenNodes) + len(g.OutputNodes)

	// Are both nodes hidden? then order the edge small -> low
	isHidden := false
	sanity := 100
	for sanity > 0 {
		var r1, r2 int
		for {
			r1 = rand.Intn(nIn)
			if r1 >= len(g.SensorNodes) {
				r1 -= len(g.SensorNodes)
				r1 = int(g.HiddenNodes[r1])
				isHidden = true
			} else {
				r1 = int(g.SensorNodes[r1])
				isHidden = false
			}

			r2 = rand.Intn(nOut)
			if r2 >= len(g.HiddenNodes) {
				r2 -= len(g.HiddenNodes)
				r2 = int(g.OutputNodes[r2])
				isHidden = false
			} else {
				r2 = int(g.HiddenNodes[r2])
			}

			if r2 == r1 {
				continue
			} else {
				break
			}
		}

		if feedForward && isHidden && r2 < r1 {
			r1, r2 = r2, r1
		}

		duplicateEdge := false
		for _, c := range g.Connections {
			if c.InNode == uint(r1) && c.OutNode == uint(r2) {
				duplicateEdge = true
				break
			}
		}

		if duplicateEdge {
			sanity -= 1
			continue
		}

		connection := NewEdgeGene(uint(r1), uint(r2), rand.Float64(), MutationAddConnection)
		g.Connections = append(g.Connections, connection)
		return nil
	}

	return NewAddConnectionError("Reached sanity limit trying to add a new connection")
}

func (g *Genome) AddNode() {
	// Randomly pick a connection to bifurcate
	randomGene := (g.Connections)[rand.Intn(len(g.Connections))]

	// Need to add a node between the two nodes of the existing connection. Figure out what to call that node
	var nextNode uint
	if len(g.HiddenNodes) > 0 {
		nextNode = g.HiddenNodes[len(g.HiddenNodes)-1] + 1
	} else {
		nextNode = uint(len(g.SensorNodes) + len(g.OutputNodes))
	}

	// Create two new connection genes to fit this node into the network
	// -> new is weight 1, new -> is the old edge's weight
	new1 := NewEdgeGene(randomGene.InNode, nextNode, 1, MutationAddNode)
	new2 := NewEdgeGene(nextNode, randomGene.OutNode, randomGene.Weight, MutationAddNode)

	// Disable the old connection
	randomGene.Enabled = false

	g.Connections = append(g.Connections, []*EdgeGene{new1, new2}...)
	g.HiddenNodes = append(g.HiddenNodes, nextNode)
}

func (g *Genome) MutateWeight() {
	randomGene := (g.Connections)[rand.Intn(len(g.Connections))]
	randomGene.Weight = rand.Float64() - 0.5
}

type boolpair []bool

func (g *Genome) PopulateNodeSlices() {
	// Make sure node slices aren't already populated
	if len(g.SensorNodes)+len(g.HiddenNodes)+len(g.OutputNodes) > 0 {
		g.SensorNodes = make([]uint, 0)
		g.HiddenNodes = make([]uint, 0)
		g.OutputNodes = make([]uint, 0)
	}

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

	for k, v := range nodeClassifications {
		if v[0] && v[1] {
			g.HiddenNodes = append(g.HiddenNodes, k)
		} else if v[0] && !v[1] {
			g.SensorNodes = append(g.SensorNodes, k)
		} else if !v[0] && v[1] {
			g.OutputNodes = append(g.OutputNodes, k)
		} else {
			// This should be logically impossible to reach
			fmt.Printf("Found an orphan node during PopulateNodeSlices!\n")
		}
	}
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

	N = math.Max(float64(len(g.Connections)), float64(len(g2.Connections)))

	// It's the same iteration as during crossover
	g.SortConnections()
	g2.SortConnections()

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

			break
		}

		if g.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Check difference in weights for shared genes
			nShared += 1
			W += math.Abs(g.Connections[i1].Weight - g2.Connections[i2].Weight)

			i1 += 1
			i2 += 1
		} else if g.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i1 += 1
		} else if g.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i2 += 1
		}
	}

	if nShared > 0 {
		W /= nShared
	}

	// Should never be negative, but take absolute value just in case
	distance := math.Abs(c1*nExcess/N + c2*nDisjoint/N + c3*W)
	return distance
}

type SortableGenes struct {
	genes []*EdgeGene
}

func (g SortableGenes) Len() int {
	return len(g.genes)
}

// Want to sort in descending order, so less + greater are swapped
func (g SortableGenes) Less(i, j int) bool {
	return g.genes[i].InnovationNumber < g.genes[j].InnovationNumber
}

func (g SortableGenes) Swap(i, j int) {
	g.genes[i], g.genes[j] = g.genes[j], g.genes[i]
}

func (g *Genome) SortConnections() {
	sg := SortableGenes{
		genes: g.Connections,
	}
	sort.Sort(sg)
}
