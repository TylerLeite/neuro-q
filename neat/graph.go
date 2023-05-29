package neat

import "math/rand"

// TODO: node genes, will they always just work out if they go sensor -> output -> hidden?
type Graph struct {
	Connections []*Connection
	SensorNodes []uint
	HiddenNodes []uint
	OutputNodes []uint
}

func (g *Graph) Fitness() float64 {
	// TODO: fitness
	return 0
}

func (g *Graph) AdjustedFitness(s *Species) float64 {
	return g.Fitness() / float64(len(s.Members))
}

// NOTE: need to map innovation numbers within a generation to specific mutations
// -> keep a list of innovations that occurred in this generation

func (g *Graph) AddConnectionMutation(mutations map[string]uint) {
	//
}

func (g *Graph) AddNodeMutation(mutations map[string]uint) {
	// Randomly pick a connection to bifurcate
	randomGene := (g.Connections)[rand.Intn(len(g.Connections))]

	nextNode := g.HiddenNodes[len(g.HiddenNodes)-1] + 1

	// Create two new connection genes to fit this node into the network
	// -> new is weight 1, new -> is the old edge's weight
	new1 := NewConnection(randomGene.InNode, nextNode, 1, mutations)
	new2 := NewConnection(nextNode, randomGene.OutNode, randomGene.Weight, mutations)

	// Disable the old connection
	randomGene.Enabled = false

	g.Connections = append(g.Connections, []*Connection{new1, new2}...)
	g.HiddenNodes = append(g.HiddenNodes, nextNode)
}

type boolpair []bool

func (g *Graph) PopulateNodeSlices() {
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
}
