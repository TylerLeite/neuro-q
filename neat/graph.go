package neat

import "math/rand"

// TODO: node genes, will they always just work out if they go sensor -> output -> hidden?
// TODO: bias flag for nodes
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

func (g *Graph) AddConnectionMutation(mutations map[string]uint, feedForward bool) {
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

	connection := NewConnection(uint(r1), uint(r2), rand.Float64(), mutations)
	g.Connections = append(g.Connections, connection)
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
