package neat

import (
	"testing"
)

// Initialize population with no hidden layers
// TODO: for specifications with high input*output nodes, add some hidden layers instead
// Also can leave some inputs disconnected in such cases

// Champion of a species with size >= 5 is passed onto the next generation unchanged
// Probability of interspecies mating is 0.001

func TestDraw(t *testing.T) {
	ResetInnovationHistory()

	seedGenome := &Genome{}
	seedGenome.SensorNodes = []uint{0, 1, 2}
	seedGenome.OutputNodes = []uint{3, 4, 5}
	seedGenome.HiddenNodes = []uint{6, 7, 8, 9}

	seedGenome.Connections = []*EdgeGene{
		NewEdgeGene(0, 6, 0, NoMutation),
		NewEdgeGene(1, 6, 0, NoMutation),
		NewEdgeGene(1, 7, 0, NoMutation),
		NewEdgeGene(2, 7, 0, NoMutation),

		NewEdgeGene(6, 8, 0, NoMutation),
		NewEdgeGene(7, 8, 0, NoMutation),
		NewEdgeGene(7, 9, 0, NoMutation),

		NewEdgeGene(8, 6, 0, NoMutation),

		NewEdgeGene(8, 3, 0, NoMutation),
		NewEdgeGene(8, 4, 0, NoMutation),
		NewEdgeGene(9, 4, 0, NoMutation),
		NewEdgeGene(9, 5, 0, NoMutation),
	}

	for i, e := range seedGenome.Connections {
		e.Weight = float64(i)/6 - 1
	}

	network := NewNetwork(seedGenome, nil)
	network.Draw("test.bmp")
}

func TestMassiveDraw(t *testing.T) {
	genome := NewGenome(32, 32, true, -15, 15)

	for i := 0; i < 1000-64; i += 1 {
		genome.AddNode()
	}

	for i := 0; i < 256; i += 1 {
		genome.AddConnection(false)
	}

	network := NewNetwork(genome, nil)
	network.Draw("test_massive.bmp")
}

// func TestXor(t *testing.T) {

// 	var fitness float64
// 	var seedGenome *Genome

// 	manualWeights := []float64{
// 		0.12891183580278853,
// 		-0.6017346437838056,
// 		-1.0789994134152487,
// 		-0.758092908174699,
// 		3.108815121480327,
// 		2.010407441584877,
// 		4.961511425606139,
// 	}

// 	seedGenome = &Genome{
// 		SensorNodes: []uint{0, 1, 2},
// 		OutputNodes: []uint{3},
// 		HiddenNodes: []uint{4},
// 		Connections: []*EdgeGene{
// 			NewEdgeGene(0, 3, manualWeights[0], NoMutation),
// 			NewEdgeGene(1, 3, manualWeights[1], NoMutation),
// 			NewEdgeGene(2, 3, manualWeights[2], NoMutation),
// 			NewEdgeGene(0, 4, manualWeights[3], NoMutation),
// 			NewEdgeGene(1, 4, manualWeights[4], NoMutation),
// 			NewEdgeGene(2, 4, manualWeights[5], NoMutation),
// 			NewEdgeGene(4, 3, manualWeights[6], NoMutation),
// 		},

// 		UsesBias: true,
// 	}
// 	fmt.Println(seedGenome.String())

// 	network := NewNetwork(seedGenome, nil)
// 	fitness = XorFitness(ma.Organism(network))
// 	fmt.Printf("Fitness of manual xor solution: %.2g\n", fitness)
// }
