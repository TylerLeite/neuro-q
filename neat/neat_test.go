package neat

import (
	"fmt"
	"math"
	"testing"

	"github.com/TylerLeite/neuro-q/log"
	"github.com/TylerLeite/neuro-q/ma"
)

// Initialize population with no hidden layers
// TODO: for specifications with high input*output nodes, add some hidden layers instead
// Also can leave some inputs disconnected in such cases

// Champion of a species with size >= 5 is passed onto the next generation unchanged

// Genomes have an 80% chance of mutating weights
// if mutated, each connection has a 90% chance of being uniformly perturbed, 10% chance of getting a random value

// 25% of offspring each generation are from mutation without crossover

// Probability of interspecies mating is 0.001

// chance of mutations:
// addNode = 0.03
// addConnection = 0.05

func xor(x, y int) int {
	if x == 0 && y == 0 {
		return 0
	} else if x == 1 && y == 1 {
		return 0
	} else {
		return 1
	}
}

func NeatFitness(o ma.Organism) float64 {
	n := o.(*Network)
	n.Compile()

	// Test 1: XOR
	var (
		bias *Node
		inX  *Node
		inY  *Node
	)

	for _, nodeI := range n.DNA.SensorNodes {
		node := n.Nodes[nodeI]
		if node.Label == "0" {
			bias = node
		} else if node.Label == "1" {
			inX = node
		} else {
			inY = node
		}
	}

	log.Book(fmt.Sprintf("Bias label: %s, X: %s, Y: %s\n", bias.Label, inX.Label, inY.Label), log.DEBUG, log.DEBUG_PROPAGATION)

	out := n.Nodes[n.DNA.OutputNodes[0]]

	fitness := float64(0)

	correctAnswers := 0

	for x := 0; x < 2; x += 1 {
		for y := 0; y < 2; y += 1 {
			inputValues := []float64{1, float64(x), float64(y)}
			n.Activate(inputValues, []*Node{bias, inX, inY}, []*Node{out})

			result := out.Value()
			if math.IsNaN(result) {
				log.Book(n.ToString(), log.DEBUG)
				panic("NaN network")
			}
			log.Book(fmt.Sprintf("Inputs were: %v and output was: %0.2g\n", inputValues, out.value), log.DEBUG, log.DEBUG_PROPAGATION)

			target := float64(xor(x, y))

			if target == 1 && result >= 0.5 || target == 0 && result < 0.5 {
				correctAnswers += 1
			}

			fitness += (result - target) * (result - target)
		}
	}

	// Take reciprocal of square, want maximum value at minimum difference between result + target
	fitness = 16 / (1 + fitness)
	if correctAnswers == 4 {
		fitness = math.Inf(1)
	}

	return fitness
}

func NeatVerify(o ma.Organism) int {
	n := o.(*Network)
	n.Compile()

	// Test 1: XOR
	var (
		bias *Node
		inX  *Node
		inY  *Node
	)

	for _, nodeI := range n.DNA.SensorNodes {
		node := n.Nodes[nodeI]
		if node.Label == "0" {
			bias = node
		} else if node.Label == "1" {
			inX = node
		} else {
			inY = node
		}
	}

	out := n.Nodes[n.DNA.OutputNodes[0]]
	testsPassed := 0
	for x := 0; x < 2; x += 1 {
		for y := 0; y < 2; y += 1 {
			inputValues := []float64{1, float64(x), float64(y)}
			n.Activate(inputValues, []*Node{bias, inX, inY}, []*Node{out})

			outValue := out.Value()
			if math.IsNaN(outValue) {
				log.Book(n.ToString(), log.DEBUG)
				panic("NaN network")
			}

			target := xor(x, y)
			result := 0
			if outValue >= 0.5 {
				result = 1
			}

			if result == target {
				testsPassed += 1
			}
			log.Book(fmt.Sprintf("Inputs were: %v and output was: %dg\n", inputValues, result), log.DEBUG, log.DEBUG_PROPAGATION)
		}
	}

	return testsPassed
}

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
	genome := NewGenome(32, 32, true)

	for i := 0; i < 1000-64; i += 1 {
		genome.AddNode()
	}

	for i := 0; i < 256; i += 1 {
		genome.AddConnection(false)
	}

	network := NewNetwork(genome, nil)
	network.Draw("test_massive.bmp")
}

func TestXor(t *testing.T) {

	var fitness float64
	var seedGenome *Genome

	manualWeights := []float64{
		0.12891183580278853,
		-0.6017346437838056,
		-1.0789994134152487,
		-0.758092908174699,
		3.108815121480327,
		2.010407441584877,
		4.961511425606139,
	}

	seedGenome = &Genome{
		SensorNodes: []uint{0, 1, 2},
		OutputNodes: []uint{3},
		HiddenNodes: []uint{4},
		Connections: []*EdgeGene{
			NewEdgeGene(0, 3, manualWeights[0], NoMutation),
			NewEdgeGene(1, 3, manualWeights[1], NoMutation),
			NewEdgeGene(2, 3, manualWeights[2], NoMutation),
			NewEdgeGene(0, 4, manualWeights[3], NoMutation),
			NewEdgeGene(1, 4, manualWeights[4], NoMutation),
			NewEdgeGene(2, 4, manualWeights[5], NoMutation),
			NewEdgeGene(4, 3, manualWeights[6], NoMutation),
		},

		UsesBias: true,
	}
	fmt.Println(seedGenome.ToString())

	network := NewNetwork(seedGenome, nil)
	fitness = NeatFitness(ma.Organism(network))
	fmt.Printf("Fitness of manual xor solution: %.2g\n", fitness)
}

// TODO: This is mostly the same from generation to generation. Maybe add it as a function in epoch.go
func TestEvolution(t *testing.T) {
	ResetInnovationHistory()

	seedGenome := NewGenome(2, 1, true)
	seedNetwork := NewNetwork(seedGenome, nil)

	p := ma.NewPopulation(ma.Organism(seedNetwork), NeatFitness)
	seedNetwork.Population = p

	p.Size = 100
	p.DistanceThreshold = 1.5
	p.CullingPercent = 0.5
	p.RecombinationPercent = 0.8
	p.MinimumEntropy = 0.35
	p.LocalSearchGenerations = 16
	p.DropoffAge = 20

	p.Cs = []float64{10, 1, 0.4, 0}

	// TODO: debug log flag for console output

	fmt.Printf("Generate...\n")
	p.Generate()

	// run for at most G generations
	const G int = 1000
	maxFitnessHistory := make([]float64, G)
	fullyVerified := false
	i := 0
	for !fullyVerified {
		i += 1
		if i > G {
			fmt.Println("Reached generation limit")
			break
		}

		fmt.Printf("New generation, %d/%d [%d species]\n", i+1, G, len(p.Species))

		championDNAs, championFitnesses := p.Epoch()

		maxFitnessThisGeneration := 0.0
		for j, championFitness := range championFitnesses {
			if championFitness > maxFitnessThisGeneration {
				maxFitnessThisGeneration = championFitness
			}

			championNetwork := NewNetwork(championDNAs[j].(*Genome), p)
			championNetwork.Draw(fmt.Sprintf("drawn/%d_%d.bmp", i, j))

			if math.IsInf(championFitness, 1) {
				fmt.Println("Found a fully verified network!")
				fullyVerified = true
			}
		}

		if i > 0 {
			maxFitnessHistory[i-1] = maxFitnessThisGeneration
		}
	}

	fmt.Println("Fitness history:")
	for j, fitness := range maxFitnessHistory[:i] {
		fmt.Printf("Generation %d: %g\n", j+1, fitness)
	}
}
