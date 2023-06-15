package neat

import (
	"fmt"
	"math"
	"testing"

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

	Log(fmt.Sprintf("Bias label: %s, X: %s, Y: %s\n", bias.Label, inX.Label, inY.Label), DEBUG, DEBUG_PROPAGATION)

	out := n.Nodes[n.DNA.OutputNodes[0]]

	fitness := float64(0)

	for x := 0; x < 2; x += 1 {
		for y := 0; y < 2; y += 1 {
			inputValues := []float64{1, float64(x), float64(y)}
			n.Activate(inputValues, []*Node{bias, inX, inY}, []*Node{out})

			result := out.Value()
			if math.IsNaN(result) {
				Log(n.ToString(), DEBUG)
				panic("NaN network")
			}
			Log(fmt.Sprintf("Inputs were: %v and output was: %0.2g\n", inputValues, out.value), DEBUG, DEBUG_PROPAGATION)

			target := float64(xor(x, y))

			fitness += (result - target) * (result - target)
		}
	}

	// sizeScore := 1 / float64(len(n.Edges))

	// Take reciprocal of square, want maximum value at minimum difference between result + target
	fitness = 16 / (1 + fitness) // - sizeScore
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
				Log(n.ToString(), DEBUG)
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
			Log(fmt.Sprintf("Inputs were: %v and output was: %dg\n", inputValues, result), DEBUG, DEBUG_PROPAGATION)
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

func TestEvolution(t *testing.T) {
	ResetInnovationHistory()

	seedGenome := NewGenome(2, 1, true)
	seedNetwork := NewNetwork(seedGenome, nil)

	p := ma.NewPopulation(ma.Organism(seedNetwork), NeatFitness)
	seedNetwork.Population = p

	p.DistanceThreshold = 2
	p.CullingPercent = 0.5
	p.RecombinationPercent = 0.8
	p.MinimumEntropy = 0.35
	p.LocalSearchGenerations = 16

	p.C1 = 10
	p.C3 = 0.4

	fmt.Printf("Generate...\n")
	p.Generate()
	fmt.Printf("Separate into species...\n")
	p = p.SeparateIntoSpecies()

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

		fmt.Printf("New generation, %d/%d\n", i+1, G)

		speciesLengths := make([]int, len(p.Species))
		for j, species := range p.Species {
			speciesLengths[j] = len(species.Members)
		}
		fmt.Printf("%d species, lengths: %v\n", len(p.Species), speciesLengths)

		p2 := p.Copy()
		// TODO: sort by max fitness, kill off unfit species
		// p2.SortSpecies()
		for j := len(p2.Species) - 1; j >= 0; j -= 1 {
			species := p2.Species[j]
			species.UpdateFitnessHistory()
			fmt.Printf("Local search, %d/%d...\n", j+1, len(p2.Species))
			species.LocalSearch()
			if species.HasStagnated() {
				fmt.Printf("Stagnation, %d/%d...\n", j+1, len(p2.Species))
				p2.Species = append(p2.Species[:j], p2.Species[j+1:]...)
				continue
			}
			fmt.Printf("Selection, %d/%d...\n", j+1, len(p2.Species))
			species.Selection()
		}

		// Need another loop so recombination happens after all stagnant species are culled
		culledPopulationCount := float64(p2.CountMembers())
		for j, species := range p2.Species {
			fmt.Printf("Recombination, %d/%d...\n", j+1, len(p2.Species))
			species.Recombination(culledPopulationCount)
		}

		fmt.Printf("Separate into species, %d/%d..\n", i+1, G)
		p = p2.SeparateIntoSpecies()

		massExtinct := true
		for _, species := range p2.Species {
			if len(species.Members) > 0 {
				massExtinct = false
			}

			maxFitnessHistory = maxFitnessHistory[:i]
			for j, fitness := range maxFitnessHistory {
				fmt.Printf("Generation %d: %g\n", j+1, fitness)
			}
		}
		if massExtinct {
			panic("Science went too far")
		}

		maxFitnessThisGeneration := float64(0)
		fmt.Println("Champion fitness per species:")

		for j, species := range p2.Species {
			champion := species.Champion()

			championFitness := p2.FitnessOf(champion)
			championNetwork := champion.(*Network)
			championNetwork.Draw(fmt.Sprintf("drawn/%d_%d.bmp", i, j))
			fmt.Printf("species #%d/%d: f=%.2g\n%s\n%s\n", j+1, len(p2.Species), championFitness, championNetwork.DNA.ToString(), championNetwork.ToString())

			if championFitness > maxFitnessThisGeneration {
				maxFitnessThisGeneration = championFitness
			}

			correctAnswers := NeatVerify(championNetwork)
			if correctAnswers == 4 {
				fullyVerified = true
			} else {
				fmt.Printf("%d correct answers\n", correctAnswers)
			}
		}

		if i > 0 {
			maxFitnessHistory[i-1] = maxFitnessThisGeneration
		}
		fmt.Println()
	}

	if fullyVerified {
		fmt.Println("Found a fully verified network!")
	}

	fmt.Println("Fitness history:")
	for i, fitness := range maxFitnessHistory {
		fmt.Printf("Generation %d: %g\n", i+1, fitness)
	}
}
