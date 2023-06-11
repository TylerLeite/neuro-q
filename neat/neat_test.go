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
	fmt.Println("fitness start")
	o.(*Network).Draw("debug.bmp")
	fmt.Println(o.GeneticCode().ToString())
	n := o.(*Network)
	n.Compile()

	// Test 1: XOR
	bias := n.Nodes[n.DNA.SensorNodes[0]]
	inX := n.Nodes[n.DNA.SensorNodes[1]]
	inY := n.Nodes[n.DNA.SensorNodes[2]]

	out := n.Nodes[n.DNA.OutputNodes[0]]

	fitness := float64(8)

	for x := 0; x < 2; x += 1 {
		for y := 0; y < 2; y += 1 {
			bias.SetDefaultValue(float64(1))
			inX.SetDefaultValue(float64(x))
			inY.SetDefaultValue(float64(y))

			result := out.CalculateValue(nil)
			fitness -= math.Abs(result - float64(xor(x, y)))

			out.Reset()
		}
	}

	fmt.Println("fitness end")
	return fitness
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

		NewEdgeGene(8, 3, 0, NoMutation),
		NewEdgeGene(8, 4, 0, NoMutation),
		NewEdgeGene(9, 4, 0, NoMutation),
		NewEdgeGene(9, 5, 0, NoMutation),
	}

	network := NewNetwork(seedGenome, nil)
	network.Draw("test.bmp")
}

func TestEvolution(t *testing.T) {
	ResetInnovationHistory()

	seedGenome := NewGenome(3, 1)
	seedNetwork := NewNetwork(seedGenome, nil)

	p := ma.NewPopulation(100, ma.Organism(seedNetwork), NeatFitness)
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

	// run for G generations
	const G int = 50
	for i := 0; i < G; i += 1 {
		fmt.Printf("New generation, %d/%d\n", i+1, G)

		speciesLengths := make([]int, len(p.Species))
		for j, species := range p.Species {
			speciesLengths[j] = len(species.Members)
		}
		fmt.Printf("%d species, lengths: %v\n", len(p.Species), speciesLengths)

		p2 := p.Copy()
		for j, species := range p2.Species {
			fmt.Printf("Local search, %d/%d...\n", j+1, len(p2.Species))
			species.LocalSearch()
			fmt.Printf("Selection, %d/%d...\n", j+1, len(p2.Species))
			species.Selection()
			fmt.Printf("Recombination, %d/%d...\n", j+1, len(p2.Species))
			species.Recombination()
		}

		fmt.Printf("Separate into species, %d/%d..\n", i+1, G)
		p = p2.SeparateIntoSpecies()

		fmt.Println("Champion fitness per species:")
		for j, species := range p2.Species {
			champion := species.Champion()

			championNetwork := champion.(*Network)
			championNetwork.Draw(fmt.Sprintf("drawn/%d_%d.bmp", i, j))
			fmt.Printf("species #%d/%d: f=%.2g\n%s\n", j+1, len(p2.Species), p2.FitnessOf(champion), champion.(*Network).ToString())
		}

		fmt.Println()
	}
}
