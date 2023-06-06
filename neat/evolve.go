package neat

import (
	"github.com/TylerLeite/neuro-q/ma"
)

// Initialize population with no hidden layers
// TODO: for specifications with high input*output nodes, add some hidden layers instead
// Also can leave some inputs disconnected in such cases

func NextGeneration(p ma.Species, species *ma.Species) (*ma.Species, []ma.Species) {
	// ThisGenerationMutations := make(map[string]uint)

	// Max stagnant generations = 15 (species can no longer reproduce if its max fitness doesn't increase for 15 generations)
	return nil, nil
}

func Gogogo() {
	// listOfSpecies := make(Species, 0)
	// initalPopulation := InitializePopulation(32*32, 3, 150)
}

// Champion of a species with size >= 5 is passed onto the next generation unchanged

// Genomes have an 80% chance of mutating weights
// if mutated, each connection has a 90% chance of being uniformly perturbed, 10% chance of getting a random value

// 25% of offspring each generation are from mutation without crossover

// Probability of interspecies mating is 0.001

// chance of mutations:
// addNode = 0.03
// addConnection = 0.05

// activation function = p(x) = 1 / (1 + e^-4.9x)
