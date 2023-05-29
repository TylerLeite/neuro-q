package neat

import (
	"math"
	"math/rand"
)

// TODO: also pass map of changes to innovation numbers, reset map after each generation
func Crossover(g1, g2 *Graph) *Graph {
	// Need to know which parent is more fit for inheriting excess and disjoint genes
	moreFitParent := g1
	if g1.Fitness() < g2.Fitness() {
		moreFitParent = g2
	}

	g := Graph{
		Connections: make([]*Connection, 0),
		SensorNodes: make([]uint, 0),
		HiddenNodes: make([]uint, 0),
		OutputNodes: make([]uint, 0),
	}

	// Line up genes by innovation number
	var i1, i2 int
	for {
		// Check if we are in excess node territory
		if i1 >= len(g1.Connections) {
			if i2 >= len(g2.Connections) {
				break
			} else if g1 == moreFitParent {
				// Inherit excess genes from the more fit parent
				for ; i1 < len(g1.Connections); i1 += 1 {
					g.Connections = append(g.Connections, g1.Connections[i1])
				}
			}
		} else if i2 >= len(g2.Connections) && g2 == moreFitParent {
			// Inherit excess genes from the more fit parent
			for ; i2 < len(g2.Connections); i2 += 1 {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}
		}

		if g1.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Inherit a gene randomly when there is an innovation number  match
			if rand.Intn(2) == 0 {
				g.Connections = append(g.Connections, g1.Connections[i1])
			} else {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}

			// If either gene is disabled, there is a 75% chance the inherited gene is disabled as well

			i1 += 1
			i2 += 1
		} else if g1.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			// Inherit disjoint genes from the more fit parent
			if g1 == moreFitParent {
				g.Connections = append(g.Connections, g1.Connections[i1])
			}

			i1 += 1
		} else if g1.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			if g2 == moreFitParent {
				g.Connections = append(g.Connections, g2.Connections[i2])
			}

			i2 += 1
		}
	}

	return nil
}

func GetGenomeDistance(g1, g2 *Graph, c1, c2, c3 float64) float64 {
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
		if i1 >= len(g1.Connections) {
			if i2 < len(g2.Connections)-1 {
				nExcess = float64(len(g2.Connections) - i2 - 1)
			}

			break
		} else if i2 >= len(g2.Connections) {
			if i1 < len(g1.Connections)-1 {
				nExcess = float64(len(g1.Connections) - i1 - 1)
			}
		}

		if g1.Connections[i1].InnovationNumber == g2.Connections[i2].InnovationNumber {
			// Check difference in weights for shared genes
			nShared += 1
			W += math.Abs(g1.Connections[i1].Weight - g2.Connections[i2].Weight)
		} else if g1.Connections[i1].InnovationNumber < g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i1 += 1
		} else if g1.Connections[i1].InnovationNumber > g2.Connections[i2].InnovationNumber {
			nDisjoint += 1
			i2 += 1
		}
	}

	W /= nShared

	return c1*nExcess/N + c2*nDisjoint/N + c3*W
}

func SeparateIntoSpecies(currentPopulation Population, previousSpecies []Species, threshold float64) []Species {
	representatives := make([]*Graph, len(previousSpecies))
	thisGenerationSpecies := make([]Species, len(previousSpecies))

	for _, currentIndividual := range currentPopulation {
		foundASpecies := false

		// TODO: also need to be able to include newly created species in the loop
		for i, species := range previousSpecies {
			if representatives[i] == nil {
				// First pick a representative for the species if one doesn't exist
				representatives[i] = species.GetRepresentative()
			}

			// Place this individual into the first species where it fits
			// TODO: figure out actual values for these constants
			// pop 150 -> c3 = 0.4, threshold = 3.0 | pop 1000 -> c3 = 3.0, threshold = 4.0
			d := GetGenomeDistance(currentIndividual, representatives[i], 1, 1, 0.4)
			if d < threshold {
				thisGenerationSpecies[i].Members = append(thisGenerationSpecies[i].Members, currentIndividual)
				foundASpecies = true
				break
			}
		}

		if !foundASpecies {
			// Make a new species with this individual as the representative
			thisGenerationSpecies = append(thisGenerationSpecies, *NewSpecies())
			thisGenerationSpecies[len(thisGenerationSpecies)-1].Members[0] = currentIndividual
			representatives = append(representatives, currentIndividual)
		}
	}

	// Clean up empty species
	return thisGenerationSpecies
}

// Initialize population with no hidden layers
func InitializePopulation(inNodes uint, outNodes uint, size int) (Population, []Species) {
	firstGenerationMutations := make(map[string]uint)

	population := make(Population, size)

	for i := 0; i < size; i += 1 {
		g := Graph{}
		g.Connections = make([]*Connection, 0)
		g.HiddenNodes = make([]uint, 0)
		g.SensorNodes = make([]uint, 0)
		g.OutputNodes = make([]uint, 0)

		// Make sure all nodes are connected at least once
		nodesConnected := make([]bool, outNodes)
		for s := uint(0); s < inNodes; s += 1 {
			g.SensorNodes = append(g.SensorNodes, s)

			outNode := uint(rand.Intn(int(outNodes)))
			nodesConnected[outNode] = true

			c := NewConnection(s, outNode, rand.Float64()-0.5, firstGenerationMutations)
			g.Connections = append(g.Connections, c)
		}

		for o := uint(0); o < outNodes; o += 1 {
			// No need to double up
			if nodesConnected[o] {
				continue
			}
			g.OutputNodes = append(g.OutputNodes, o+inNodes)
		}

		population[i] = nil
	}

	// TODO: threshold from config instead of hardcoded
	species := SeparateIntoSpecies(population, make([]Species, 0), 3.0)
	return population, species
}

func NextGeneration(p Population, species *Species) (Population, []Species) {
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

// transfer function = p(x) = 1 / (1 + e^-4.9x)
