package neat

import (
	"fmt"
	"math"

	"github.com/TylerLeite/neuro-q/log"
	"github.com/TylerLeite/neuro-q/ma"
)

func xor(x, y int) int {
	if x == 0 && y == 0 {
		return 0
	} else if x == 1 && y == 1 {
		return 0
	} else {
		return 1
	}
}

func XorFitness(o ma.Organism) float64 {
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
				log.Book(n.String(), log.DEBUG)
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
	fitness = 4 - fitness
	if correctAnswers == 4 {
		fitness = math.Inf(1)
	}

	return fitness
}

func XorVerify(o ma.Organism) int {
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
				log.Book(n.String(), log.DEBUG)
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

// TODO: This is mostly the same from generation to generation. Maybe add it as a function in epoch.go
func XorEvolution() error {
	ResetInnovationHistory()

	seedGenome := NewGenome(2, 1, true, -5, 5)
	seedGenome.MutationRatios = map[ma.MutationType]float64{
		MutationAddConnection: 0.05,
		MutationAddNode:       0.03,
		MutationMutateWeights: 0.92,
	}
	seedNetwork := NewNetwork(seedGenome, nil)

	p := ma.NewPopulation(ma.Organism(seedNetwork), XorFitness)
	seedNetwork.Population = p

	p.Size = 150
	p.DistanceThreshold = 2.0
	p.CullingPercent = 0.5
	p.RecombinationPercent = 0.8
	p.MinimumEntropy = 0.35
	p.LocalSearchGenerations = 8
	p.DropoffAge = 15

	p.Cs = []float64{1, 1, 0.4, 0}

	speciesTargetMin := 7
	speciesTargetMax := 13
	distanceThresholdEpsilon := 0.1

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

		fmt.Printf("New generation, %d/%d [%d species] dt=%.2g\n", i+1, G, len(p.Species), p.DistanceThreshold)

		championDNAs, championFitnesses, err := p.Epoch()

		if len(p.Species) > speciesTargetMax {
			p.DistanceThreshold += distanceThresholdEpsilon
		} else if len(p.Species) < speciesTargetMin {
			p.DistanceThreshold -= distanceThresholdEpsilon
		}

		if err != nil {
			return err
		}

		maxFitnessThisGeneration := 0.0
		for j, championFitness := range championFitnesses {
			if championFitness > maxFitnessThisGeneration {
				maxFitnessThisGeneration = championFitness
			}

			championNetwork := NewNetwork(championDNAs[j].(*Genome), p)
			championNetwork.Draw(fmt.Sprintf("neat/drawn/%d_%d.bmp", i, j))

			if math.IsInf(championFitness, 1) {
				fmt.Println("Found a fully verified network!")
				fmt.Println(championNetwork.String())
				fmt.Println(championNetwork.DNA.ToPretty())
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

	return nil
}
