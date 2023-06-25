package ge

import (
	"fmt"
	"math"

	"github.com/TylerLeite/neuro-q/config"
	"github.com/TylerLeite/neuro-q/ma"
)

func nodeValue(node *DerivationTree, x, y float64) float64 {
	switch node.SymbolNames[node.Value] {
	case "0":
		return 0
	case "1":
		return 1
	case "x":
		return x
	case "y":
		return y
	case "sqrt":
		// sqrt of absolute value
		return math.Sqrt(math.Abs(nodeValue(node.Children[0], x, y)))
	case "+":
		return nodeValue(node.Children[0], x, y) + nodeValue(node.Children[1], x, y)
	case "-":
		return nodeValue(node.Children[0], x, y) - nodeValue(node.Children[1], x, y)
	case "*":
		return nodeValue(node.Children[0], x, y) * nodeValue(node.Children[1], x, y)
	case "/":
		// don't divide by 0
		return nodeValue(node.Children[0], x, y) / math.Max(1, nodeValue(node.Children[1], x, y))
	default:
		return 1
	}
}

func SmallValueApproximationEvolution() {
	targetFunc := func(x, y float64) float64 {
		return math.Sin((x + y) / 2)
	}

	fitnessOf := func(o ma.Organism) float64 {
		program := o.(*Program)
		program.Compile()

		fitness := float64(0)

		for x := float64(-1); x <= 1; x += 0.1 {
			for y := float64(-1); y <= 1; y += 0.1 {
				target := targetFunc(x, y)
				value := nodeValue(program.SyntaxTree, x, y)

				// fmt.Printf("t(%.2g, %.2g) = %.4g\n", x, y, target)
				// fmt.Printf("f(%.2g, %.2g) = %.4g\n", x, y, value)
				// fmt.Println(program.SyntaxTree.String())

				// TODO: scale with size of x + y?
				fitness -= (target - value) * (target - value)
			}
		}

		return fitness
	}

	rules, symbolNames := LoadRulesFromFile("./ge/grammars/polynomial.grmr")
	fmt.Println("Rules:\n", rules)
	fmt.Println("Symbols:\n", symbolNames)

	seedGenome := NewGenome(make([]byte, 8))
	seedProgram := NewProgram(seedGenome, rules, symbolNames)

	popCfg := config.PopulationDefault()
	popCfg.Size = 512
	popCfg.MaxEpochs = 1000
	popCfg.RecombinationPercent = 0.75
	popCfg.LocalSearchGenerations = 16
	popCfg.DistanceThreshold = 1
	popCfg.DropoffAge = 15
	popCfg.SharingFunctionConstants = []float64{1}

	p := ma.NewPopulation(ma.Organism(seedProgram), fitnessOf)

	p.Size = popCfg.Size
	p.DistanceThreshold = popCfg.DistanceThreshold
	p.DropoffAge = popCfg.DropoffAge

	p.RecombinationPercent = popCfg.RecombinationPercent
	p.LocalSearchGenerations = popCfg.LocalSearchGenerations

	p.CullingPercent = popCfg.CullingPercent
	p.MinimumEntropy = popCfg.MinimumEntropy
	p.Cs = popCfg.SharingFunctionConstants

	manualGenome := NewGenome([]byte{4, 0, 2, 0, 0, 2, 2, 0, 1, 0, 1, 0, 1, 1, 1, 1})
	manualProgram := NewProgram(manualGenome, rules, symbolNames)
	manualProgram.Compile()

	fmt.Println("Generating...")
	p.Generate()

	G := popCfg.MaxEpochs
	maxFitness := math.Inf(-1)
	for i := 0; i < G; i += 1 {
		fmt.Printf("New generation, %d/%d [%d species] dt=%.2g\n", i+1, G, len(p.Species), p.DistanceThreshold)

		p.Epoch()

		foundOptimalSolution := false
		fmt.Println("Champion Genomes:")
		for j, species := range p.Species {
			championProgram := species.Champion().(*Program)
			fitness := fitnessOf(championProgram)
			fmt.Printf("\t%d (fitness = %.4g): %s\n", j+1, fitness, championProgram.GeneticCode().ToString())

			if fitness > maxFitness {
				maxFitness = fitness
				fmt.Println("New max fitness!")
				fmt.Println(championProgram.SyntaxTree.String())
			} else if fitness == 0 {
				fmt.Println(championProgram.SyntaxTree.String())
			}

			if fitness == 0 {
				foundOptimalSolution = true
			}
		}

		if foundOptimalSolution {
			fmt.Println("Found an optimal solution, ending early")
			break
		}
	}
}
