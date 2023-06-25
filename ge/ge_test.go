package ge

import (
	"fmt"
	"math"
	"testing"

	"github.com/TylerLeite/neuro-q/config"
	"github.com/TylerLeite/neuro-q/ma"
)

func TestGrammar(t *testing.T) {
	rules, symbolNames := LoadRulesFromFile("./grammars/clifford_plus_t.grmr")

	fmt.Println(rules)
	fmt.Println(symbolNames)

	a := RootNodeFromRules(rules, symbolNames)
	a.Value = 0x01
	// op
	b := RootNodeFromRules(rules, symbolNames)
	b.Value = 0x02
	a.AppendChild(b)
	// CNOT
	e := RootNodeFromRules(rules, symbolNames)
	e.Value = 0x09
	b.AppendChild(e)

	// K
	c := RootNodeFromRules(rules, symbolNames)
	c.Value = 0x03
	e.AppendChild(c)
	// op
	j := RootNodeFromRules(rules, symbolNames)
	j.Value = 0x02
	c.AppendChild(j)
	// CNOT
	o := RootNodeFromRules(rules, symbolNames)
	o.Value = 0x09
	j.AppendChild(o)

	// K
	k := RootNodeFromRules(rules, symbolNames)
	k.Value = 0x03
	o.AppendChild(k)
	// q
	f := RootNodeFromRules(rules, symbolNames)
	f.Value = 0x05
	k.AppendChild(f)
	// q0
	// h := RootNodeFromRules(rules, symbolNames)
	// h.Value = 0x0B
	// f.AppendChild(h)

	// K
	l := RootNodeFromRules(rules, symbolNames)
	l.Value = 0x03
	o.AppendChild(l)
	// q
	m := RootNodeFromRules(rules, symbolNames)
	m.Value = 0x05
	l.AppendChild(m)
	// q1
	n := RootNodeFromRules(rules, symbolNames)
	n.Value = 0x0C
	m.AppendChild(n)

	// K
	d := RootNodeFromRules(rules, symbolNames)
	d.Value = 0x03
	e.AppendChild(d)
	// const
	g := RootNodeFromRules(rules, symbolNames)
	g.Value = 0x04
	d.AppendChild(g)
	// 1-ket
	i := RootNodeFromRules(rules, symbolNames)
	i.Value = 0x14
	g.AppendChild(i)

	fmt.Println(a)
	a.Finish()
	fmt.Println(a)

	s := a.ToSyntaxTree()
	fmt.Println(s)
}

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

func TestEvolution(t *testing.T) {
	targetFunc := func(x, y float64) float64 {
		return math.Sqrt(x*x + y)
	}

	fitnessOf := func(o ma.Organism) float64 {
		program := o.(*Program)
		program.Compile()

		fitness := float64(0)

		for x := float64(0); x < 4; x += 1 {
			for y := float64(0); y < 4; y += 1 {
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

	rules, symbolNames := LoadRulesFromFile("./grammars/polynomial.grmr")
	fmt.Println("Rules:\n", rules)
	fmt.Println("Symbols:\n", symbolNames)

	seedGenome := NewGenome(make([]byte, 8))
	seedProgram := NewProgram(seedGenome, rules, symbolNames)

	popCfg := config.PopulationDefault()
	popCfg.Size = 512
	popCfg.MaxEpochs = 1000
	popCfg.DistanceThreshold = 1
	popCfg.DistanceThresholdEpsilon = 0.1
	popCfg.TargetMinSpecies = 7
	popCfg.TargetMaxSpecies = 15
	popCfg.RecombinationPercent = 0.75
	popCfg.LocalSearchGenerations = 16

	p := ma.NewPopulation(ma.Organism(seedProgram), fitnessOf)

	p.Size = popCfg.Size
	p.DistanceThreshold = popCfg.DistanceThreshold
	p.RecombinationPercent = popCfg.RecombinationPercent
	p.LocalSearchGenerations = popCfg.LocalSearchGenerations

	speciesTargetMin := popCfg.TargetMinSpecies
	speciesTargetMax := popCfg.TargetMaxSpecies
	distanceThresholdEpsilon := popCfg.DistanceThresholdEpsilon

	p.CullingPercent = popCfg.CullingPercent
	p.MinimumEntropy = popCfg.MinimumEntropy
	p.DropoffAge = popCfg.DropoffAge
	p.Cs = popCfg.SharingFunctionConstants

	manualGenome := NewGenome([]byte{4, 0, 2, 0, 0, 2, 2, 0, 1, 0, 1, 0, 1, 1, 1, 1})
	manualProgram := NewProgram(manualGenome, rules, symbolNames)
	manualProgram.Compile()

	fmt.Println("Generating...")
	p.Generate()

	G := popCfg.MaxEpochs
	for i := 0; i < G; i += 1 {
		fmt.Printf("New generation, %d/%d [%d species] dt=%.2g\n", i+1, G, len(p.Species), p.DistanceThreshold)

		p.Epoch()

		// TODO: should this be a binary search?
		if len(p.Species) > speciesTargetMax {
			p.DistanceThreshold *= (1 + distanceThresholdEpsilon)
		} else if len(p.Species) < speciesTargetMin {
			p.DistanceThreshold *= (1 - distanceThresholdEpsilon)
		}

		foundOptimalSolution := false
		fmt.Println("Champion Genomes:")
		for j, species := range p.Species {
			championProgram := species.Champion().(*Program)
			fitness := fitnessOf(championProgram)
			fmt.Printf("\t%d (fitness = %.4g): %s\n", j+1, fitness, championProgram.GeneticCode().ToString())
			if fitness == 0 {
				fmt.Println(championProgram.SyntaxTree.String())
				foundOptimalSolution = true
			}
		}

		if foundOptimalSolution {
			fmt.Println("Found an optimal solution, ending early")
			break
		}
	}
}
