package ge

import (
	"fmt"
	"math"
	"testing"

	"github.com/TylerLeite/neuro-q/config"
	"github.com/TylerLeite/neuro-q/ma"
)

func TestGrammar(t *testing.T) {
	return
	rules, symbolNames := LoadRulesFromFile("./grammars/clifford_plus_t.grmr")

	fmt.Println(rules)
	fmt.Println(symbolNames)

	a := RootNodeFromRules(rules, symbolNames)
	// op
	b := RootNodeFromRules(rules, symbolNames)
	b.Value = 0x03
	a.AppendChild(b)
	// CNOT
	e := RootNodeFromRules(rules, symbolNames)
	e.Value = 0x0A
	b.AppendChild(e)

	// K
	c := RootNodeFromRules(rules, symbolNames)
	a.AppendChild(c)
	// op
	j := RootNodeFromRules(rules, symbolNames)
	j.Value = 0x03
	c.AppendChild(j)
	// CNOT
	o := RootNodeFromRules(rules, symbolNames)
	o.Value = 0x0A
	j.AppendChild(o)

	// K
	k := RootNodeFromRules(rules, symbolNames)
	c.AppendChild(k)
	// q
	f := RootNodeFromRules(rules, symbolNames)
	f.Value = 0x09
	k.AppendChild(f)
	// q0
	// h := RootNodeFromRules(rules, symbolNames)
	// h.Value = 0x0B
	// f.AppendChild(h)

	// K
	l := RootNodeFromRules(rules, symbolNames)
	c.AppendChild(l)
	// q
	m := RootNodeFromRules(rules, symbolNames)
	m.Value = 0x09
	l.AppendChild(m)
	// q1
	n := RootNodeFromRules(rules, symbolNames)
	n.Value = 0x0D
	m.AppendChild(n)

	// K
	d := RootNodeFromRules(rules, symbolNames)
	a.AppendChild(d)
	// const
	g := RootNodeFromRules(rules, symbolNames)
	g.Value = 0x08
	d.AppendChild(g)
	// 1-ket
	i := RootNodeFromRules(rules, symbolNames)
	i.Value = 0x15
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
		return math.Sqrt(x*x + y*y)
	}

	fitnessOf := func(o ma.Organism) float64 {
		program := o.(*Program)
		program.Compile()

		fitness := float64(0)

		for x := float64(0); x < 100; x += 1 {
			for y := float64(0); y < 100; y += 1 {
				target := targetFunc(x, y)
				value := nodeValue(program.SyntaxTree, x, y)

				// TODO: scale with size of x + y
				fitness -= math.Sqrt(target*target + value*value)
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
	popCfg.MaxEpochs = 100
	popCfg.DistanceThreshold = 3
	popCfg.DistanceThresholdEpsilon = 1
	popCfg.TargetMinSpecies = 7
	popCfg.TargetMaxSpecies = 15
	popCfg.RecombinationPercent = 0.75
	popCfg.LocalSearchGenerations = 8

	p := ma.NewPopulation(ma.Organism(seedProgram), fitnessOf)
	// seedProgram.Population = p

	p.Size = popCfg.Size
	p.DistanceThreshold = popCfg.DistanceThreshold
	p.CullingPercent = popCfg.CullingPercent
	p.RecombinationPercent = popCfg.RecombinationPercent
	p.MinimumEntropy = popCfg.MinimumEntropy
	p.LocalSearchGenerations = popCfg.LocalSearchGenerations
	p.DropoffAge = popCfg.DropoffAge

	p.Cs = popCfg.SharingFunctionConstants // TODO: use copy()?

	speciesTargetMin := popCfg.TargetMinSpecies
	speciesTargetMax := popCfg.TargetMaxSpecies
	distanceThresholdEpsilon := popCfg.DistanceThresholdEpsilon

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

		fmt.Println("Champion Genomes:")
		for j, species := range p.Species {
			championProgram := species.Champion().(*Program)
			fitness := fitnessOf(championProgram)
			fmt.Printf("\t%d (fitness = %.4g): %s\n", j+1, fitness, championProgram.GeneticCode().ToString())
		}
	}

	a := RootNodeFromRules(rules, symbolNames)
	fmt.Println(a)
}
