package ge

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/TylerLeite/neuro-q/ma"
)

type Genome struct {
	Genes []byte
}

func NewGenome(g []byte) *Genome {
	return &Genome{
		Genes: g,
	}
}

func (g *Genome) Copy() ma.GeneticCode {
	newGenome := &Genome{
		Genes: make([]byte, len(g.Genes)),
	}

	copy(newGenome.Genes, g.Genes)

	return ma.GeneticCode(newGenome)
}

func (g *Genome) Randomize() {
	for i := 0; i < len(g.Genes); i += 1 {
		g.Genes[i] = byte(rand.Intn(256))
	}
}

const (
	NoMutation ma.MutationType = iota
	MutationDuplicateCodon
	MutationAppendCodon
	MutationRemoveCodon
	MutationMutateCodon
)

func (g *Genome) ListMutations() map[string]ma.MutationType {
	return map[string]ma.MutationType{
		"Duplicate": MutationDuplicateCodon,
		"Append":    MutationAppendCodon,
		"Remove":    MutationRemoveCodon,
		"Mutate":    MutationMutateCodon,
	}
}

func (g *Genome) MutationOdds() map[ma.MutationType]float64 {
	return map[ma.MutationType]float64{
		MutationDuplicateCodon: 0.3,
		MutationAppendCodon:    0.25,
		MutationRemoveCodon:    0.2,
		MutationMutateCodon:    0.25,
	}
}

func (g *Genome) Mutate(typ ma.MutationType, args interface{}) {
	randi := rand.Intn(len(g.Genes))
	randc := byte(rand.Intn(256))

	switch typ {
	case MutationDuplicateCodon:
		g.Genes = append(g.Genes[:randi], append([]byte{g.Genes[randi]}, g.Genes[randi:]...)...)
	case MutationAppendCodon:
		g.Genes = append(g.Genes, randc)
	case MutationRemoveCodon:
		if len(g.Genes) > 1 {
			g.Genes = append(g.Genes[:randi], g.Genes[randi+1:]...)
		}
	case MutationMutateCodon:
		g.Genes[randi] = randc
	default:
		return
	}
}

func (g *Genome) DistanceFrom(other ma.GeneticCode, args ...float64) float64 {
	return float64(len(g.Genes) - len(other.(*Genome).Genes))
}

func (g *Genome) String() string {
	return fmt.Sprintf("%v", g.Genes)
}

type Program struct {
	DNA *Genome

	Rules
	SymbolNames

	SyntaxTree *DerivationTree
}

func NewProgram(dna *Genome, r Rules, s SymbolNames) *Program {
	return &Program{
		DNA: dna,

		Rules:       r,
		SymbolNames: s,
	}
}

func (p *Program) Copy() ma.Organism {
	newProgram := NewProgram(p.DNA.Copy().(*Genome), p.Rules, p.SymbolNames)
	return ma.Organism(newProgram)
}

func (p *Program) RandomNeighbor() ma.Organism {
	neighbor := p.Copy()

	r := rand.Float64()
	mutation := MutationAppendCodon
	for k, v := range neighbor.GeneticCode().MutationOdds() {
		r -= v
		if r <= 0 {
			mutation = k
		}
	}

	neighbor.GeneticCode().Mutate(mutation, nil)
	return neighbor
}

func (p *Program) NewFromGeneticCode(dna ma.GeneticCode) ma.Organism {
	return ma.Organism(NewProgram(dna.(*Genome), p.Rules, p.SymbolNames))
}

func (p *Program) Crossover(others []ma.Organism) ma.Organism {
	parents := make([]*Program, len(others)+1)
	parents[0] = p
	for i, other := range others {
		parents[i+1] = other.(*Program)
	}
	codons := make([]byte, 0)

	scale := float64(0)
	crossoverPercents := make([]float64, len(parents))
	for i := 0; i < len(crossoverPercents); i += 1 {
		crossoverPercents[i] = rand.Float64()
		scale += crossoverPercents[i]
	}
	for i := 0; i < len(crossoverPercents); i += 1 {
		// normalize crossover points to sum to 1
		crossoverPercents[i] /= scale
	}

	cumulativeCrossoverPercents := make([]float64, len(crossoverPercents))
	for i := 0; i < len(cumulativeCrossoverPercents); i += 1 {
		for j := 0; j <= i; j += 1 {
			cumulativeCrossoverPercents[i] += crossoverPercents[j]
		}
	}

	for i, parent := range parents {
		crossoverStart := 0
		crossoverEnd := int(math.Min(math.Ceil(cumulativeCrossoverPercents[i]*float64(len(parent.DNA.Genes))), float64(len(parent.DNA.Genes))))
		if i > 0 {
			crossoverStart = int(math.Min(math.Ceil(cumulativeCrossoverPercents[i-1]*float64(len(parent.DNA.Genes))), float64(len(parent.DNA.Genes))))
		}
		codons = append(codons, parent.DNA.Genes[crossoverStart:crossoverEnd]...)
	}

	if len(codons) == 0 {
		fmt.Println(parents[0].DNA)
		fmt.Println(parents[1].DNA)
		fmt.Println(cumulativeCrossoverPercents)
		panic("uhoh :/\n")
	}

	child := NewProgram(NewGenome(codons), p.Rules, p.SymbolNames)
	return ma.Organism(child)
}

func (p *Program) GeneticCode() ma.GeneticCode {
	return ma.GeneticCode(p.DNA)
}

func (p *Program) LoadGeneticCode(dna ma.GeneticCode) {
	p.DNA = dna.(*Genome)
}

// TODO: support for identity detection, automatic optimization of generated function (plus output metric for how un-optimized a tree is)
func (p *Program) Compile() error {
	codons := p.DNA.Genes
	root := RootNodeFromRules(p.Rules, p.SymbolNames)

	i := 0
	for {
		if i >= len(codons) {
			// fmt.Println("Ran out of genetic code, finishing root automatically")
			root.Finish()
			break
		}

		// Find where in the tree to grow from
		node := root.GetFirstNonTerminalLeaf()
		if node == nil {
			// Have a complete program before reaching the end of the genome
			break
		}

		possibleRules := p.Rules[node.Value]

		children := possibleRules[int(codons[i])%len(possibleRules)]
		for _, symbol := range children {
			newNode := RootNodeFromRules(p.Rules, p.SymbolNames)
			newNode.Value = symbol
			node.AppendChild(newNode)
		}

		i += 1
	}

	p.SyntaxTree = root.ToSyntaxTree()
	// fmt.Println(p.SyntaxTree.String())
	return nil
}

func (p *Program) IsCompiled() bool {
	return p.SyntaxTree == nil
}
