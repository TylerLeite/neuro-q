package ma

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

var Alphabet = [26]string{
	"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "a", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m",
}

// Genetic Code
type EvolvingString struct {
	Code string
}

func (e *EvolvingString) Copy() GeneticCode {
	copyStr := strings.Clone(e.Code)
	out := EvolvingString{
		Code: copyStr,
	}
	return GeneticCode(&out)
}

func (e *EvolvingString) Randomize() {
	out := Alphabet[rand.Intn(26)]
	cont := 0
	for cont < 7 {
		out = out + Alphabet[rand.Intn(26)]
		cont = rand.Intn(8)
	}

	e.Code = out
}

const (
	AddLetter MutationType = iota
	RemoveLetter
	ChangeLetter
)

func (e *EvolvingString) ListMutations() map[string]MutationType {
	out := make(map[string]MutationType)
	out["Add Letter"] = AddLetter
	out["Remove Letter"] = RemoveLetter
	out["Change Letter"] = ChangeLetter
	return out
}
func (e *EvolvingString) MutationOdds() map[MutationType]float64 {
	return make(map[MutationType]float64)
}

func (e *EvolvingString) Mutate(mutationType MutationType, args interface{}) {
	var s string

	switch mutationType {
	case 0:
		// Add
		position := rand.Intn(len(e.Code) + 1)
		r := Alphabet[rand.Intn(26)]
		s = e.Code[:position] + r + e.Code[position:]
	case 1:
		// Remove
		position := rand.Intn(len(e.Code))
		s = e.Code[:position] + e.Code[position+1:]
	case 2:
		// Chage
		position := rand.Intn(len(e.Code))
		r := Alphabet[rand.Intn(26)]
		s = e.Code[:position] + r + e.Code[position+1:]
	default:
		//
	}

	e.Code = s
}

// Want all strings to be in the same species -> always return 0 distance
func (e *EvolvingString) DistanceFrom(gc GeneticCode, cs ...float64) float64 {
	return 0
}

func (e *EvolvingString) ToString() string {
	return e.Code
}

// Organism

type StringOrganism struct {
	Genome   *EvolvingString
	compiled bool
}

func (s *StringOrganism) RandomNeighbor() Organism {
	mutationType := MutationType(rand.Intn(3))
	mutatedGenome := s.Genome.Copy()
	mutatedGenome.Mutate(mutationType, nil)
	neighbor := &StringOrganism{
		Genome: mutatedGenome.(*EvolvingString),
	}

	return Organism(neighbor)
}

func (s *StringOrganism) Copy() Organism {
	newOrganism := &StringOrganism{
		Genome:   s.Genome.Copy().(*EvolvingString),
		compiled: s.IsCompiled(),
	}

	return Organism(newOrganism)
}

func (s StringOrganism) Crossover(others []Organism) Organism {
	// In this case, only look at the first element of others
	other := others[0].(*StringOrganism)

	// Take some from the start of mom, some from the end of dad, mash em together
	cutFromEnd := rand.Intn(len(s.Genome.Code))
	cutFromStart := rand.Intn(len(other.Genome.Code))

	child := s.Genome.Code[cutFromEnd:] + other.Genome.Code[:cutFromStart]

	newGenome := &EvolvingString{
		Code: child,
	}
	return s.NewFromGeneticCode(GeneticCode(newGenome))
}

func (s *StringOrganism) NewFromGeneticCode(g GeneticCode) Organism {
	newOrganism := s.Copy()
	newOrganism.LoadGeneticCode(g)

	return Organism(newOrganism)
}

func (s *StringOrganism) LoadGeneticCode(g GeneticCode) {
	s.Genome = g.(*EvolvingString)
}

func (s *StringOrganism) GeneticCode() GeneticCode {
	return GeneticCode(s.Genome)
}

func (s *StringOrganism) Compile() error {
	s.compiled = true
	return nil
}

func (s *StringOrganism) IsCompiled() bool {
	return s.compiled
}

func StringOrganismFitness(o Organism) float64 {
	s := o.(*StringOrganism)
	var fitness float64

	dna := s.GeneticCode().ToString()

	// Length gives up to 10 fitness point, based on how close it is to 8 characters
	lenFactor := float64(len(dna) - 7)
	fitness = 10 - (lenFactor * lenFactor)

	// Want to favor letters later in the alphabet. Score up to 10 points (all z)
	var averageValue float64 = 0
	for _, v := range dna {
		averageValue += float64(v - 'a')
	}
	averageValue = averageValue / float64(len(dna)*25)
	fitness += 10 * averageValue

	// Want to favor variation in letters. Score up to 10 points (all different)
	var averageCount float64 = 0
	for _, v := range dna {
		averageCount += float64(strings.Count(string(dna), string(v)))
	}
	averageCount = averageCount / float64(len(dna))

	fitness += 10 * (1 - (averageCount / float64(len(dna))))

	return fitness
}

func (e EvolvingString) CodeString() string {
	return e.Code
}

// TODO: Better tests (convergeance, each generation improving, etc.)
func TestEvolution(t *testing.T) {
	genome := EvolvingString{
		Code: "abcdef",
	}
	organism := &StringOrganism{
		Genome: &genome,
	}

	seed := Organism(organism)

	p1 := NewPopulation(seed, StringOrganismFitness)
	p1.CullingPercent = 0.5
	p1.RecombinationPercent = 1
	p1.MinimumEntropy = 0.35
	p1.LocalSearchGenerations = 16

	p1.Generate()

	for i := 0; i < 100; i += 1 {
		p2 := p1.Copy()
		for _, species := range p2.Species {
			species.LocalSearch()
			species.Selection()
		}

		culledPopulationCount := float64(p2.CountMembers())
		for _, species := range p2.Species {
			species.Recombination(culledPopulationCount)
		}

		p1 = p2

		// TODO: test convergeance + stagnation checks
	}

	averageFitness := p1.Species[0].AverageFitness()
	champion := p1.Species[0].Champion().(*StringOrganism)
	championGenome := champion.Genome.ToString()
	championFitness := p1.FitnessOf(Organism(champion))

	fmt.Println(championGenome, championFitness, "(", averageFitness/float64(len(p1.Species[0].Members)), ")")
}
