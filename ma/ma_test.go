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
type EvolvingString string

func (e EvolvingString) Copy() GeneticCode {
	copyStr := strings.Clone(string(e))
	out := GeneticCode(EvolvingString(copyStr))
	return out
}

func (e EvolvingString) Randomize() GeneticCode {
	out := Alphabet[rand.Intn(26)]
	cont := 0
	for cont < 7 {
		out = out + Alphabet[rand.Intn(26)]
		cont = rand.Intn(8)
	}

	return GeneticCode(EvolvingString(out))
}

const (
	AddLetter MutationType = iota
	RemoveLetter
	ChangeLetter
)

func (e EvolvingString) ListMutations() map[string]MutationType {
	out := make(map[string]MutationType)
	out["Add Letter"] = AddLetter
	out["Remove Letter"] = RemoveLetter
	out["Change Letter"] = ChangeLetter
	return out
}

func (e EvolvingString) Mutate(mutationType MutationType) GeneticCode {
	var s string

	switch mutationType {
	case 0:
		// Add
		position := rand.Intn(len(e) + 1)
		r := Alphabet[rand.Intn(26)]
		s = string(e)[:position] + r + string(e)[position:]
	case 1:
		// Remove
		position := rand.Intn(len(e))
		s = string(e[:position] + e[position+1:])
	case 2:
		// Chage
		position := rand.Intn(len(e))
		r := Alphabet[rand.Intn(26)]
		s = string(e)[:position] + r + string(e)[position+1:]
	default:
		return nil
	}

	neighbor := GeneticCode(EvolvingString(s))
	return neighbor
}

func (e EvolvingString) Crossover(others []GeneticCode) GeneticCode {
	// In this case, only look at the first element of others
	other := others[0].(EvolvingString)

	// Take some from the start of mom, some from the end of dad, mash em together
	cutFromEnd := rand.Intn(len(e))
	cutFromStart := rand.Intn(len(other))

	child := GeneticCode(e[cutFromEnd:] + other[:cutFromStart])
	return child
}

func (e EvolvingString) ToString() string {
	return string(e)
}

// Organism

type StringOrganism struct {
	Genome   EvolvingString
	compiled bool
}

func (s *StringOrganism) RandomNeighbor() Organism {
	mutationType := MutationType(rand.Intn(3))
	mutatedGenome := s.Genome.Mutate(mutationType)
	neighbor := &StringOrganism{
		Genome: mutatedGenome.(EvolvingString),
	}

	return Organism(neighbor)
}

func (s *StringOrganism) Copy() Organism {
	newOrganism := &StringOrganism{
		Genome:   s.Genome.Copy().(EvolvingString),
		compiled: s.IsCompiled(),
	}

	return Organism(newOrganism)
}

func (s *StringOrganism) NewFromGeneticCode(g GeneticCode) Organism {
	newOrganism := s.Copy()
	newOrganism.LoadGeneticCode(g)

	return Organism(newOrganism)
}

func (s *StringOrganism) LoadGeneticCode(g GeneticCode) {
	s.Genome = g.(EvolvingString)
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

func (s StringOrganism) Fitness() float64 {
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
	return string(e)
}

// TODO: Better tests (convergeance, each generation improving, etc.)
func TestEvolution(t *testing.T) {
	genome := EvolvingString("abcdef")
	organism := &StringOrganism{
		Genome: genome,
	}

	seed := Organism(organism)

	p1 := NewPopulation(100)
	p1.Seed = seed
	p1.CullingPercent = 0.5
	p1.RecombinationPercent = 1
	p1.MinimumEntropy = 0.35
	p1.LocalSearchGenerations = 16

	p1.Generate()

	for i := 0; i < 100; i += 1 {
		s1 := make([]*Species, 0)
		for _, species := range p1.Species {
			s1 = append(s1, species.LocalSearch())
		}

		s2 := make([]*Species, 0)
		for _, species := range s1 {
			s2 = append(s2, species.Selection())
		}

		s3 := make([]*Species, 0)
		for _, species := range s2 {
			s3 = append(s3, species.Recombination())
		}

		p2 := p1.Copy()
		p2.Species = s3
		p3 := p2.Stabilization()

		p1 = p3
		// if p1.HasConverged() {
		// 	break
		// }
	}

	var averageFitness float64 = 0
	maxFitness := p1.Species[0].Members[0].Fitness()
	maxJ := 0
	for j, v := range p1.Species[0].Members {
		f := v.Fitness()
		averageFitness += f
		if f > maxFitness {
			maxFitness = f
			maxJ = j
		}
	}

	fmt.Println(p1.Species[0].Members[maxJ], maxFitness, "(", averageFitness/float64(len(p1.Species[0].Members)), ")")
}
