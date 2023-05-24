package ma

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

type EvolvingString string

var Alphabet = [26]string{
	"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "a", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m",
}

func (e EvolvingString) Copy() Individual {
	copyStr := strings.Clone(string(e))
	out := Individual(EvolvingString(copyStr))
	return out
}

func (e EvolvingString) Randomize() Individual {
	out := Alphabet[rand.Intn(26)]
	cont := 0
	for cont < 7 {
		out = out + Alphabet[rand.Intn(26)]
		cont = rand.Intn(8)
	}

	return Individual(EvolvingString(out))
}

func (e EvolvingString) RandomNeighbor() Individual {
	// Possible mutations: add a letter, change a letter, remove a letter

	var s string
	switch mutationType := rand.Intn(3); mutationType {
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

	neighbor := Individual(EvolvingString(s))
	return neighbor
}

func (e EvolvingString) Fitness() float64 {
	var fitness float64

	// Length gives up to 10 fitness point, based on how close it is to 8 characters
	lenFactor := float64(len(e) - 7)
	fitness = 10 - (lenFactor * lenFactor)

	// Want to favor letters later in the alphabet. Score up to 10 points (all z)
	var averageValue float64 = 0
	for _, v := range e {
		averageValue += float64(v - 'a')
	}
	averageValue = averageValue / float64(len(e)*25)
	fitness += 10 * averageValue

	// Want to favor variation in letters. Score up to 10 points (all different)
	var averageCount float64 = 0
	for _, v := range e {
		averageCount += float64(strings.Count(string(e), string(v)))
	}
	averageCount = averageCount / float64(len(e))

	fitness += 10 * (1 - (averageCount / float64(len(e))))

	return fitness
}

func (e EvolvingString) Crossover(others []Individual) Individual {
	// In this case, only look at the first element of others
	other := others[0].(EvolvingString)

	// Take some from the start of mom, some from the end of dad, mash em together
	cutFromEnd := rand.Intn(len(e))
	cutFromStart := rand.Intn(len(other))

	child := Individual(e[cutFromEnd:] + other[:cutFromStart])
	return child
}

func (e EvolvingString) CodeString() string {
	return string(e)
}

// TODO: Better tests (convergeance, each generation improving, etc.)
func TestEvolution(t *testing.T) {
	seed := Individual(EvolvingString("abcdef"))

	p1 := NewPopulation(
		100,
		&seed,
		0.5,
		1,
		0.35,
		16,
	)

	p1.Generate()

	for i := 0; i < 100; i += 1 {
		p2 := p1.LocalSearch()
		p3 := p2.Selection()
		p4 := p3.Recombination()
		p5 := p4.Stabilization()

		p1 = p5
		if p1.HasConverged() {
			break
		}
	}

	var averageFitness float64 = 0
	maxFitness := p1.Individuals[0].Fitness()
	maxJ := 0
	for j, v := range p1.Individuals {
		f := v.Fitness()
		averageFitness += f
		if f > maxFitness {
			maxFitness = f
			maxJ = j
		}
	}
	fmt.Println(p1.Individuals[maxJ], maxFitness, "(", averageFitness/float64(len(p1.Individuals)), ")")

}
