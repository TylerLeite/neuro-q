package ma

import (
	"bytes"
	"compress/flate"
	"math"
	"math/rand"
	"sort"
)

// Used for sorting by fitness
type SortableIndividuals []Individual

func (individuals SortableIndividuals) Len() int {
	return len(individuals)
}

// Want to sort in descending order, so less + greater are swapped
func (individuals SortableIndividuals) Less(i, j int) bool {
	return individuals[i].Fitness() > individuals[j].Fitness()
}

func (individuals SortableIndividuals) Swap(i, j int) {
	individuals[i], individuals[j] = individuals[j], individuals[i]
}

type Population struct {
	CullingPct             float64
	RecombinationPct       float64
	LocalSearchGenerations int
	MinimumEntropy         float64

	Size int

	Individuals []Individual
}

func NewPopulation(
	size int,
	seedIndividual *Individual,
	cullingPercent float64,
	recombinationPercent float64,
	minimumEntropy float64,
	localSearchGenerations int,
) *Population {
	p := Population{
		CullingPct:             cullingPercent,
		RecombinationPct:       recombinationPercent,
		LocalSearchGenerations: localSearchGenerations,
		MinimumEntropy:         minimumEntropy,
		Size:                   size,
		Individuals:            make([]Individual, size),
	}

	for i := range p.Individuals {
		p.Individuals[i] = (*seedIndividual).Copy()
	}

	return &p
}

// generate initial population
func (p *Population) Generate() {
	for i, individual := range p.Individuals {
		p.Individuals[i] = individual.Randomize()
	}
}

func (p *Population) Copy() *Population {
	newPopulation := Population{
		CullingPct:             p.CullingPct,
		RecombinationPct:       p.RecombinationPct,
		LocalSearchGenerations: p.LocalSearchGenerations,
		MinimumEntropy:         p.MinimumEntropy,
		Size:                   p.Size,
		Individuals:            make([]Individual, len(p.Individuals)),
	}

	for i, v := range p.Individuals {
		newPopulation.Individuals[i] = v.Copy()
	}

	return &newPopulation
}

// Local search
func (p *Population) LocalSearch() *Population {
	newPopulation := p.Copy()

	for i, individual := range p.Individuals {
		currentFitness := individual.Fitness()
		currentIndividual := individual
		for j := 0; j < p.LocalSearchGenerations; j += 1 {
			neighbor := individual.RandomNeighbor()
			neighborFitness := neighbor.Fitness()
			if neighbor.Fitness() > currentFitness {
				currentIndividual = neighbor
				currentFitness = neighborFitness
			}
		}

		// Lamarckian learning: the new individual replaces the old one
		newPopulation.Individuals[i] = currentIndividual
	}

	return newPopulation
}

// selection
func (p *Population) Selection() *Population {
	newPopulation := p.Copy()

	// Sort by fitness
	sort.Sort(SortableIndividuals(newPopulation.Individuals))

	// Cull the least fit individuals
	// TODO: also cull some randomly following a power law
	numberToCull := int(math.Floor(newPopulation.CullingPct * float64(newPopulation.Size)))
	newPopulation.Individuals = newPopulation.Individuals[:newPopulation.Size-numberToCull]

	return newPopulation
}

// recombination
func (p *Population) Recombination() *Population {
	newPopulation := p.Copy()

	// Store children in a new slice during recombination so they aren't chosen as parents
	numberToRecombine := int(math.Floor(newPopulation.RecombinationPct * float64(newPopulation.Size)))
	children := make([]Individual, numberToRecombine)

	for i := 0; i < numberToRecombine; i += 1 {
		// Select two random parents
		r1 := rand.Intn(len(newPopulation.Individuals))
		r2 := r1
		for r2 == r1 {
			r2 = rand.Intn(len(newPopulation.Individuals))
		}

		// Baby make
		child := newPopulation.Individuals[r1].Crossover([]Individual{newPopulation.Individuals[r2]})
		children[i] = child
	}

	newPopulation.Individuals = append(newPopulation.Individuals, children...)
	return newPopulation
}

// stabilization
func (p *Population) Stabilization() *Population {
	newPopulation := p.Copy()

	// Make sure the population is the correct size
	extraIndividuals := len(newPopulation.Individuals) - newPopulation.Size

	if extraIndividuals < 0 {
		for i := 0; i < -extraIndividuals; i += 1 {
			randomIndividual := newPopulation.Individuals[i].Copy()
			newPopulation.Individuals = append(newPopulation.Individuals, randomIndividual.Randomize())
		}
	} else if extraIndividuals > 0 {
		sort.Sort(SortableIndividuals(newPopulation.Individuals))
		newPopulation.Individuals = newPopulation.Individuals[:newPopulation.Size]
	}

	return newPopulation
}

// check convergeance
func (p *Population) HasConverged() bool {
	corpus := ""
	for _, v := range p.Individuals {
		corpus += v.CodeString()
	}

	// Check entropy by compressing the population
	var buf bytes.Buffer
	w, _ := flate.NewWriter(&buf, 9)
	w.Write([]byte(corpus))
	w.Close()

	ratio := float64(len(buf.Bytes())) / float64(len(corpus))
	return ratio < p.MinimumEntropy
}
