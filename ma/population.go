package ma

import (
	"fmt"
	"math"
	"sort"

	"github.com/TylerLeite/neuro-q/log"
)

type FitnessFunction func(Organism) float64

type Population struct {
	Species []*Species

	Size int

	Seed Organism

	FitnessOf FitnessFunction

	CullingPercent         float64
	RecombinationPercent   float64
	MinimumEntropy         float64
	LocalSearchGenerations int
	DropoffAge             int
	DistanceThreshold      float64

	// Constants for distance function
	C1 float64
	C2 float64
	C3 float64
}

func NewPopulation(seed Organism, fitnessFunction FitnessFunction) *Population {
	p := Population{
		Species:   make([]*Species, 0),
		Seed:      seed,
		FitnessOf: fitnessFunction,

		// Default config values
		Size:                   100,
		CullingPercent:         0.5,
		RecombinationPercent:   1,
		MinimumEntropy:         0.5,
		LocalSearchGenerations: 16,
		DropoffAge:             15,
		DistanceThreshold:      math.MaxFloat64,
		C1:                     1,
		C2:                     1,
		C3:                     0.4,
	}

	return &p
}

func (p *Population) Copy() *Population {
	newPopulation := p.CopyConfig()

	newPopulation.Species = make([]*Species, len(p.Species))
	for i, v := range p.Species {
		newPopulation.Species[i] = v.Copy(newPopulation)
	}

	return newPopulation
}

func (p *Population) CopyConfig() *Population {
	newPopulation := Population{
		Species: nil,
		Size:    p.Size,

		Seed:      p.Seed,
		FitnessOf: p.FitnessOf,

		CullingPercent:         p.CullingPercent,
		RecombinationPercent:   p.RecombinationPercent,
		MinimumEntropy:         p.MinimumEntropy,
		LocalSearchGenerations: p.LocalSearchGenerations,
		DropoffAge:             p.DropoffAge,
		DistanceThreshold:      p.DistanceThreshold,
		C1:                     p.C1,
		C2:                     p.C2,
		C3:                     p.C3,
	}

	return &newPopulation
}

func (p *Population) Members() []Organism {
	// Aggregate members across all species in population
	members := make([]Organism, 0)

	for _, species := range p.Species {
		members = append(members, species.Members...)
	}

	return members
}

func (p *Population) CountMembers() int {
	var total int

	for _, species := range p.Species {
		total += len(species.Members)
	}

	return total
}

// generate initial population
func (p *Population) Generate() {
	p.Species = make([]*Species, 1)
	p.Species[0] = NewSpecies(p)
	for i := 0; i < p.Size; i += 1 {
		newOrganism := p.Seed.Copy()
		newOrganism.GeneticCode().Randomize()

		p.Species[0].Members = append(p.Species[0].Members, newOrganism)
	}

	p.SeparateIntoSpecies()
}

// Output a new, speciated population
func (p *Population) SeparateIntoSpecies() {
	// newPopulation := p.CopyConfig()

	representatives := make([]Organism, len(p.Species))
	for i, species := range p.Species {
		if representatives[i] == nil {
			// First pick a representative for the species if one doesn't exist
			representatives[i] = species.RandomOrganism()
		}
	}

	nextGenSpecies := make([]*Species, len(p.Species)) // Need new species to line up with matching old species

	for _, currentIndividual := range p.Members() {
		foundASpecies := false

		for i, representative := range representatives {
			// Place this individual into the first species where it fits
			// TODO: figure out actual values for these constants
			d := currentIndividual.GeneticCode().DistanceFrom(representative.GeneticCode(), p.C1, p.C2, p.C3)
			if d < p.DistanceThreshold {
				if nextGenSpecies[i] == nil {
					nextGenSpecies[i] = NewSpecies(p)
				}

				nextGenSpecies[i].Members = append(nextGenSpecies[i].Members, currentIndividual.Copy())
				foundASpecies = true
				break
			}
		}

		if !foundASpecies {
			// Make a new species with this individual as the representative
			newSpecies := NewSpecies(p)
			newSpecies.Members = append(newSpecies.Members, currentIndividual.Copy())
			nextGenSpecies = append(nextGenSpecies, newSpecies)

			// Also need a representative for this species
			representatives = append(representatives, currentIndividual)
		}
	}

	// Clean up empty species
	for i := len(nextGenSpecies) - 1; i >= 0; i -= 1 {
		if nextGenSpecies[i] == nil {
			nextGenSpecies = append(nextGenSpecies[:i], nextGenSpecies[i+1:]...)
		}
	}

	p.Species = nextGenSpecies
}

func (p *Population) SortSpecies() []*Species {
	sortable := SortableSpecies(p.Species)
	sort.Sort(sortable)

	return []*Species(sortable)
}

// TODO: Make sure population is at its size. Either cull extra organisms or distribute new organisms among most fit species
func (p *Population) Stabilization() {
	//
}

// TODO: Wrap the output in a new type?
func (p *Population) Epoch() ([]GeneticCode, []float64) {
	speciesLengths := make([]int, len(p.Species))
	for i, species := range p.Species {
		speciesLengths[i] = len(species.Members)
	}
	log.Book(fmt.Sprintf("%d species, lengths: %v\n", len(p.Species), speciesLengths), log.DEBUG, log.DEBUG_EPOCH)

	// TODO: sort by max fitness, kill off unfit species
	// p.SortSpecies()
	for j := len(p.Species) - 1; j >= 0; j -= 1 {
		species := p.Species[j]
		species.UpdateFitnessHistory()
		log.Book(fmt.Sprintf("Local search, %d/%d...\n", j+1, len(p.Species)), log.DEBUG, log.DEBUG_EPOCH)
		species.LocalSearch()
		if species.HasStagnated() {
			log.Book(fmt.Sprintf("Stagnation, %d/%d...\n", j+1, len(p.Species)), log.DEBUG, log.DEBUG_EPOCH)
			p.Species = append(p.Species[:j], p.Species[j+1:]...)
			continue
		}
		log.Book(fmt.Sprintf("Selection, %d/%d...\n", j+1, len(p.Species)), log.DEBUG, log.DEBUG_EPOCH)
		species.Selection()
	}

	// Need another loop so recombination happens after all stagnant species are culled
	culledPopulationCount := float64(p.CountMembers())
	for i, species := range p.Species {
		log.Book(fmt.Sprintf("Recombination, %d/%d...\n", i+1, len(p.Species)), log.DEBUG, log.DEBUG_EPOCH)
		species.Recombination(culledPopulationCount)
	}

	log.Book("Separate into species...\n", log.DEBUG, log.DEBUG_EPOCH)
	p.SeparateIntoSpecies()

	massExtinct := true
	for _, species := range p.Species {
		if len(species.Members) > 0 {
			massExtinct = false
		}
	}
	if massExtinct {
		panic("Science went too far")
	}

	log.Book("Champion fitness per species:\n", log.DEBUG, log.DEBUG_EPOCH)

	champions := make([]GeneticCode, len(p.Species))
	fitnesses := make([]float64, len(p.Species))
	for i, species := range p.Species {
		champion := species.Champion()

		champions[i] = champion.GeneticCode()
		fitnesses[i] = p.FitnessOf(champion)

		log.Book(fmt.Sprintf("species #%d/%d: f=%.2g\n%s\n", i+1, len(p.Species), fitnesses[i], champions[i].ToString()), log.DEBUG, log.DEBUG_EPOCH)
	}

	log.Break(log.NL, log.DEBUG, log.DEBUG_EPOCH)

	return champions, fitnesses
}

////

// Sorting shenanigans
type SortableSpecies []*Species

func (s SortableSpecies) Len() int {
	return len(s)
}

// Want to sort in descending order, so less + greater are swapped
func (s SortableSpecies) Less(i, j int) bool {
	// Could also use average fitness instead of max fitness. Math is on the NEAT homepage
	s1MaxFitness := s[i].Population.FitnessOf(s[i].Champion())
	s2MaxFitness := s[j].Population.FitnessOf(s[j].Champion())
	return s1MaxFitness > s2MaxFitness
}

func (s SortableSpecies) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
