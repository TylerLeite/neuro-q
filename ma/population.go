package ma

import "math"

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
}

func NewPopulation(size int, seed Organism, fitnessFunction FitnessFunction) *Population {
	p := Population{
		Species:   make([]*Species, 0),
		Size:      size,
		Seed:      seed,
		FitnessOf: fitnessFunction,

		// Default config values
		CullingPercent:         0.5,
		RecombinationPercent:   1,
		MinimumEntropy:         0.5,
		LocalSearchGenerations: 16,
		DropoffAge:             15,
		DistanceThreshold:      math.MaxFloat64,
	}

	return &p
}

func (p *Population) Copy() *Population {
	newPopulation := Population{
		Species:   make([]*Species, len(p.Species)),
		Size:      p.Size,
		Seed:      p.Seed,
		FitnessOf: p.FitnessOf,

		CullingPercent:         p.CullingPercent,
		RecombinationPercent:   p.RecombinationPercent,
		MinimumEntropy:         p.MinimumEntropy,
		LocalSearchGenerations: p.LocalSearchGenerations,
		DropoffAge:             p.DropoffAge,
		DistanceThreshold:      p.DistanceThreshold,
	}

	for i, v := range p.Species {
		newPopulation.Species[i] = v.Copy()
	}

	return &newPopulation
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

func (p *Population) CountMembers() uint {
	var total uint

	for _, species := range p.Species {
		total += uint(len(species.Members))
	}

	return total
}

// generate initial population
func (p *Population) Generate() {
	p.Species = make([]*Species, 1)
	p.Species[0] = NewSpecies(p)
	for i := 0; i < p.Size; i += 1 {
		newOrganism := p.Seed.Copy()
		randomGenome := newOrganism.GeneticCode().Randomize()
		newOrganism.LoadGeneticCode(randomGenome)

		p.Species[0].Members = append(p.Species[0].Members, newOrganism)
	}
}

// Output a new, speciated population
func (p *Population) SeparateIntoSpecies() *Population {
	newPopulation := p.CopyConfig()

	representatives := make([]Organism, len(p.Species))
	newPopulation.Species = make([]*Species, len(p.Species))

	for _, currentIndividual := range p.Members() {
		foundASpecies := false

		for i := 0; i < len(p.Species); i += 1 {
			species := p.Species[i]
			if representatives[i] == nil {
				// First pick a representative for the species if one doesn't exist
				representatives[i] = species.RandomOrganism()
			}

			// Place this individual into the first species where it fits
			// TODO: figure out actual values for these constants
			// pop 150 -> c3 = 0.4, threshold = 3.0 | pop 1000 -> c3 = 3.0, threshold = 4.0
			d := currentIndividual.GeneticCode().DistanceFrom(representatives[i].GeneticCode(), 1, 1, 0.4)
			if d < newPopulation.DistanceThreshold {
				newPopulation.Species[i].Members = append(newPopulation.Species[i].Members, currentIndividual.Copy())
				foundASpecies = true
				break
			}
		}

		if !foundASpecies {
			// Make a new species with this individual as the representative
			newPopulation.Species = append(newPopulation.Species, NewSpecies(newPopulation))
			newPopulation.Species[len(newPopulation.Species)-1].Members[0] = currentIndividual.Copy()
			representatives = append(representatives, currentIndividual)
		}
	}

	// TODO: Clean up empty species
	return newPopulation
}

// TODO: Make sure population is at its size. Either cull extra organisms or distribute new organisms among most fit species
func (p *Population) Stabilization() *Population {
	return p
}
