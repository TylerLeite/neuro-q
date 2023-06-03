package ma

type Population struct {
	Species []*Species

	Size int

	Seed Organism

	CullingPercent         float64
	RecombinationPercent   float64
	MinimumEntropy         float64
	LocalSearchGenerations int
	DropoffAge             int
}

func NewPopulation(size int) *Population {
	p := Population{
		Species: make([]*Species, 0),
		Size:    size,

		// Default config values
		CullingPercent:         0.5,
		RecombinationPercent:   1,
		MinimumEntropy:         0.5,
		LocalSearchGenerations: 16,
		DropoffAge:             15,
	}

	return &p
}

func (p *Population) Copy() *Population {
	newPopulation := Population{
		Species: make([]*Species, len(p.Species)),
		Size:    p.Size,

		Seed: p.Seed,

		CullingPercent:         p.CullingPercent,
		RecombinationPercent:   p.RecombinationPercent,
		MinimumEntropy:         p.MinimumEntropy,
		LocalSearchGenerations: p.LocalSearchGenerations,
		DropoffAge:             p.DropoffAge,
	}

	for i, v := range p.Species {
		newPopulation.Species[i] = v.Copy()
	}

	return &newPopulation
}

func (p *Population) Members() []Organism {
	//Aggregate members across all species in population
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

// TODO: Make sure population is at its size. Either cull extra organisms or distribute new organisms among most fit species
func (p *Population) Stabilization() *Population {
	return p
}
