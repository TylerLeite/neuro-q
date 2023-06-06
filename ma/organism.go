package ma

type Organism interface {
	Copy() Organism                          // Duplicate this Individual (same genome)
	RandomNeighbor() Organism                // Get a new individual with a slight mutation compared to this one
	NewFromGeneticCode(GeneticCode) Organism // TODO: this is a bit janky. Is there a better way?

	Crossover([]Organism) Organism // I am very progressive, so individuals can have any positive number of parents

	GeneticCode() GeneticCode
	LoadGeneticCode(GeneticCode)
	Compile() error
	IsCompiled() bool
}
