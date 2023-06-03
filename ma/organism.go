package ma

type Organism interface {
	Copy() Organism                          // Duplicate this Individual (same genome)
	RandomNeighbor() Organism                // Get a new individual with a slight mutation compared to this one
	NewFromGeneticCode(GeneticCode) Organism // TODO: this is a bit janky. Is there a better way?

	GeneticCode() GeneticCode
	LoadGeneticCode(GeneticCode)
	Compile() error
	IsCompiled() bool

	Fitness() float64
}
