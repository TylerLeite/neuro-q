package ma

type Individual interface {
	Copy() Individual           // Duplicate this Individual (same genome)
	Randomize() Individual      // Overwrite the genome with a random one
	RandomNeighbor() Individual // Get a new individual with a slight mutation compared to this one

	Fitness() float64
	Crossover(others []Individual) Individual // I am very progressive, so individuals can have any positive number of parents

	CodeString() string // Genetic code as a string, used for calculating population entropy
}
