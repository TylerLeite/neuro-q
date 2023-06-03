package ma

type MutationType uint8

type GeneticCode interface {
	Copy() GeneticCode

	Randomize() GeneticCode // Return a random genetic code

	ListMutations() map[string]MutationType
	Mutate(MutationType) GeneticCode // Perform a specific mutation

	Crossover(others []GeneticCode) GeneticCode // I am very progressive, so individuals can have any positive number of parents

	ToString() string // Genetic code as a string, used for calculating population entropy
}
