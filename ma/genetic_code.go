package ma

type MutationType uint8

type GeneticCode interface {
	Copy() GeneticCode

	Randomize() // Return a random genetic code

	ListMutations() map[string]MutationType
	Mutate(MutationType, interface{}) // Perform a specific mutation

	DistanceFrom(GeneticCode, float64, float64, float64) float64

	ToString() string // Genetic code as a string, used for calculating population entropy
}
