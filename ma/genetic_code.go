package ma

type MutationType uint8

type GeneticCode interface {
	Copy() GeneticCode

	Randomize() GeneticCode // Return a random genetic code

	ListMutations() map[string]MutationType
	Mutate(MutationType, interface{}) GeneticCode // Perform a specific mutation

	DistanceFrom(GeneticCode, float64, float64, float64) float64

	ToString() string // Genetic code as a string, used for calculating population entropy
}
