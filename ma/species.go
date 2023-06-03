package ma

import (
	"bytes"
	"compress/flate"
	"math"
	"math/rand"
	"sort"
)

// NOTE: current idea of a population fits more closely to a species, actually

// Used for sorting by fitness
type SortableOrganisms []Organism

func (organisms SortableOrganisms) Len() int {
	return len(organisms)
}

// Want to sort in descending order, so less + greater are swapped
func (organisms SortableOrganisms) Less(i, j int) bool {
	return organisms[i].Fitness() > organisms[j].Fitness()
}

func (organisms SortableOrganisms) Swap(i, j int) {
	organisms[i], organisms[j] = organisms[j], organisms[i]
}

type Species struct {
	Population     *Population
	Members        []Organism
	FitnessHistory []float64

	dropoffAge int // How many generations the species can go without improving its max fitness
}

func NewSpecies(p *Population) *Species {
	s := Species{
		Population:     p,
		Members:        make([]Organism, 0),
		FitnessHistory: make([]float64, 0),

		dropoffAge: p.DropoffAge,
	}

	return &s
}

func (s *Species) Copy() *Species {
	newSpecies := Species{
		Population:     s.Population,
		Members:        make([]Organism, len(s.Members)),
		FitnessHistory: make([]float64, len(s.FitnessHistory)),
	}

	for i, v := range s.Members {
		newSpecies.Members[i] = v.Copy()
	}

	copy(newSpecies.FitnessHistory, s.FitnessHistory)

	return &newSpecies
}

// Get the member of this species with the highest fitness
func (s *Species) GetChampion() Organism {
	var champion Organism
	for _, v := range s.Members {
		if v.Fitness() > champion.Fitness() {
			champion = v
		}
	}

	return champion
}

// Get a random member of this species
func (s *Species) GetRandomOrganism() Organism {
	return s.Members[rand.Intn(len(s.Members)-1)]
}

// Local search
func (s *Species) LocalSearch() *Species {
	newSpecies := s.Copy()

	for i, organism := range s.Members {
		currentFitness := organism.Fitness()
		currentOrganism := organism
		for j := 0; j < s.Population.LocalSearchGenerations; j += 1 {
			neighbor := organism.RandomNeighbor()
			neighborFitness := neighbor.Fitness()
			if neighbor.Fitness() > currentFitness {
				currentOrganism = neighbor
				currentFitness = neighborFitness
			}
		}

		// Lamarckian learning: the new organism replaces the old one
		newSpecies.Members[i] = currentOrganism
	}

	return newSpecies
}

// selection
func (s *Species) Selection() *Species {
	newSpecies := s.Copy()

	// Sort by fitness
	sort.Sort(SortableOrganisms(newSpecies.Members))

	// Cull the least fit organisms
	// TODO: also cull some randomly following a power law
	numberToCull := int(math.Floor(newSpecies.Population.CullingPercent * float64(len(newSpecies.Members))))
	newSpecies.Members = newSpecies.Members[:len(newSpecies.Members)-numberToCull]

	return newSpecies
}

// Recombination (mating)
func (s *Species) Recombination() *Species {
	newSpecies := s.Copy()

	// Store children in a new slice during recombination so they aren't chosen as parents
	numberToRecombine := int(math.Floor(newSpecies.Population.RecombinationPercent * float64(len(newSpecies.Members))))
	children := make([]Organism, numberToRecombine)

	for i := 0; i < numberToRecombine; i += 1 {
		// Select two random parents
		r1 := rand.Intn(len(newSpecies.Members))
		r2 := r1
		for r2 == r1 {
			r2 = rand.Intn(len(newSpecies.Members))
		}

		// Baby make
		childGeneticCode := newSpecies.Members[r1].GeneticCode().Crossover([]GeneticCode{newSpecies.Members[r2].GeneticCode()})
		child := newSpecies.Members[r1].Copy()
		child.LoadGeneticCode(childGeneticCode)
		children[i] = child
	}

	newSpecies.Members = append(newSpecies.Members, children...)
	return newSpecies
}

// May want to check stagnation, so each species keeps a history of their max fitness
func (s *Species) UpdateFitnessHistory() {
	s.FitnessHistory = append(s.FitnessHistory, s.GetChampion().Fitness())
}

// Check convergeance of a species by measuring its entropy
// This is done indirectly by compressing the concatenation of all genomes in the species
func (s *Species) HasConverged() bool {
	corpus := ""
	for _, v := range s.Members {
		corpus += v.GeneticCode().ToString()
	}

	// Check entropy by compressing the population
	var buf bytes.Buffer
	w, _ := flate.NewWriter(&buf, 9)
	w.Write([]byte(corpus))
	w.Close()

	ratio := float64(len(buf.Bytes())) / float64(len(corpus))
	return ratio < s.Population.MinimumEntropy
}

func (s *Species) HasStagnated() bool {
	age := len(s.FitnessHistory)

	// Species is not old enough to check for stagnation yet
	if age < s.dropoffAge {
		return false
	}

	// If there has been a fitness increase anywhere in the last 15 generations, the species has not stagnated
	for i := age - 1; i < age-s.dropoffAge; i -= 1 {
		if s.FitnessHistory[i] > s.FitnessHistory[i-1] {
			return false
		}
	}

	return true
}
