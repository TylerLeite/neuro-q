package ma

import (
	"bytes"
	"compress/flate"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/TylerLeite/neuro-q/log"
)

// Used for sorting by fitness
type SortableOrganisms struct {
	organisms []Organism
	ff        FitnessFunction
}

func (o SortableOrganisms) Len() int {
	return len(o.organisms)
}

// Want to sort in descending order, so less + greater are swapped
func (o SortableOrganisms) Less(i, j int) bool {
	return o.ff(o.organisms[i]) > o.ff(o.organisms[j])
}

func (o SortableOrganisms) Swap(i, j int) {
	o.organisms[i], o.organisms[j] = o.organisms[j], o.organisms[i]
}

// TODO: Species.ID for image naming
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

func (s *Species) String() string {
	out := ""
	champ := s.Champion()
	for i, o := range s.Members {
		distance := o.GeneticCode().DistanceFrom(champ.GeneticCode(), s.Population.Cs...)
		out += fmt.Sprintf("%d [%.2g]: %s\n", i, distance, o.GeneticCode().String())
	}
	return out
}

func (s *Species) Copy(to *Population) *Species {
	newSpecies := Species{
		Population:     to,
		Members:        make([]Organism, len(s.Members)),
		FitnessHistory: make([]float64, len(s.FitnessHistory)),
		dropoffAge:     s.dropoffAge,
	}

	for i, v := range s.Members {
		newSpecies.Members[i] = v.Copy()
	}

	copy(newSpecies.FitnessHistory, s.FitnessHistory)

	return &newSpecies
}

// Get the member of this species with the highest fitness
func (s *Species) Champion() Organism {
	champion := s.Members[0]
	for _, v := range s.Members {
		if s.Population.FitnessOf(v) > s.Population.FitnessOf(champion) {
			champion = v
		}
	}

	return champion
}

// Get a random member of this species
func (s *Species) RandomOrganism() Organism {
	if len(s.Members) == 1 {
		return s.Members[0]
	}
	return s.Members[rand.Intn(len(s.Members)-1)]
}

func (s *Species) AverageFitness() float64 {
	totalFitness := float64(0)

	for _, v := range s.Members {
		totalFitness += s.Population.FitnessOf(v)
	}

	return totalFitness / float64(len(s.Members))
}

// Local search
func (s *Species) LocalSearch() {
	for i, organism := range s.Members {
		currentFitness := s.Population.FitnessOf(organism)
		mostFitNeighbor := organism

		for j := 0; j < s.Population.LocalSearchGenerations; j += 1 {
			neighbor := organism.RandomNeighbor()
			neighborFitness := s.Population.FitnessOf(neighbor)
			if neighborFitness > currentFitness {
				mostFitNeighbor = neighbor
				currentFitness = neighborFitness
			}
		}

		// Lamarckian learning: the new organism replaces the old one
		s.Members[i] = mostFitNeighbor
	}
}

// selection
func (s *Species) Selection() {
	if len(s.Members) == 1 {
		return
	}

	// Sort by fitness
	so := SortableOrganisms{
		organisms: s.Members,
		ff:        s.Population.FitnessOf,
	}
	sort.Sort(so)

	// Cull the least fit organisms
	// TODO: distribute cullling randomly following a power law
	// idea: nondeterministic # culled -> recombine + add random to meet total population, following config %s
	numberToCull := int(math.Round(s.Population.CullingPercent * float64(len(s.Members))))
	s.Members = s.Members[:len(s.Members)-numberToCull]
}

// Recombination (mating)
func (s *Species) Recombination(culledPopulationCount float64) {
	// Store children in a new slice during recombination so they aren't chosen as parents

	// TODO: Ability to enable fitness sharing in config (speciesPopulationPercent based on species fitness, scaled by current species size)
	// NOTE: with fitness sharing enabled, negative fitness must be disallowed -> translate fitness values by -(minimum fitness in population)
	thisSpeciesPopulationPercent := float64(len(s.Members)) / culledPopulationCount
	speciesTargetSize := int(math.Round(float64(s.Population.Size) * thisSpeciesPopulationPercent))
	numberToRecombine := int(math.Round(float64(speciesTargetSize) * s.Population.RecombinationPercent))
	numberToMutate := speciesTargetSize - numberToRecombine

	log.Book(fmt.Sprintf("Next population stats: r=%d, m=%d\n", numberToRecombine, numberToMutate), log.DEBUG, log.DEBUG_RECOMBINATION)

	// Make room for last generation's champion
	if numberToRecombine > numberToMutate {
		numberToRecombine -= 1
	} else {
		numberToMutate -= 1
	}

	children := make([]Organism, numberToRecombine)
	for i := 0; i < numberToRecombine; i += 1 {
		var child Organism
		r1 := rand.Intn(len(s.Members))

		if len(s.Members) < 2 {
			// Can't do crossover, so reproduce asexually
			child = s.Members[r1].RandomNeighbor()
		} else {
			// Select another parent at random
			// TODO: sexual selection?
			r2 := r1
			for r2 == r1 {
				r2 = rand.Intn(len(s.Members))
			}

			// Baby make
			child = s.Members[r1].Crossover([]Organism{s.Members[r2]})
		}

		children[i] = child
	}

	clones := make([]Organism, numberToMutate)
	for i := 0; i < numberToMutate; i += 1 {
		r := rand.Intn(len(s.Members))
		clone := s.Members[r].RandomNeighbor()
		clones[i] = clone
	}

	champion := s.Champion()

	s.Members = []Organism{champion}
	s.Members = append(s.Members, children...)
	s.Members = append(s.Members, clones...)
}

// May want to check stagnation, so each species keeps a history of their max fitness
func (s *Species) UpdateFitnessHistory() {
	s.FitnessHistory = append(s.FitnessHistory, s.Population.FitnessOf(s.Champion()))
}

// Check convergeance of a species by measuring its entropy
// This is done indirectly by compressing the concatenation of all genomes in the species
func (s *Species) HasConverged() bool {
	corpus := ""
	for _, v := range s.Members {
		corpus += v.GeneticCode().String()
	}

	// Check entropy by compressing the population
	var buf bytes.Buffer
	w, _ := flate.NewWriter(&buf, 9)
	w.Write([]byte(corpus))
	w.Close()

	ratio := float64(len(buf.Bytes())) / float64(len(corpus))
	return ratio < s.Population.MinimumEntropy
}

// TODO: config
const epsilon = 0.01

func (s *Species) HasStagnated() bool {
	age := len(s.FitnessHistory)

	// Species is not old enough to check for stagnation yet
	if age < s.dropoffAge {
		return false
	}

	// If there has been a fitness increase anywhere in the last 15 generations, the species has not stagnated
	for i := age - 1; i > age-s.dropoffAge; i -= 1 {
		if s.FitnessHistory[i] > s.FitnessHistory[i-1]+epsilon {
			return false
		}
	}

	return true
}
