package neat

import "math/rand"

type Species struct {
	Members            []*Graph
	MaxFitnessOverTime []float64
}

func NewSpecies() *Species {
	s := Species{
		MaxFitnessOverTime: make([]float64, 0),
	}

	return &s
}

// Get the member of this species with the highest fitness
func (s *Species) GetChampion() *Graph {
	var champion *Graph
	for _, v := range s.Members {
		if v.Fitness() > champion.Fitness() {
			champion = v
		}
	}

	return champion
}

// Get a random member of this species
func (s *Species) GetRepresentative() *Graph {
	return s.Members[rand.Intn(len(s.Members)-1)]
}
