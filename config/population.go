package config

import (
	"math"

	"github.com/TylerLeite/neuro-q/ma"
)

type Population struct {
	Size                     int
	DistanceThreshold        float64
	DistanceThresholdEpsilon float64

	TargetMinSpecies int
	TargetMaxSpecies int

	CullingPercent       float64
	RecombinationPercent float64
	MinimumEntropy       float64

	LocalSearchGenerations   int
	MaxEpochs                int
	DropoffAge               int
	SharingFunctionConstants []float64

	*Epoch
}

// Default config values

func PopulationDefault() *Population {
	return &Population{
		Size:                     100,
		DistanceThreshold:        math.MaxFloat64,
		DistanceThresholdEpsilon: 0,

		TargetMinSpecies: 0,
		TargetMaxSpecies: math.MaxInt,

		CullingPercent:       0.5,
		RecombinationPercent: 1,
		MinimumEntropy:       0,

		LocalSearchGenerations:   16,
		MaxEpochs:                256,
		DropoffAge:               math.MaxInt,
		SharingFunctionConstants: []float64{1, 1, 0.4, 0.1},

		Epoch: EpochDefault(),
	}
}

// new ma.Population from a config.Population
func (cfg *Population) Configure(seed ma.Organism, fitnessFunction ma.FitnessFunction) *ma.Population {
	p := ma.NewPopulation(seed, fitnessFunction)

	p.Size = cfg.Size
	p.DistanceThreshold = cfg.DistanceThreshold
	p.CullingPercent = cfg.CullingPercent
	p.RecombinationPercent = cfg.RecombinationPercent
	p.MinimumEntropy = cfg.MinimumEntropy
	p.LocalSearchGenerations = cfg.LocalSearchGenerations
	p.DropoffAge = cfg.DropoffAge
	p.Cs = make([]float64, len(cfg.SharingFunctionConstants))
	copy(p.Cs, cfg.SharingFunctionConstants)

	return p
}

func (p *Population) Load(fName string) {
	// NOTE: file @ fName should contain epoch config if not using default
}

func LoadPopulation(fName string) *Population {
	p := PopulationDefault()
	p.Load(fName)
	return p
}

type Epoch struct {
	DrawChampions   bool
	DrawPopulations bool

	LogNetworks bool
	LogGenomes  bool
	LogFitness  bool
}

func EpochDefault() *Epoch {
	return &Epoch{
		DrawChampions:   false,
		DrawPopulations: false,

		LogNetworks: false,
		LogGenomes:  false,
		LogFitness:  true,
	}
}

func (e *Epoch) Load(fName string) {
	//
}

func LoadEpoch(fName string) *Epoch {
	e := EpochDefault()
	e.Load(fName)
	return e
}
