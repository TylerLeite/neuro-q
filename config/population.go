package config

import "math"

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
