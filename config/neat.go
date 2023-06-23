package config

import (
	"github.com/TylerLeite/neuro-q/ma"
	"github.com/TylerLeite/neuro-q/neat"
)

type NEAT struct {
	SensorNodes int
	UsesBias    bool
	HiddenNodes int
	OutputNodes int

	RandomInitialActivations bool
	ConstantActivations      bool

	MinWeight float64
	MaxWeight float64

	MutationRatios map[ma.MutationType]float64
}

func NEATDefault() *NEAT {
	return &NEAT{
		SensorNodes: 2,
		UsesBias:    true,
		HiddenNodes: 0,
		OutputNodes: 1,

		RandomInitialActivations: false,
		ConstantActivations:      true,

		MinWeight: -1,
		MaxWeight: 1,

		MutationRatios: map[ma.MutationType]float64{
			neat.MutationAddConnection: 0.05,
			neat.MutationAddNode:       0.03,
			neat.MutationMutateWeights: 0.92,
		},
	}
}

func CPPNDefault() *NEAT {
	return &NEAT{
		SensorNodes:              3,
		UsesBias:                 true,
		HiddenNodes:              1,
		OutputNodes:              1,
		RandomInitialActivations: true,
		ConstantActivations:      false,

		MinWeight: -1,
		MaxWeight: 1,

		MutationRatios: map[ma.MutationType]float64{
			neat.MutationAddConnection:   0.05,
			neat.MutationAddNode:         0.03,
			neat.MutationMutateWeights:   0.89,
			neat.MutationChangeAFunction: 0.03,
		},
	}
}

func (*NEAT) Load(fName string) {
	//
}

func LoadNEAT(fName string) *NEAT {
	n := NEATDefault()
	n.Load(fName)
	return n
}

func LoadCPPN(fName string) *NEAT {
	n := CPPNDefault()
	n.Load(fName)
	return n
}
