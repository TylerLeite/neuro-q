package cppn

import (
	"math"
)

type Func func(float64) float64

var seed int64 = 0

func randi() int64 {
	seed = (1028597*seed + 488249) % 1737017
	return seed
}

func Seed(s int64) {
	seed = s
}

func RandomFunc() (Func, string) {
	const totalWeights = 11

	p := randi() % totalWeights
	if p <= 0 {
		return SinFunc, "Sine"
	} else if p <= 1 {
		return Sin2Func, "Double-period sine"
	} else if p <= 2 {
		return AbsFunc, "Absolute value"
	} else if p <= 3 {
		return NullFunc, "Null function"
	} else if p <= 4 {
		return GaussianFunc, "Gaussian function"
	} else if p <= 5 {
		return SigmoidFunc, "Sigmoid"
	} else if p <= 10 {
		return SigmoidFunc, "NEAT Sigmoid"
	} else if p <= 6 {
		return BipolarSigmoidFunc, "Bipolar sigmoid"
	} else if p <= 7 {
		return QuadraticFunc, "Quadratic"
	} else if p <= 8 {
		return StepFunc, "Step function"
	} else if p <= 9 {
		return InversionFunc, "Negative"
	} else {
		return IdentityFunc, "Identity"
	}
}

func SinFunc(x float64) float64 {
	return math.Sin(x)
}

func Sin2Func(x float64) float64 {
	return math.Sin(2 * x)
}

func AbsFunc(x float64) float64 {
	return math.Abs(x)
}

func NullFunc(x float64) float64 {
	return 0.0
}

func GaussianFunc(x float64) float64 {
	return 2.0*math.Exp(-math.Pow(2.5*x, 2)) - 1.0
}

func SigmoidFunc(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func NEATSigmoidFunc(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-4.9*x))
}

func BipolarSigmoidFunc(x float64) float64 {
	return 2.0/(1.0+math.Exp(-4.9*x)) - 1.0
}

func QuadraticFunc(x float64) float64 {
	return x * x
}

func StepFunc(x float64) float64 {
	return math.Mod(x, 1)
}

func InversionFunc(x float64) float64 {
	return -x
}

func IdentityFunc(x float64) float64 {
	return x
}
