package neat

import (
	"math"
)

var seed int64 = 0

func randi() int64 {
	seed = (1028597*seed + 488249) % 1737017
	return seed
}

func Seed(s int64) {
	seed = s
}

type ActivationFunction func(float64) float64

// TODO: Use an enum for function names
func RandomFunc() (ActivationFunction, string) {
	const totalFunctions = 15

	p := randi() % totalFunctions
	if p <= 0 {
		return SinFunc, "Sine eave"
	} else if p <= 1 {
		return Sin2Func, "Double-period sine wave"
	} else if p <= 2 {
		return AbsFunc, "Absolute value"
	} else if p <= 3 {
		return NullFunc, "Null function"
	} else if p <= 4 {
		return GaussianFunc, "Gaussian function"
	} else if p <= 5 {
		return SigmoidFunc, "Sigmoid"
	} else if p <= 10 {
		return NEATSigmoidFunc, "NEAT sigmoid"
	} else if p <= 6 {
		return BipolarSigmoidFunc, "Bipolar sigmoid"
	} else if p <= 7 {
		return QuadraticFunc, "Quadratic"
	} else if p <= 8 {
		return StepFunc, "Step function"
	} else if p <= 9 {
		return InversionFunc, "Negative"
	} else if p <= 11 {
		return ExponentiationFunc, "Exponentiation"
	} else if p <= 12 {
		return TetrationFunc, "Second Tetration"
	} else if p <= 13 {
		return SawFunc, "Sawtooth wave"
	} else {
		return IdentityFunc, "Identity"
	}
}

func FuncByName(name string) ActivationFunction {
	switch name {
	case "Sine wave":
		return SinFunc
	case "Double-period sine wave":
		return Sin2Func
	case "Absolute value":
		return AbsFunc
	case "Null function":
		return NullFunc
	case "Gaussian function":
		return GaussianFunc
	case "Sigmoid":
		return SigmoidFunc
	case "NEAT sigmoid":
		return NEATSigmoidFunc
	case "Bipolar sigmoid":
		return BipolarSigmoidFunc
	case "Quadratic":
		return QuadraticFunc
	case "Step function":
		return StepFunc
	case "Negative":
		return InversionFunc
	case "Exponentiation":
		return ExponentiationFunc
	case "Second Tetration":
		return TetrationFunc
	case "Sawtooth wave":
		return SawFunc
	default:
		return IdentityFunc
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

func BipolarSigmoidFunc(x float64) float64 {
	return 2.0/(1.0+math.Exp(-4.9*x)) - 1.0
}

func QuadraticFunc(x float64) float64 {
	return x * x
}

func SawFunc(x float64) float64 {
	return math.Mod(x, 1)
}

func StepFunc(x float64) float64 {
	return math.Floor(x*10) / 10
}

func InversionFunc(x float64) float64 {
	return -x
}

func ExponentiationFunc(x float64) float64 {
	// x-1 so that |f(x)| on (-1, 1) stays <= 1
	return math.Exp(x - 1)
}

func TetrationFunc(x float64) float64 {
	return math.Pow(x, x)
}

func IdentityFunc(x float64) float64 {
	return x
}

func NEATSigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-4.9*x))
}
