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

const (
	SinStr            = "Sine wave"
	Sin2Str           = "Double-period sine wave"
	AbsStr            = "Absolute value"
	NullStr           = "Null function"
	GaussianStr       = "Gaussian function"
	SigmoidStr        = "Sigmoid"
	NEATSigmoidStr    = "NEAT sigmoid"
	BipolarSigmoidStr = "Bipolar sigmoid"
	QuadraticStr      = "Quadratic"
	StepStr           = "Step function"
	InversionStr      = "Negative"
	ExponentiationStr = "Exponentiation"
	TetrationStr      = "Second Tetration"
	SawStr            = "Sawtooth wave"
	IdentityStr       = "Identity"
)

// TODO: Use an enum for function names
func RandomFunc() (ActivationFunction, string) {
	const totalFunctions = 15

	p := randi() % totalFunctions
	if p <= 0 {
		return SinFunc, SinStr
	} else if p <= 1 {
		return Sin2Func, Sin2Str
	} else if p <= 2 {
		return AbsFunc, AbsStr
	} else if p <= 3 {
		return NullFunc, NullStr
	} else if p <= 4 {
		return GaussianFunc, GaussianStr
	} else if p <= 5 {
		return SigmoidFunc, SigmoidStr
	} else if p <= 10 {
		return NEATSigmoidFunc, NEATSigmoidStr
	} else if p <= 6 {
		return BipolarSigmoidFunc, BipolarSigmoidStr
	} else if p <= 7 {
		return QuadraticFunc, QuadraticStr
	} else if p <= 8 {
		return StepFunc, StepStr
	} else if p <= 9 {
		return InversionFunc, InversionStr
	} else if p <= 11 {
		return ExponentiationFunc, ExponentiationStr
	} else if p <= 12 {
		return TetrationFunc, TetrationStr
	} else if p <= 13 {
		return SawFunc, SawStr
	} else {
		return IdentityFunc, IdentityStr
	}
}

func FuncByName(name string) ActivationFunction {
	switch name {
	case SinStr:
		return SinFunc
	case Sin2Str:
		return Sin2Func
	case AbsStr:
		return AbsFunc
	case NullStr:
		return NullFunc
	case GaussianStr:
		return GaussianFunc
	case SigmoidStr:
		return SigmoidFunc
	case NEATSigmoidStr:
		return NEATSigmoidFunc
	case BipolarSigmoidStr:
		return BipolarSigmoidFunc
	case QuadraticStr:
		return QuadraticFunc
	case StepStr:
		return StepFunc
	case InversionStr:
		return InversionFunc
	case ExponentiationStr:
		return ExponentiationFunc
	case TetrationStr:
		return TetrationFunc
	case SawStr:
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
	// Sadly, tetrating a negative number gives you a complex result :/
	return math.Pow(math.Abs(x), x)
}

func IdentityFunc(x float64) float64 {
	return x
}

func NEATSigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-4.9*x))
}
