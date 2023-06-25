package main

import (
	"flag"
	"fmt"

	"github.com/TylerLeite/neuro-q/cppn"
	"github.com/TylerLeite/neuro-q/ge"
	"github.com/TylerLeite/neuro-q/neat"
)

func main() {

	var experiment = flag.String("test", "ge", "name of the test to run")
	flag.Parse()

	fmt.Println(*experiment)

	switch *experiment {
	case "ge":
		ge.SmallValueApproximationEvolution()
	case "xor":
		neat.XorEvolution()
	case "cppn_test":
		cppn.TestActivation()
	case "noise":
		cppn.NoiseEvolution()
	case "mandelbrot":
		cppn.MandelbrotEvolution()
	default:
		fmt.Println("bye.")
	}
}
