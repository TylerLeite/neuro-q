package cppn

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/TylerLeite/neuro-q/log"
	"github.com/TylerLeite/neuro-q/ma"
	"github.com/TylerLeite/neuro-q/neat"
)

func Evolution(fn ma.FitnessFunction, dfn func(ma.Organism, string) float64) {
	neat.ResetInnovationHistory()

	seedGenome := neat.NewGenome(2, 3, true, -1, 1)

	// TODO: generation strategy that allows for hidden nodes in seed genome
	seedGenome.ActivationFunctions = map[uint]string{
		0: "Identity",
		1: "Identity",
		2: "Identity",
		3: "Sigmoid",
		4: "Sigmoid",
		5: "Sigmoid",
	}
	seedGenome.MutationRatios = map[ma.MutationType]float64{
		neat.MutationAddConnection: 0.05,
		neat.MutationAddNode:       0.15,
		neat.MutationMutateWeights: 0.8,
		// Also should add MutationMutateActivation
	}

	seedNetwork := neat.NewNetwork(seedGenome, nil)

	p := ma.NewPopulation(ma.Organism(seedNetwork), fn)
	seedNetwork.Population = p

	p.Size = 64
	p.DistanceThreshold = 1
	p.CullingPercent = 0.5
	p.RecombinationPercent = 0.75
	p.MinimumEntropy = 0.35
	p.LocalSearchGenerations = 8
	p.DropoffAge = 15
	p.Cs = []float64{1, 2, 0.4, 1}

	speciesTargetMin := 7
	speciesTargetMax := 13
	distanceThresholdEpsilon := 0.1

	fmt.Println("Generating...")
	p.Generate()

	// Randomize activation functions of seed members
	for _, o := range p.Members() {
		network := o.(*neat.Network)
		genome := network.DNA

		for nodeId := range genome.ActivationFunctions {
			_, newFnName := neat.RandomFunc()
			genome.ActivationFunctions[nodeId] = newFnName
		}

		network.ForceCompile()
	}

	G := 100
	for i := 0; i < G; i += 1 {
		fmt.Printf("New generation, %d/%d [%d species] dt=%.2g\n", i+1, G, len(p.Species), p.DistanceThreshold)

		p.Epoch()

		if len(p.Species) > speciesTargetMax {
			p.DistanceThreshold += distanceThresholdEpsilon
		} else if len(p.Species) < speciesTargetMin {
			p.DistanceThreshold -= distanceThresholdEpsilon
		}

		for j, species := range p.Species {
			championNetwork := species.Champion().(*neat.Network)
			championNetwork.Draw(fmt.Sprintf("cppn/drawn/%d_%d.bmp", i, j))
			fitness := dfn(championNetwork, fmt.Sprintf("cppn/drawn/%d_%d.png", i, j))
			fmt.Printf("\tSpecies %d fitness = %.4g\n", j+1, fitness)
		}
	}
}

func _CPPNFitness(o ma.Organism, filename string) float64 {
	n := o.(*neat.Network)
	n.Compile()

	var (
		bias *neat.Node
		inX  *neat.Node
		inY  *neat.Node
	)

	for _, nodeI := range n.DNA.SensorNodes {
		node := n.Nodes[nodeI]
		if node.Label == "0" {
			bias = node
		} else if node.Label == "1" {
			inX = node
		} else {
			inY = node
		}
	}

	var (
		outR *neat.Node
		outG *neat.Node
		outB *neat.Node
	)

	for _, nodeI := range n.DNA.OutputNodes {
		node := n.Nodes[nodeI]
		if node.Label == "3" {
			outR = node
		} else if node.Label == "4" {
			outG = node
		} else {
			outB = node
		}
	}

	fitness := float64(0)

	usedColors := make(map[string]bool)

	const (
		w = 32
		h = 32
	)

	var img *image.RGBA
	if filename != "" {
		img = image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})
	}

	for x := 0; x < w; x += 1 {
		for y := 0; y < h; y += 1 {
			inputValues := []float64{1, float64(x) / w, float64(y) / h}
			n.Activate(inputValues, []*neat.Node{bias, inX, inY}, []*neat.Node{outR, outG, outB})

			r := math.Floor(16 * outR.Value())
			g := math.Floor(16 * outG.Value())
			b := math.Floor(16 * outB.Value())

			colorString := fmt.Sprintf("%X.%X.%X", r, g, b)

			if filename != "" {
				img.Set(x, y, color.RGBA{uint8(r * 16), uint8(g * 16), uint8(b * 16), 0xff})
			}

			if _, ok := usedColors[colorString]; !ok {
				usedColors[colorString] = true
				fitness += 1
			}

			if math.IsNaN(r) || math.IsNaN(g) || math.IsNaN(b) {
				log.Book(n.ToString(), log.DEBUG)
				panic("NaN network")
			}
		}
	}

	if filename != "" {
		f, _ := os.Create(filename)
		png.Encode(f, img)
	}

	return fitness
}

func CPPNFitness(o ma.Organism) float64 {
	return _CPPNFitness(o, "")
}

func NoiseEvolution() {
	Evolution(CPPNFitness, _CPPNFitness)
}

func calculateMandelbrotAt(x0, y0, scaleX, scaleY float64) uint8 {
	x0 = 2.0 - x0/scaleX*2.47
	y0 = 1.12 - y0/scaleY*2.24

	x := x0
	y := y0

	i := uint8(0)
	for (x*x+y*y <= 2*2) && (i < math.MaxUint8) {
		x, y = x*x-y*y+x0, 2*x*y+y0
		i += 1
	}

	return i
}

func MandelbrotEvolution() {
	mandelbrotPixels := make([][]uint8, 224)
	for i := 0; i < len(mandelbrotPixels); i += 1 {
		mandelbrotPixels[i] = make([]uint8, 247)
		for j := 0; j < len(mandelbrotPixels[i]); j += 1 {
			mandelbrotPixels[i][j] = calculateMandelbrotAt(float64(j), float64(i), 247, 224)
		}
	}

	_mandelbrotFitness := func(o ma.Organism, filename string) float64 {
		n := o.(*neat.Network)
		n.Compile()

		var (
			bias *neat.Node
			inX  *neat.Node
			inY  *neat.Node
		)
		for _, nodeI := range n.DNA.SensorNodes {
			node := n.Nodes[nodeI]
			if node.Label == "0" {
				bias = node
			} else if node.Label == "1" {
				inX = node
			} else {
				inY = node
			}
		}

		out := n.Nodes[n.DNA.OutputNodes[0]]

		fitness := float64(0)

		const (
			w = 247
			h = 224
		)

		var img *image.RGBA
		if filename != "" {
			img = image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})
		}

		for x := 0; x < w; x += 1 {
			for y := 0; y < h; y += 1 {
				inputValues := []float64{1, float64(x) / w, float64(y) / h}
				n.Activate(inputValues, []*neat.Node{bias, inX, inY}, []*neat.Node{out})

				value := math.Floor(255 * out.Value())

				if math.IsNaN(value) {
					log.Book(n.ToString(), log.DEBUG)
					panic("NaN network")
				}

				r := uint8(16 * (int(value) % 16))
				g := uint8(16 * (math.Floor(value / 16)))
				b := uint8(value)

				if filename != "" {
					img.Set(x, y, color.RGBA{r, g, b, 0xff})
				}

				diff := value - float64(mandelbrotPixels[y][x])
				fitness += diff * diff

			}
		}

		if filename != "" {
			f, _ := os.Create(filename)
			png.Encode(f, img)
		}

		return fitness
	}

	mandelbrotFitness := func(o ma.Organism) float64 {
		return _mandelbrotFitness(o, "")
	}

	Evolution(mandelbrotFitness, _mandelbrotFitness)
}
