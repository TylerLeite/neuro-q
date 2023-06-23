package cppn

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strconv"

	"github.com/TylerLeite/neuro-q/config"
	"github.com/TylerLeite/neuro-q/log"
	"github.com/TylerLeite/neuro-q/ma"
	"github.com/TylerLeite/neuro-q/neat"
)

type DrawFunction func(ma.Organism, string) error

func Evolution(
	fn ma.FitnessFunction,
	drawFn DrawFunction,
	popCfg *config.Population,
	neatCfg *config.NEAT,
) {
	neat.ResetInnovationHistory()

	// TODO: NewGenomeFromConfig
	seedGenome := neat.NewGenome(
		neatCfg.SensorNodes,
		neatCfg.OutputNodes,
		neatCfg.UsesBias,
		neatCfg.MinWeight,
		neatCfg.MaxWeight,
	)

	if !neatCfg.ConstantActivations {
		i := 0
		seedGenome.ActivationFunctions = make(map[uint]string)

		if neatCfg.UsesBias {
			seedGenome.ActivationFunctions[0] = "Identity"
			i += 1
		}

		for j := 0; j < neatCfg.SensorNodes; j += 1 {
			seedGenome.ActivationFunctions[uint(i+j)] = neat.IdentityStr
		}
		i += neatCfg.SensorNodes

		for j := 0; j < neatCfg.OutputNodes; j += 1 {
			seedGenome.ActivationFunctions[uint(i+j)] = neat.NEATSigmoidStr
		}

		// TODO: allow for hidden nodes in seed genome
	}

	seedGenome.MutationRatios = neatCfg.MutationRatios // TODO: deep copy?

	seedNetwork := neat.NewNetwork(seedGenome, nil)

	// TODO: NewPopulationFromConfig
	p := ma.NewPopulation(ma.Organism(seedNetwork), fn)
	seedNetwork.Population = p

	p.Size = popCfg.Size
	p.DistanceThreshold = popCfg.DistanceThreshold
	p.CullingPercent = popCfg.CullingPercent
	p.RecombinationPercent = popCfg.RecombinationPercent
	p.MinimumEntropy = popCfg.MinimumEntropy
	p.LocalSearchGenerations = popCfg.LocalSearchGenerations
	p.DropoffAge = popCfg.DropoffAge

	p.Cs = popCfg.SharingFunctionConstants // TODO: use copy()?

	speciesTargetMin := popCfg.TargetMinSpecies
	speciesTargetMax := popCfg.TargetMaxSpecies
	distanceThresholdEpsilon := popCfg.DistanceThresholdEpsilon

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

	G := popCfg.MaxEpochs
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
			drawFn(championNetwork, fmt.Sprintf("cppn/drawn/%d_%d.png", i, j))

			fitness := fn(championNetwork)

			fmt.Printf("\tSpecies %d fitness = %.4g\n", j+1, fitness)
		}
	}
}

type NetworkInputFunction func(...float64) float64

func ActivateNetwork(n *neat.Network, dimensions []int, otherInputs []NetworkInputFunction) [][]float64 {
	outSize := 1
	for _, dimension := range dimensions {
		outSize *= dimension
	}

	log.Book(fmt.Sprintf("Outmatrix shape %dx%d\n", outSize, len(n.DNA.OutputNodes)), log.DEBUG, log.DEBUG_EXPERIMENT)

	outMatrix := make([][]float64, outSize)

	sensorNodes := make([]*neat.Node, len(n.DNA.SensorNodes))
	for _, nodeI := range n.DNA.SensorNodes {
		node := n.Nodes[nodeI]
		label, err := strconv.Atoi(node.Label)
		if err != nil {
			panic(fmt.Sprintf("Node had non-integer label: %s\n", node.Label))
		}
		sensorNodes[label] = node
	}

	outputNodes := make([]*neat.Node, len(n.DNA.OutputNodes))
	for _, nodeI := range n.DNA.OutputNodes {
		node := n.Nodes[nodeI]
		label, err := strconv.Atoi(node.Label)
		if err != nil {
			panic(fmt.Sprintf("Node had non-integer label: %s\n", node.Label))
		}
		outputNodes[label-len(sensorNodes)] = node
	}

	indices := make([]int, len(dimensions))
	for i := 0; i < outSize; i += 1 {
		// Current values for each index
		dimensionalInputs := make([]float64, len(dimensions))

		// Can't have an arbitrarily-nested for loop, so unroll indices here
		indices[0] += 1
		for j := 0; j < len(dimensions)-1; j += 1 {
			// Don't need to actually check the last dimension since it can't roll over
			if indices[j] >= dimensions[j] {
				indices[j] = 0
				indices[j+1] += 1
			}

			// Scale input to range [-1,1]
			dimensionalInputs[j] = 2*float64(indices[j])/float64(dimensions[j]) - 1.0
		}

		computedInputs := make([]float64, 0)
		if otherInputs != nil {
			computedInputs = make([]float64, len(otherInputs))

			for j, fn := range otherInputs {
				computedInputs[j] = fn(dimensionalInputs...)
			}
		}

		inputs := append(dimensionalInputs, computedInputs...)
		if n.DNA.UsesBias {
			inputs = append([]float64{1}, inputs...)
		}

		log.Book(fmt.Sprintf("%d: inputs:%v, #sensors: %d, #outputs: %d\n", i, inputs, len(sensorNodes), len(outputNodes)), log.DEBUG, log.DEBUG_EXPERIMENT)
		n.Activate(inputs, sensorNodes, outputNodes)

		outMatrixEntry := make([]float64, len(outputNodes))

		for j, outputNode := range outputNodes {
			if math.IsNaN(outputNode.Value()) {
				log.Book(n.ToString(), log.DEBUG)
				panic("NaN network")
			}

			outMatrixEntry[j] = outputNode.Value()
		}

		outMatrix[i] = outMatrixEntry
	}

	return outMatrix
}

func NoiseFitness(o ma.Organism) float64 {
	n := o.(*neat.Network)
	n.Compile()

	const (
		w = 32
		h = 32
	)

	networkOutput := ActivateNetwork(n, []int{w, h}, nil)
	log.Book(fmt.Sprintf("Recieved network output, shape %dx%d\n", len(networkOutput), len(networkOutput[0])), log.DEBUG, log.DEBUG_EXPERIMENT)

	fitness := float64(0)
	usedColors := make(map[string]bool)

	for y := 0; y < h; y += 1 {
		for x := 0; x < w; x += 1 {
			pix := networkOutput[x+w*y]
			r := uint8(math.Floor(16 * pix[0]))
			g := uint8(math.Floor(16 * pix[1]))
			b := uint8(math.Floor(16 * pix[2]))
			colorString := fmt.Sprintf("%X.%X.%X", r, g, b)
			if _, ok := usedColors[colorString]; !ok {
				usedColors[colorString] = true
				fitness += 1
			}
		}
	}

	return fitness
}

func DrawNoiseNetwork(o ma.Organism, fName string) error {
	n := o.(*neat.Network)
	n.Compile()

	networkOutput := ActivateNetwork(n, []int{32, 32}, nil)
	return DrawNoiseImage(networkOutput, fName)
}

func DrawNoiseImage(networkOutput [][]float64, fName string) error {
	const (
		w = 32
		h = 32
	)
	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})

	for y := 0; y < h; y += 1 {
		for x := 0; x < w; x += 1 {
			pix := networkOutput[x+w*y]
			img.Set(x, y, color.RGBA{uint8(pix[0] * 16), uint8(pix[1] * 16), uint8(pix[2] * 16), 0xff})
		}
	}

	f, err := os.Create(fName)

	if err != nil {
		return err
	}

	err = png.Encode(f, img)

	if err != nil {
		return err
	}

	return nil
}

func NoiseEvolution() {
	popConfig := config.PopulationDefault()
	popConfig.Size = 64
	popConfig.DistanceThreshold = 3
	popConfig.DistanceThresholdEpsilon = 0.1
	popConfig.TargetMinSpecies = 7
	popConfig.TargetMaxSpecies = 13
	popConfig.RecombinationPercent = 0.8
	popConfig.LocalSearchGenerations = 8

	cppnConfig := config.CPPNDefault()
	cppnConfig.SensorNodes = 2
	cppnConfig.OutputNodes = 3
	cppnConfig.MinWeight = -15
	cppnConfig.MaxWeight = 15
	cppnConfig.MutationRatios = map[ma.MutationType]float64{
		neat.MutationAddConnection:   0.2,
		neat.MutationAddNode:         0.1,
		neat.MutationMutateWeights:   0.6,
		neat.MutationChangeAFunction: 0.1,
	}

	Evolution(NoiseFitness, DrawNoiseNetwork, popConfig, cppnConfig)
}

// func calculateMandelbrotAt(x0, y0, scaleX, scaleY float64) uint8 {
// 	x0 = 2.47*x0/scaleX - 2.0
// 	y0 = 2.24*y0/scaleY - 1.12

// 	x := x0
// 	y := y0

// 	i := uint8(0)
// 	for (x*x+y*y <= 2*2) && (i < math.MaxUint8) {
// 		x, y = x*x-y*y+x0, 2*x*y+y0
// 		i += 1
// 	}

// 	return i
// }

func MandelbrotEvolution() {
	// const (
	// 	w = 25 //47
	// 	h = 22 //4
	// )

	// mandelbrotPixels := make([][]uint8, 224)
	// for y := 0; y < len(mandelbrotPixels); y += 1 {
	// 	mandelbrotPixels[y] = make([]uint8, 247)
	// 	for x := 0; x < len(mandelbrotPixels[y]); x += 1 {
	// 		mandelbrotPixels[y][x] = calculateMandelbrotAt(float64(x), float64(y), float64(w), float64(h))
	// 	}
	// }

	// evalImg := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})
	// for x := 0; x < 247; x += 1 {
	// 	for y := 0; y < 224; y += 1 {
	// 		value := float64(mandelbrotPixels[y][x])
	// 		r := uint8(16 * (int(value) % 16))
	// 		g := uint8(16 * (math.Floor(value / 16)))
	// 		b := uint8(value)
	// 		evalImg.Set(x, y, color.RGBA{r, g, b, 0xff})
	// 	}
	// }

	// f, _ := os.Create("cppn/drawn/mandelbrot.png")
	// png.Encode(f, evalImg)

	// drawMandelbrot := func(networkOutput [][]float64, w, h int, fName string) error {
	// 	n := *neat.NewNetwork(nil, nil)
	// 	n.Compile()

	// 	filename := ""

	// 	var (
	// 		bias *neat.Node
	// 		inX  *neat.Node
	// 		inY  *neat.Node
	// 		inD  *neat.Node
	// 	)
	// 	for _, nodeI := range n.DNA.SensorNodes {
	// 		node := n.Nodes[nodeI]
	// 		if node.Label == "0" {
	// 			bias = node
	// 		} else if node.Label == "1" {
	// 			inX = node
	// 		} else if node.Label == "2" {
	// 			inY = node
	// 		} else {
	// 			inD = node
	// 		}
	// 	}

	// 	out := n.Nodes[n.DNA.OutputNodes[0]]

	// 	var img *image.RGBA
	// 	if filename != "" {
	// 		img = image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})
	// 	}

	// 	mse := float64(0)
	// 	matches := float64(0)

	// 	maxTarget := math.Inf(-1)
	// 	minTarget := math.Inf(1)
	// 	maxOut := math.Inf(-1)
	// 	minOut := math.Inf(1)
	// 	const errorEpsilon = 0.1

	// 	for x := 0; x < w; x += 1 {
	// 		for y := 0; y < h; y += 1 {
	// 			xs := float64(x)/w - 0.5
	// 			ys := float64(y)/h - 0.5
	// 			d := math.Sqrt(xs*xs + ys*ys)
	// 			inputValues := []float64{1, xs, ys, d}
	// 			n.Activate(inputValues, []*neat.Node{bias, inX, inY, inD}, []*neat.Node{out})

	// 			value := math.Floor(128 * (out.Value() + 1))

	// 			if math.IsNaN(value) {
	// 				log.Book(n.ToString(), log.DEBUG)
	// 				log.Book(n.DNA.ToString(), log.DEBUG)
	// 				log.Book(fmt.Sprintf("Inputs: {x:%.4g y:%.4g d:%.4g}\n", xs, ys, d), log.DEBUG)
	// 				panic("NaN network")
	// 			}

	// 			r := uint8(16 * (int(value) % 16))
	// 			g := uint8(16 * (math.Floor(value / 16)))
	// 			b := uint8(value)

	// 			if filename != "" {
	// 				img.Set(x, y, color.RGBA{r, g, b, 0xff})
	// 			}

	// 			outScaled := value / 255.0
	// 			targetScaled := float64(mandelbrotPixels[y][x]) / 255.0

	// 			if outScaled > maxOut {
	// 				maxOut = outScaled
	// 			}
	// 			if outScaled < minOut {
	// 				minOut = outScaled
	// 			}

	// 			if targetScaled > maxTarget {
	// 				maxTarget = targetScaled
	// 			}
	// 			if targetScaled < minTarget {
	// 				minTarget = targetScaled
	// 			}

	// 			squaredError := (outScaled - targetScaled) * (outScaled - targetScaled)
	// 			mse += squaredError

	// 			if squaredError <= errorEpsilon {
	// 				matches += 1
	// 			}
	// 		}
	// 	}

	// 	mse /= w * h
	// 	matches /= w * h
	// 	da := 1 - ((maxTarget-minTarget)-(maxOut-minOut))*((maxTarget-minTarget)-(maxOut-minOut))

	// 	fitness := 1 - mse + 2*matches + 3*da

	// 	if filename != "" {
	// 		f, _ := os.Create(filename)
	// 		png.Encode(f, img)
	// 	}

	// 	return fitness
	// }

	// mandelbrotFitness := func(o ma.Organism) float64 {
	// 	return 0
	// }

	// popConfig := config.PopulationDefault()
	// cppnConfig := config.CPPNDefault()
	// Evolution(mandelbrotFitness, drawMandelbrot, popConfig, cppnConfig)
}
