package cppn

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"

	"github.com/TylerLeite/neuro-q/neat"
)

func TestImage(t *testing.T) {
	Seed(5)

	f01, _ := RandomFunc()
	f02, _ := RandomFunc()
	xIn := neat.NewNode(f01)
	yIn := neat.NewNode(f02)

	f11, _ := RandomFunc()
	f12, _ := RandomFunc()
	f13, _ := RandomFunc()
	inner1 := neat.NewNode(f11)
	inner2 := neat.NewNode(f12)
	inner3 := neat.NewNode(f13)

	xIn.AddChild(inner1)
	yIn.AddChild(inner2)

	xIn.AddChild(inner3)
	yIn.AddChild(inner3)

	f21, _ := RandomFunc()
	f22, _ := RandomFunc()
	f23, _ := RandomFunc()
	rOut := neat.NewNode(f21)
	gOut := neat.NewNode(f22)
	bOut := neat.NewNode(f23)

	inner1.AddChild(rOut)
	inner2.AddChild(gOut)
	inner3.AddChild(bOut)

	const (
		w = 160
		h = 90
	)

	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})

	for x := float64(0); x < w; x += 1 {
		for y := float64(0); y < h; y += 1 {
			xIn.SetDefaultValue(x / w)
			yIn.SetDefaultValue(y / h)

			r := 255 * rOut.CalculateValue()
			g := 255 * gOut.CalculateValue()
			b := 255 * bOut.CalculateValue()

			rOut.Reset()
			gOut.Reset()
			bOut.Reset()

			img.Set(int(x), int(y), color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}
	}

	f, _ := os.Create("_random.png")
	png.Encode(f, img)
}

func TestGeneration(t *testing.T) {
	xIn := neat.NewNode(IdentityFunc)
	yIn := neat.NewNode(IdentityFunc)

	const layerNum = 6
	layers := make([][]*neat.Node, layerNum)

	// Layer 0 is input nodes
	layers[0] = make([]*neat.Node, 2)
	layers[0][0] = xIn
	layers[0][1] = yIn

	for layerIdx := 1; layerIdx < layerNum; layerIdx += 1 {
		dieRolls := []int{5, 3, 6, 7, 4, 6}
		nodesInLayer := dieRolls[layerIdx]
		// To be random:
		// nodesInLayer := rand.Intn(8) + 1
		layers[layerIdx] = make([]*neat.Node, nodesInLayer)

		for nodeIdx := 0; nodeIdx < nodesInLayer; nodeIdx += 1 {
			fn, _ := RandomFunc()
			node := neat.NewNode(fn)
			layers[layerIdx][nodeIdx] = node

			// Try to find a parent for this new node in the previous layer
			// If you don't go up a layer, and so on
			// If you totally fail, pick one od the input nodes randomly to be the parent
			foundAtLeastOneParent := false
			workingLayer := layerIdx - 1
			for !foundAtLeastOneParent {
				for i := 0; i < len(layers[workingLayer]); i += 1 {
					if (i+layerIdx+nodeIdx)%8 == 0 {
						// To be random:
						// if rand.Intn(len(layers[workingLayer])) == 0 {
						layers[workingLayer][i].AddChild(node)
						foundAtLeastOneParent = true
					}
				}

				if !foundAtLeastOneParent {
					workingLayer -= 1
				}

				if workingLayer < 0 {
					// Last resort, pick a random input node
					foundAtLeastOneParent = true

					if nodeIdx%2 == 0 {
						// To be random:
						// if rand.Intn(2) == 0 {
						xIn.AddChild(node)
					} else {
						yIn.AddChild(node)
					}
				}
			}
		}
	}

	rOut := neat.NewNode(AbsFunc)
	gOut := neat.NewNode(AbsFunc)
	bOut := neat.NewNode(AbsFunc)

	lastLayerIndices := []int{4, 2, 5, 3, 0, 1}
	// To be random instead:
	// lastLayerIndices := make([]int, len(layers[layerNum-1]))
	// for i := 0; i < len(lastLayerIndices); i += 1 {
	// 	lastLayerIndices[i] = i
	// }
	// rand.Shuffle(
	// 	len(lastLayerIndices),
	// 	func(i, j int) {
	// 		lastLayerIndices[i], lastLayerIndices[j] = lastLayerIndices[j], lastLayerIndices[i]
	// 	},
	// )

	// For each node in the last layer, add one of the outputs as a child
	// Make sure each output is chosen at least once
	for i, v := range lastLayerIndices {
		currentNode := layers[layerNum-1][v]
		switch i {
		case 0:
			currentNode.AddChild(rOut)
		case 1:
			currentNode.AddChild(gOut)
		case 2:
			currentNode.AddChild(bOut)
		default:
			dieRolls := []int{2, 0, 1, 1, 2, 0, 2, 0}
			outI := dieRolls[i]
			// To be random instead:
			// outI := rand.Intn(3)
			switch outI {
			case 0:
				currentNode.AddChild(rOut)
			case 1:
				currentNode.AddChild(gOut)
			case 2:
				currentNode.AddChild(bOut)
			}
		}
	}

	// Output as an image
	const (
		w = 512
		h = 256
	)

	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})

	for x := float64(0); x < w; x += 1 {
		for y := float64(0); y < h; y += 1 {
			xIn.SetDefaultValue(x / w)
			yIn.SetDefaultValue(y / h)

			r := 255 * rOut.CalculateValue()
			g := 255 * gOut.CalculateValue()
			b := 255 * bOut.CalculateValue()

			rOut.Reset()
			gOut.Reset()
			bOut.Reset()

			img.Set(int(x), int(y), color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}
	}

	f, _ := os.Create("_generated.png")
	png.Encode(f, img)
}

func TestEvolution(t *testing.T) {
	// // firstGenerationMutations := make(map[string]uint)

	// // Create seed
	// // TODO: include seed file in population, read it here
	// seedGenome := Genome{}
	// seedGenome.SensorNodes = make([]uint, inNodes)
	// seedGenome.OutputNodes = make([]uint, outNodes)

	// // TODO: NewOrganism() in cppn
	// var seed ma.Organism = nil
	// seed.LoadGeneticCode(&seedGenome)

	// population := ma.NewPopulation(size, seed)
	// species := ma.NewSpecies(population)
	// species.Members = make([]ma.Organism, size)

	// for i := 0; i < size; i += 1 {
	// 	genome := seedGenome.Randomize()
	// 	organism := seed.NewFromGeneticCode(genome)
	// 	species.Members[i] = organism
	// }

	// return population.SeparateIntoSpecies()
}
