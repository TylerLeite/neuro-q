package cppn

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"testing"

	"github.com/TylerLeite/neuro-q/neat"
)

func Activate(inputs []float64, sensors, outputs []*neat.Node, allNodes []*neat.Node) error {
	for _, node := range allNodes {
		node.Reset()
	}

	done := false
	sanity := 100
	for !done && sanity > 0 {
		for _, node := range allNodes {
			node.Deactivate()
		}

		for i, in := range sensors {
			in.SetDefaultValue(inputs[i])
			in.ForwardPropogate()

			// Want to be able to revisit nodes once you have a new input
			for _, node := range allNodes {
				node.Deactivate()
			}
		}

		done = true
		for _, out := range outputs {
			if math.IsNaN(out.Value()) {
				done = false
			}
		}
		sanity -= 1
	}

	if sanity <= 0 {
		panic("Canceling activation, too many loops in the network")
	}

	return nil
}

func TestKnown(t *testing.T) {
	xIn := neat.NewNode(neat.IdentityFunc, neat.SensorNode)
	yIn := neat.NewNode(neat.IdentityFunc, neat.SensorNode)

	rOut := neat.NewNode(neat.IdentityFunc, neat.OutputNode)
	gOut := neat.NewNode(neat.IdentityFunc, neat.OutputNode)
	bOut := neat.NewNode(neat.IdentityFunc, neat.OutputNode)

	xIn.AddChild(rOut)
	yIn.AddChild(gOut)

	xIn.AddChild(bOut)
	yIn.AddChild(bOut)

	const (
		w = 160
		h = 90
	)

	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})

	for x := float64(0); x < w; x += 1 {
		for y := float64(0); y < h; y += 1 {
			Activate(
				[]float64{x / w, y / h},
				[]*neat.Node{xIn, yIn},
				[]*neat.Node{rOut, gOut, bOut},
				[]*neat.Node{xIn, yIn, rOut, gOut, bOut},
			)

			r := 255 * rOut.Value()
			g := 255 * gOut.Value()
			b := 255 * bOut.Value()

			img.Set(int(x), int(y), color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}
	}

	f, _ := os.Create("_known.png")
	png.Encode(f, img)
}

func TestRandom(t *testing.T) {
	neat.Seed(3)

	f01, _ := neat.RandomFunc()
	f02, _ := neat.RandomFunc()
	xIn := neat.NewNode(f01, neat.SensorNode)
	yIn := neat.NewNode(f02, neat.SensorNode)

	f11, _ := neat.RandomFunc()
	f12, _ := neat.RandomFunc()
	f13, _ := neat.RandomFunc()
	inner1 := neat.NewNode(f11, neat.HiddenNode)
	inner2 := neat.NewNode(f12, neat.HiddenNode)
	inner3 := neat.NewNode(f13, neat.HiddenNode)

	xIn.AddChild(inner1)
	yIn.AddChild(inner2)

	xIn.AddChild(inner3)
	yIn.AddChild(inner3)

	f21, _ := neat.RandomFunc()
	f22, _ := neat.RandomFunc()
	f23, _ := neat.RandomFunc()
	rOut := neat.NewNode(f21, neat.OutputNode)
	gOut := neat.NewNode(f22, neat.OutputNode)
	bOut := neat.NewNode(f23, neat.OutputNode)

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
			Activate(
				[]float64{x / w, y / h},
				[]*neat.Node{xIn, yIn},
				[]*neat.Node{rOut, gOut, bOut},
				[]*neat.Node{xIn, yIn, inner1, inner2, inner3, rOut, gOut, bOut},
			)

			r := 255 * rOut.Value()
			g := 255 * gOut.Value()
			b := 255 * bOut.Value()

			img.Set(int(x), int(y), color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}
	}

	f, _ := os.Create("_random.png")
	png.Encode(f, img)
}

// // Because fitness function is defined on populations and crossover is defined on organisms, need a reference here
// // TODO: maybe add crossover as a function member of population like fitness is?

func TestGeneration(t *testing.T) {
	neat.Seed(11)

	allNodes := make([]*neat.Node, 0)

	xIn := neat.NewNode(neat.IdentityFunc, neat.SensorNode)
	yIn := neat.NewNode(neat.IdentityFunc, neat.SensorNode)

	allNodes = append(allNodes, xIn)
	allNodes = append(allNodes, yIn)

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
			fn, _ := neat.RandomFunc()
			node := neat.NewNode(fn, neat.HiddenNode)
			layers[layerIdx][nodeIdx] = node
			allNodes = append(allNodes, node)

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

	rOut := neat.NewNode(neat.AbsFunc, neat.OutputNode)
	gOut := neat.NewNode(neat.AbsFunc, neat.OutputNode)
	bOut := neat.NewNode(neat.AbsFunc, neat.OutputNode)

	allNodes = append(allNodes, rOut)
	allNodes = append(allNodes, gOut)
	allNodes = append(allNodes, bOut)

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

			Activate(
				[]float64{x / w, y / h},
				[]*neat.Node{xIn, yIn},
				[]*neat.Node{rOut, gOut, bOut},
				allNodes,
			)

			r := 255 * rOut.Value()
			g := 255 * gOut.Value()
			b := 255 * bOut.Value()

			xIn.Reset()
			yIn.Reset()

			// TODO: reset layers

			rOut.Reset()
			gOut.Reset()
			bOut.Reset()

			img.Set(int(x), int(y), color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}
	}

	f, _ := os.Create("_generated.png")
	png.Encode(f, img)
}

func clamp(n float64) uint8 {
	if n < 0 {
		return 0
	} else if n > 255 {
		return 255
	} else {
		return uint8(n)
	}
}

func TestMassive(t *testing.T) {
	genome := neat.NewGenome(2, 3, true, -15, 15)

	for i := 0; i < 10; i += 1 {
		genome.AddNode()
	}

	for i := 0; i < 20; i += 1 {
		genome.AddConnection(false)
	}

	nodes := make(map[uint]bool)
	for _, edgeGene := range genome.Connections {
		nodes[edgeGene.InNode] = true
		nodes[edgeGene.OutNode] = true
	}

	genome.ActivationFunctions = make(map[uint]string)
	for nodeId := range nodes {
		_, fnName := neat.RandomFunc()
		genome.ActivationFunctions[nodeId] = fnName
	}

	n := neat.NewNetwork(genome, nil)
	n.Draw("massive_network.bmp")

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

	// Output as an image
	const (
		w = 32
		h = 32
	)

	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{w, h}})

	scale := 1.0

	for x := float64(0); x < w; x += 1 {
		for y := float64(0); y < h; y += 1 {

			err := n.Activate(
				[]float64{1, scale * (x - w/2), scale * (y - h/2)},
				[]*neat.Node{bias, inX, inY},
				[]*neat.Node{outR, outG, outB},
			)

			if err != nil {
				panic(err)
			}

			_r := outR.Value()
			_g := outG.Value()
			_b := outB.Value()
			r := clamp(256 * neat.GaussianFunc(_r))
			g := clamp(256 * neat.StepFunc(_g))
			b := clamp(256 * neat.NEATSigmoidFunc(_b))

			img.Set(int(x), int(y), color.RGBA{r, g, b, 0xff})
			// img.Set(int(x), int(y), color.RGBA{0xee, 0x11, 0x22, 0xff})

			// if int(x)%int(w/10) == 0 && int(y)%int(h/10) == 0 {
			fmt.Printf("(%d,%d): %.2g,%.2g,%.2g -> %d,%d,%d\n", int(x), int(y), _r, _g, _b, r, g, b)
			// }
		}
	}

	f, _ := os.Create("massive_generated.png")
	png.Encode(f, img)
}
