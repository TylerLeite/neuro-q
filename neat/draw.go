package neat

import (
	"bytes"
	_ "embed"
	"image"
	"image/color"
	"math"
	"os"

	"golang.org/x/image/bmp"
)

//go:embed font.bmp
var fontImage []byte

var alphabet = map[string][2]int{
	"a": {0, 0},
	"b": {3, 0},
	"c": {6, 0},
	"d": {9, 0},
	"e": {12, 0},
	"f": {15, 0},
	"g": {18, 0},
	"h": {21, 0},
	"i": {0, 5},
	"j": {3, 5},
	"k": {6, 5},
	"l": {9, 5},
	"m": {12, 5},
	"n": {15, 5},
	"o": {18, 5},
	"0": {18, 5},
	"p": {21, 5},
	"q": {0, 10},
	"r": {3, 10},
	"s": {6, 10},
	"5": {6, 10},
	"t": {9, 10},
	"u": {12, 10},
	"v": {15, 10},
	"w": {18, 10},
	"x": {21, 10},
	"y": {0, 15},
	"z": {3, 15},
	"1": {6, 15},
	"2": {9, 15},
	"3": {12, 15},
	"4": {15, 15},
	"6": {18, 15},
	"7": {21, 15},
	"8": {0, 20},
	"9": {3, 20},
	".": {6, 20},
	"!": {9, 20},
	"?": {12, 20},
	"-": {15, 0},
	"(": {18, 0},
	")": {21, 0},
}

func drawNode(x, y int, label string, font image.Image, canvas *image.RGBA) {
	for i := 0; i < 15; i += 1 {
		for j := 0; j < 9; j += 1 {
			if (j == 0 || j == 8) || (i == 0 || i == 14) {
				canvas.Set(x+i, y+j, color.RGBA{R: math.MaxUint8, G: 0, B: 0, A: math.MaxUint8})
			} else {
				canvas.Set(x+i, y+j, color.RGBA{R: math.MaxUint8, G: math.MaxUint8, B: math.MaxUint8, A: math.MaxUint8})
			}
		}
	}

	for ci, c := range label {
		i, j := alphabet[string(c)][0], alphabet[string(c)][1]
		for di := 0; di < 3; di += 1 {
			for dj := 0; dj < 5; dj += 1 {
				r, _, _, _ := font.At(i+di, j+dj).RGBA()
				if r > 0 {
					canvas.Set(x+2+di+4*ci, y+2+dj, color.RGBA{R: math.MaxUint8, G: 0, B: 0, A: math.MaxUint8})
				}
			}
		}
	}
}

func drawEdge(x0, y0, x1, y1 int, label string, weight float64, font image.Image, canvas *image.RGBA) {
	x := float64(x0)
	y := float64(y0)

	slope := float64(y1-y0) / math.Abs(float64(x1-x0))

	dx := float64(1)
	if x0 > x1 {
		dx = float64(-1)
	}

	for {
		x += dx
		y += slope

		if dx > 0 && x > float64(x1) || dx < 0 && x < float64(x1) {
			break
		}
		if slope > 0 && y > float64(y1) || slope < 0 && y < float64(y1) {
			break
		}

		g := float64(math.MaxUint8) * weight
		b := g

		if g > 0 {
			g = 0
		} else {
			g = -g
		}

		if b < 0 {
			b = 0
		}

		canvas.Set(int(math.Round(x)), int(math.Round(y)), color.RGBA{R: 128, G: uint8(g), B: uint8(b), A: math.MaxUint8})
	}
}

func (n *Network) SeparateIntoLayers() [][]*Node {
	layers := make([][]*Node, 0)
	visited := make(map[*Node]bool)

	for _, node := range n.Nodes {
		visited[node] = false
	}

	inputLayer := make([]*Node, 0)
	for _, nodeId := range n.DNA.SensorNodes {
		sensorNode := n.Nodes[nodeId]
		visited[sensorNode] = true
		inputLayer = append(inputLayer, sensorNode)
	}
	layers = append(layers, inputLayer)

	index := 0
	for {
		thisLayer := layers[index]
		nextLayer := make([]*Node, 0)

		for _, node := range thisLayer {
			for _, outEdge := range node.Out {
				child := outEdge.Out
				if visited[child] {
					continue
				} else {
					visited[child] = true
					nextLayer = append(nextLayer, child)
				}
			}
		}

		if len(nextLayer) == 0 {
			break
		} else {
			layers = append(layers, nextLayer)
			index += 1
		}
	}

	return layers
}

func (n *Network) Draw(fName string) error {
	layers := n.SeparateIntoLayers()
	maxDepth := len(layers)

	// Oddly enough, width is how tall the graph is at a given column
	maxWidth := 0
	for _, layer := range layers {
		widthAtLayer := len(layer)
		if widthAtLayer > maxWidth {
			maxWidth = widthAtLayer
		}
	}

	// Nodes are 15x9, padding 14x5
	canvasWidth := 2 + 15*maxDepth + 14*(maxDepth-1)
	canvasHeight := 2 + 9*maxWidth + 5*(maxWidth-1)
	canvas := image.NewRGBA(image.Rect(0, 0, canvasWidth, canvasHeight))

	// Calculate y positions of each node
	rowLocation := make(map[*Node]int)
	columnLocation := make(map[*Node]int)
	for xi, layer := range layers {
		for yi, node := range layer {
			rowLocation[node] = xi
			columnLocation[node] = yi
		}
	}

	// Get the position in the "grid" of a given node
	nodePosition := func(node *Node) (int, int) {
		X := rowLocation[node]
		Y := columnLocation[node]

		x := X*29 + 1
		y := canvasHeight - len(layers[X])*9 - (len(layers[X])-1)*5 - 1 + Y*14
		y -= (maxWidth - len(layers[X])) * 7

		return x, y
	}

	// Will need to draw text, have a custom font
	fontReader := bytes.NewReader(fontImage)
	font, err := bmp.Decode(fontReader)
	if err != nil {
		return err
	}

	// Draw edges first so they don't interfere with reading node labels
	for _, edge := range n.Edges {

		x0, y0 := nodePosition(edge.In)
		x0 += 7
		y0 += 7
		x1, y1 := nodePosition(edge.Out)
		x1 += 5
		y1 += 5

		drawEdge(x0, y0, x1, y1, edge.Label, edge.Weight, font, canvas)
	}

	// Draw columns of nodes
	for _, node := range n.Nodes {
		xOffset, yOffset := nodePosition(node)

		drawNode(xOffset, yOffset, node.Label, font, canvas)
	}

	file, _ := os.Create(fName)
	defer file.Close()

	return bmp.Encode(file, canvas)
}
