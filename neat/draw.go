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

func distanceToInput(n *Node, visited map[*Node]int) int {
	// Breadth-first search from each node
	if len(n.In) == 0 {
		visited[n] = 1
		return 1
	}

	if cachedDepth, ok := visited[n]; ok {
		return cachedDepth
	}

	maxDepth := 0
	for _, v := range n.In {
		depth := 1 + distanceToInput(v.In, visited)

		if depth > maxDepth {
			maxDepth = depth
		}
	}

	visited[n] = maxDepth
	return maxDepth
}

func (n *Network) Draw(fName string) error {
	_visited := make(map[*Node]int)

	// Figure out what size the image needs to be
	maxDepth := 0
	for _, nodeId := range n.DNA.OutputNodes {
		node := n.Nodes[nodeId]
		distance := distanceToInput(node, _visited)
		if maxDepth < distance {
			maxDepth = distance
		}
	}

	// Oddly enough, width is how tall the graph is at a given column
	widthAtDepth := make(map[int]int)
	maxWidth := 0
	for _, v := range _visited {
		if _, ok := widthAtDepth[v]; !ok {
			widthAtDepth[v] = 0
		}

		widthAtDepth[v] += 1
		if widthAtDepth[v] > maxWidth {
			maxWidth = widthAtDepth[v]
		}
	}

	// Will need to draw text, have a custom font
	fontReader := bytes.NewReader(fontImage)
	font, err := bmp.Decode(fontReader)
	if err != nil {
		return err
	}

	// Nodes are 15x9, padding 14x5
	canvasWidth := 2 + 15*maxDepth + 14*(maxDepth-1)
	canvasHeight := 2 + 9*maxWidth + 5*(maxWidth-1)
	canvas := image.NewRGBA(image.Rect(0, 0, canvasWidth, canvasHeight))

	// Calculate y positions of each node
	drawnAtDepth := make(map[int]int)
	columnLocation := make(map[*Node]int)
	for node, depth := range _visited {
		drawnSoFar, ok := drawnAtDepth[depth]
		if !ok {
			drawnAtDepth[depth] = 0
		}

		columnLocation[node] = drawnSoFar

		drawnAtDepth[depth] += 1
	}

	// TODO: functions to find x, y posiitons of nodes so there isn't a whole bunch of reduplicated spaghetti code
	// But this was all written by system 1 so idk how to think about organizing it

	// Draw edges first so they don't interfere with reading node labels
	for _, edge := range n.Edges {
		x0 := (_visited[edge.In]-1)*29 + 1 + 7
		y0 := canvasHeight - widthAtDepth[_visited[edge.In]]*9 - (widthAtDepth[_visited[edge.In]]-1)*5 - 1 + columnLocation[edge.In]*14 + 5
		y0 -= (maxWidth - widthAtDepth[_visited[edge.In]]) * 7

		x1 := (_visited[edge.Out]-1)*29 + 1 + 7
		y1 := canvasHeight - widthAtDepth[_visited[edge.Out]]*9 - (widthAtDepth[_visited[edge.Out]]-1)*5 - 1 + columnLocation[edge.Out]*14 + 5
		y1 -= (maxWidth - widthAtDepth[_visited[edge.Out]]) * 7

		drawEdge(x0, y0, x1, y1, edge.Label, edge.Weight, font, canvas)
	}

	// Draw columns of nodes
	drawnAtDepth = make(map[int]int) // Reset tracker for which nodes have been drawn
	for node, depth := range _visited {
		drawnSoFar, ok := drawnAtDepth[depth]
		if !ok {
			drawnAtDepth[depth] = 0
		}

		yOffset := canvasHeight - widthAtDepth[depth]*9 - (widthAtDepth[depth]-1)*5 - 1 + drawnSoFar*14
		yOffset -= (maxWidth - widthAtDepth[depth]) * 7
		drawNode((depth-1)*29+1, yOffset, node.Label, font, canvas)

		drawnAtDepth[depth] += 1
	}

	file, _ := os.Create(fName)
	defer file.Close()

	return bmp.Encode(file, canvas)
}
