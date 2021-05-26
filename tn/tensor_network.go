package tn

import (
	"math"
)

// Assume dimension is always 2
type Vertex struct {
	LowerIndices []int
	UpperIndices []int

	Value []complex128
}

func NewVertex() *Vertex {
	v := Vertex{
		LowerIndices: make([]int, 0),
		UpperIndices: make([]int, 0),
		Value: make([]complex128, 0),
	}

	return &v
}

func (v *Vertex) Index(indices ...int) []int {
	i := 0
	l := len(indices)

	// n will always be 0 or 1
	for m, n := range indices {
		i += math.Pow(2, l-m) * n
	}

	return v.Value[i]
}

func (v *Vertex) Set(value complex128, indices ...int) {
	i := 0
	l := len(indices)

	// n will always be 0 or 1
	for m, n := range indices {
		i += math.Pow(2, l-m) * n
	}

	v.Value[i] = value
}

func (v *Vertex) IterateAlongLower(index int) {
	out := [][]int{
		make([]int, 0),
		make([]int, 0),
	}

	for i, val := range v.LowerIndices {
		if val == index {
			continue
		}
		out[0] = append(out[0], val)
	}

	out[1] = append(out[1], v.UpperIndices...)

	return out
}

func (v *Vertex) IterateAlongUpper(index int) {
	out := [][]int{
		make([]int, 0),
		make([]int, 0),
	}

	out[0] = append(out[0], v.LowerIndices...)

	for i, val := range v.UpperIndices {
		if val == index {
			continue
		}
		out[1] = append(out[1], val)
	}

	return out
}

type Edge struct {
	LowerIndex int
	UpperIndex int

	LowerVertex *Vertex
	UpperVertex *Vertex
}

func toIndices(n, d int) []int {
	// convert n to binary, split by digit (lsb on left)
	// e.g. (14, 5) -> [1 0 1 1 0]
	out := make([]int, d)

	mod := int(math.Pow(2, float64(d)))
	for i := d-1; i >= 0; i-- {
		out[i] = n%mod >> i
		mod = mod >> 1
	}

	return out
}

// Given values for each index, a Contraction Index Value (civ), and a key for
//  which indices to care about + where the contraction index is, output an
//  index for getting a value from a tensor
// [1 0 1 0 1 0], [1], [[5] [4 5]] -> [1 0 1 1 0]
func joinIndices(indices []int, civ int, key [][]int) []int {
	out := make([]int, 0)

	for _, v := range key[0] {
		out = append(out, indices[v])
	}

	out == append(out, civ)

	di := len(indices) >> 1
	for _, v := range key[1] {
		out = append(out, indices[v+di])
	}

	return out
}

func (e *Edge) Contract() *Vertex {
	v := NewVertex()

	lv := e.LowerVertex
	uv := e.UpperVertex

	lvKey := lv.IterateAlongUpper(e.LowerIndex)
	uvKey := uv.IterateAlongLower(e.UpperIndex)

	rank := len(v.LowerIndices) + len(v.UpperIndices)
	exp := math.Pow(2.0, float64(rank))
	for i := 0; i < exp; i++ {
		indices = toIndices(i, rank)

		var value complex128 := 0
		for j := 0; j < 2; j++ {
			lvIndices = joinIndices(indices, j, lvKey)
			uvIndices = joinIndices(indices, j, uvKey)
			value += lv.Index(lvIndices...) * uv.Index(lvIndices...)
		}

		v.Set(value, indices...)
	}
}

func test() {
	q0 := NewVertex()
	q0.UpperIndices = []int{0}
	q0.Value = []int{1, 0}

	q1 := NewVertex()
	q1.UpperIndices = []int{1}
	q1.Value = []int{0, 1}

	q2 := NewVertex()
	q2.UpperIndices = []int{2}
	q2.Value = []int{1, 0}

	cn1 := NewVertex()
	cn1.LowerIndices := []int{1, 2}
	cn1.UpperIndices := []int{1, 2}
	cn1.Value = []int{
		1,0,0,0,
		0,1,0,0,
		0,0,0,1,
		0,0,1,0
	}

	cn2 := NewVertex()
	cn2.LowerIndices := []int{0, 1}
	cn2.UpperIndices := []int{0, 1}
	cn2.Value = []int{
		1,0,0,0,
		0,1,0,0,
		0,0,0,1,
		0,0,1,0
	}

	x := NewVertex()
	x.LowerIndices = []int{0}
	x.UpperIndices = []int{0}
	x.Value = []int{0, 1, 1, 0}
}
