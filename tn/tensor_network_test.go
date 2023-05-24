package tn

import "testing"

func TestConstruction(t *testing.T) {
	q0 := NewVertex()
	q0.UpperIndices = []int{0}
	q0.Value = []complex128{1, 0}

	q1 := NewVertex()
	q1.UpperIndices = []int{1}
	q1.Value = []complex128{0, 1}

	q2 := NewVertex()
	q2.UpperIndices = []int{2}
	q2.Value = []complex128{1, 0}

	// CNOT gate
	cn1 := NewVertex()
	cn1.LowerIndices = []int{1, 2}
	cn1.UpperIndices = []int{1, 2}
	cn1.Value = []complex128{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0,
	}

	// CNOT gate
	cn2 := NewVertex()
	cn2.LowerIndices = []int{0, 1}
	cn2.UpperIndices = []int{0, 1}
	cn2.Value = []complex128{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0,
	}

	// X gate
	x := NewVertex()
	x.LowerIndices = []int{0}
	x.UpperIndices = []int{0}
	x.Value = []complex128{0, 1, 1, 0}

	// t.Fatalf(``, ...)
}
