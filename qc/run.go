package qc

import (
	"errors"
)

func Operate(gate Gate, tensor Tensor) (Tensor, error) {
	sz := len(tensor)
	out := make(Tensor, sz)

	if len(gate) != sz {
		return out, errors.New("Gate does not have the proper dimensions")
	}

	for y := 0; y < sz; y++ {
		row := gate[y]
		if len(row) != sz {
			return out, errors.New("Gate does not have the proper dimensions")
		}

		for x := 0; x < sz; x++ {
			out[y] += tensor[x] * gate[y][x]
		}
	}

	return out, nil
}
