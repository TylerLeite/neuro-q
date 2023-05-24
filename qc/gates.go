package qc

import (
	"errors"
)

type Gate [][]complex128

// Gate definitions

func NewIdentity() Gate {
	return [][]complex128{
		{1, 0},
		{0, 1},
	}
}

func NewHadamard() Gate {
	return [][]complex128{
		{ONE_OVER_ROOT_2, ONE_OVER_ROOT_2},
		{ONE_OVER_ROOT_2, -ONE_OVER_ROOT_2},
	}
}

func NewT() Gate {
	return [][]complex128{
		{1, 0},
		{0, E_TO_THE_I_PI_OVER_4},
	}
}

func NewTDagger() Gate {
	return [][]complex128{
		{1, 0},
		{0, E_TO_THE_MINUS_I_PI_OVER_4},
	}
}

func NewCNOT() Gate {
	return [][]complex128{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 0, 1},
		{0, 0, 1, 0},
	}
}

func NewSqrtNot() Gate {
	return [][]complex128{
		{E_TO_THE_I_PI_OVER_4_BAR, E_TO_THE_MINUS_I_PI_OVER_4_BAR},
		{E_TO_THE_MINUS_I_PI_OVER_4_BAR, E_TO_THE_I_PI_OVER_4_BAR},
	}
}

// Gate Operations

func MatMul(b, a Gate) (Gate, error) {
	sza := len(a)
	szb := len(b)

	if sza != szb {
		return a, errors.New("gate sizes do not match")
	}

	out := make(Gate, sza)
	for i := 0; i < sza; i++ {
		out[i] = make([]complex128, sza)
	}

	for y := 0; y < sza; y++ {
		if len(a[y]) != sza {
			return a, errors.New("gate matrix is not square")
		}
		if len(b[y]) != sza {
			return a, errors.New("gate matrix is not square")
		}

		for x := 0; x < sza; x++ {
			out[y][x] = 0 // probably unnecessary

			for n := 0; n < sza; n++ {
				out[y][x] += a[y][n] * b[n][x]
			}
		}
	}

	return out, nil
}

func Combine(gates ...Gate) (Gate, error) {
	var out Gate
	for i, gate := range gates {
		if i == 0 {
			out = gate
		} else {
			tmp, err := MatMul(out, gate)
			if err != nil {
				return tmp, err
			}

			out = tmp
		}
	}

	return out, nil
}

func KroneckerProduct(a, b Gate) (Gate, error) {
	sza := len(a)
	szb := len(b)

	sz := sza * szb
	out := make(Gate, sz)
	for i := 0; i < sz; i++ {
		out[i] = make([]complex128, sz)
	}

	for y := 0; y < sza; y++ {
		if len(a[y]) != sza {
			return a, errors.New("gate matrix is not square")
		}

		for x := 0; x < sza; x++ {
			for j := 0; j < szb; j++ {
				if len(b[j]) != szb {
					return a, errors.New("matrix is not square")
				}

				for i := 0; i < szb; i++ {
					out[y*szb+j][x*szb+i] = a[y][x] * b[j][i]
				}
			}
		}
	}

	return out, nil
}
