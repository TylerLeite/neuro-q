package qc

type Tensor []complex128

// Qubit definitions

func NewZero() Tensor {
	qubit := make(Tensor, 2)
	qubit[0] = 1
	qubit[1] = 0
	return Tensor(qubit)
}

func NewOne() Tensor {
	qubit := make(Tensor, 2)
	qubit[0] = 0
	qubit[1] = 1
	return Tensor(qubit)
}

func NewPlus() Tensor {
	qubit := make(Tensor, 2)
	qubit[0] = ONE_OVER_ROOT_2
	qubit[1] = ONE_OVER_ROOT_2
	return Tensor(qubit)
}

func NewMinus() Tensor {
	qubit := make(Tensor, 2)
	qubit[0] = ONE_OVER_ROOT_2
	qubit[1] = -ONE_OVER_ROOT_2
	return Tensor(qubit)
}

// Qubit operations

func TensorProduct(l, r Tensor) Tensor {
	szl := len(l)
	szr := len(r)
	out := make(Tensor, szl*szr)

	for i := 0; i < szl; i++ { // l
		for j := 0; j < szr; j++ { // r
			out[i*szr+j] = l[i] * r[j]
		}
	}

	return out
}
