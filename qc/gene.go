package qc

import (
	"math/rand"
)

type Gene byte

func RandomGene() Gene {
	qr := rand.Intn(16)
	gateType := rand.Intn(2)

	if qr == 15 && gateType == 1 {
		gateType = 0
	}

	nGates := 0
	if gateType == 0 && rand.Intn(2) == 0 {
		nGates = rand.Intn(8)
	}

	// var out byte = qr>>4 + nGates>>1 + gateType
	out := qr>>4 + nGates>>1 + gateType
	return Gene(out)
}

func (code Gene) ToGate(nQubits int) Gate {
	var typ byte = byte(code) & 1
	var qr byte = byte(code) >> 4

	iden := NewIdentity()
	had := NewHadamard()
	t := NewT()
	cnot := NewCNOT()

	out := iden
	gateType := t

	if typ == 0 {
		nGates := (code >> 1) & 0x7

		if nGates == 0 {
			gateType = had
		} else {
			gateType = t
		}
	} else {
		gateType = cnot
	}

	i := 1
	if qr == 0 {
		out = gateType
	}

	for ; i < nQubits; i++ {
		if int(qr) == i {
			out, _ = KroneckerProduct(out, gateType)
			if typ == 1 {
				i += 1
			}
		} else {
			out, _ = KroneckerProduct(out, iden)
		}
	}

	return out
}

type Genome []Gene

func Crossover(a, b Genome) Genome {
	ai := rand.Intn(len(a))
	bi := rand.Intn(len(b))

	out := append(a[:ai], b[bi:]...)
	return Genome(out)
}

func Compile(dna Genome, nQubits int) Gate {
	gate := dna[0].ToGate(nQubits)
	for i, gene := range dna {
		if i == 0 {
			continue
		}

		gate, _ = MatMul(gate, gene.ToGate(nQubits))
	}

	return gate
}
