package ma

import (
	"math/rand"

	"github.com/TylerLeite/neuro-q/qc"
)

type Gene byte

func (code Gene) ToGate(nQubits int) qc.Gate {
	var typ byte = code & 1
	var qr byte = code >> 4

	iden := qc.NewIdentity()
	had := qc.NewHadamard()
	t := qc.NewT()
	cnot := qc.NewCNOT()

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
		if qr == i {
			out = qc.KroneckerProduct(out, gateType)
			if typ == 1 {
				i += 1
			}
		} else {
			out = qc.KroneckerProduct(out, iden)
		}
	}
}

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

	var out byte = qr>>4 + nGates>>1 + gateType
	return Gene(out)
}

type Genome []Gene

func (dna Genome) Compile(nQubits int) qc.Gate {
	gate = ToGate(dna[0], nQubits)
	for i, gene := range dna {
		if i == 0 {
			continue
		}

		gate = qc.MatMul(ToGate(gene, nQubits))
	}

	return gate
}

func Crossover(a, b Genome) Genome {
	ai := rand.Intn(len(a))
	bi := rand.Intn(len(b))

	out := append(a[:ai], b[bi:])
	return Genome(out)
}
