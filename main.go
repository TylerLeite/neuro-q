package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TylerLeite/neuro-q/gp"
	"github.com/TylerLeite/neuro-q/qc"
)

// func test() {
// 	qc.TestAll()
// 	gp.TestAll()
// }

func compileSyntaxTree(t *gp.DerivationTree) qc.Gate {
	I := qc.NewIdentity()
	out := qc.NewIdentity()

	for i := 0; i < 7; i++ {
		out, _ = qc.KroneckerProduct(out, I)
	}

	return out
}

func main() {
	rand.Seed(time.Now().UnixNano())

	rules, symbolMap := gp.LoadRulesFromFile("./gp/grammars/clifford_plus_t.grmr")

	const N = 20
	var codons = make(gp.Genome, N)
	for i := 0; i < N; i++ {
		codons[i] = gp.Gene(rand.Intn(256))
	}

	out := gp.RunDerivationSequence(rules, symbolMap, codons)
	syn := make([]*gp.DerivationTree, len(out))
	for i, branch := range out {
		syn[i] = branch.ToSyntaxTree()
	}

	fmt.Println(syn)
	// circuit := compileSyntaxTree(syn[0])
	// fmt.Println(circuit)
}
