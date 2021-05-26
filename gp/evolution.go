package gp

import (
	"fmt"
)

type Gene byte
type Genome []Gene

// Construct derivation tree
func RunDerivationSequence(rules Rules, symbolNames SymbolNames, codons Genome) []*DerivationTree {
	var root = RootNodeFromRules(rules, symbolNames)
	var out []*DerivationTree

	i := 0
	for true {
		if i >= len(codons) {
			fmt.Println("Ran out of genetic code, finishing root automatically")
			root.Finish()
			out = append(out, root)
			break
		}

		// Find where in the tree to grow from
		node := root.GetFirstNonTerminalLeaf()
		if node == nil {
			// start a new branch
			out = append(out, root)
			root = RootNodeFromRules(rules, symbolNames)
			node = root
		}

		possibleRules := node.rules[node.Value]
		children := possibleRules[int(codons[i])%len(possibleRules)]
		for _, symbol := range children {
			newNode := RootNodeFromRules(rules, symbolNames)
			newNode.Value = symbol
			node.AppendChild(newNode)
		}

		i += 1
	}

	return out
}
