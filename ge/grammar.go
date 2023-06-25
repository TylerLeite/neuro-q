package ge

import (
	"bufio"
	"log"
	"os"
	"strings"
)

type Symbol byte
type Rule []Symbol
type Rules map[Symbol][]Rule
type SymbolNames map[Symbol]string

// Loads a ruleset from a .grmr file (see grammars/README.md)
func LoadRulesFromFile(fName string) (Rules, SymbolNames) {
	symbolMap := make(map[string]Symbol)
	symbolMap["!"] = 0x01
	symbolMap["_"] = 0xFE
	symbolMap["~"] = 0xFF
	var nextSymbol Symbol = 0x02

	file, err := os.Open(fName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	out := make(Rules)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// Ignore comments
		if line[0] == '#' {
			continue
		}

		parts := strings.Split(line, " -> ")
		if symbolMap[parts[0]] == 0 {
			symbolMap[parts[0]] = nextSymbol
			nextSymbol += 1
		}

		lhsSymbol := symbolMap[parts[0]]
		if out[lhsSymbol] == nil {
			out[lhsSymbol] = make([]Rule, 0)
		}

		rhs := strings.Split(parts[1], " ")
		rhsSymbols := make([]Symbol, len(rhs))
		for i, str := range rhs {
			if symbolMap[str] == 0 {
				symbolMap[str] = nextSymbol
				nextSymbol += 1
			}

			rhsSymbols[i] = symbolMap[str]
		}

		out[lhsSymbol] = append(out[lhsSymbol], rhsSymbols)
	}

	err = scanner.Err()
	if err != nil {
		log.Fatal(err)
	}

	symbolNames := make(SymbolNames)
	for k, v := range symbolMap {
		symbolNames[v] = k
	}

	return out, symbolNames
}

// The DerivationTree structure is an intermediary step between a genome and the
//
//	algorithm (syntax tree) it represents
//
// Syntax tree is a derivatiojn tree with all introns removed
// TODO: SyntaxTree struct or IsSyntaxTree() method?
type DerivationTree struct {
	Value Symbol

	Parent   *DerivationTree
	Children []*DerivationTree

	Rules
	SymbolNames
}

// In order for a node to tell when it is terminal or an intron, it needs to
//
//	reference the rules. So each node contains a pointer to the ruleset it was
//	created from. A ruleset should only have 1 valid start node
func RootNodeFromRules(rules Rules, symbolNames SymbolNames) *DerivationTree {
	// 0x01 is reserved as '!', the starting symbol
	return &DerivationTree{
		Value:       rules[0x01][0][0],
		Parent:      nil,
		Rules:       rules,
		SymbolNames: symbolNames,
	}
}

// Quickly finish an incomplete derivation tree, filling in nodes with defaults
func (t *DerivationTree) Finish() {
	if len(t.Children) == 0 {
		// Leaf node
		if !SymbolIsTerminal(t.Value, t.Rules) {
			options := t.Rules[t.Value]

			// Use the first terminal option, if any
			for _, child := range options {
				if SymbolIsTerminal(child[0], t.Rules) {
					newNode := RootNodeFromRules(t.Rules, t.SymbolNames)
					newNode.Value = child[0]
					t.AppendChild(newNode)
					return
				}
			}

			// Did not find a terminal symbol, default to the first rule
			rule := options[0]
			for _, symbol := range rule {
				newNode := RootNodeFromRules(t.Rules, t.SymbolNames)
				newNode.Value = symbol
				t.AppendChild(newNode)
				newNode.Finish()
			}
		}
	} else {
		for _, child := range t.Children {
			child.Finish()
		}
	}
}

// Simple helper to update the parent and child simultaneously
func (t *DerivationTree) AppendChild(child *DerivationTree) {
	t.Children = append(t.Children, child)
	child.Parent = t
}

// Remove unexpressed nodes (introns)
func (t *DerivationTree) ToSyntaxTree() *DerivationTree {
	if SymbolIsUnexpressed(t.Value, t.Rules) {
		newRoot := t.Children[0].ToSyntaxTree()

		// Update parent/child pointers
		newRoot.Parent = t.Parent
		if newRoot.Parent != nil {
			for i, child := range newRoot.Parent.Children {
				// Note: it might be the case that i is always 0
				if child == t {
					newRoot.Parent.Children[i] = newRoot
				}
			}
		}

		return newRoot
	} else {
		if len(t.Children) > 0 {
			for i, child := range t.Children {
				t.Children[i] = child.ToSyntaxTree()
			}
		}

		return t
	}
}

// Recursively search through the tree depth-first, left-right
func (t *DerivationTree) GetFirstNonTerminalLeaf() *DerivationTree {
	if len(t.Children) == 0 {
		// This is a leaf node, check if non-terminal
		if !SymbolIsTerminal(t.Value, t.Rules) {
			return t
		} else {
			return nil
		}
	}

	for _, child := range t.Children {
		found := child.GetFirstNonTerminalLeaf()
		if found != nil {
			return found
		}
	}

	return nil
}

// Simple print similar to the unix 'tree' command
func (t *DerivationTree) String() string {
	const (
		childStart = ""
		indent     = "  "
	)

	out := "\n" + childStart + t.SymbolNames[t.Value]
	if len(t.Children) > 0 {
		for _, child := range t.Children {
			childString := strings.ReplaceAll(child.String(), "\n", "\n"+indent)
			out += childString
		}
	}

	return out
}

// Terminal symbols have no rules for generating child nodes.
// In a valid derivation tree, all leaf nodes are terminal symbols.
func SymbolIsTerminal(symbol Symbol, rules Rules) bool {
	rule := rules[symbol]

	// 0xFE is reserved as '_', the identity symbol
	if len(rule) == 1 && rule[0][0] == 0xFE {
		return true
	} else {
		return false
	}
}

// Unexpressed symbols are those never present in the syntax tree.
// These symboles are a.k.a 'introns'
func SymbolIsUnexpressed(symbol Symbol, rules Rules) bool {
	rule := rules[0xFF]

	// 0xFF is reserved as '~', the 'unexpression' symbol
	for _, unexpressed := range rule {
		if symbol == unexpressed[0] {
			return true
		}
	}

	return false
}
