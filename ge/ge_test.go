package ge

import (
	"fmt"
	"testing"
)

func TestGrammar(t *testing.T) {
	rules, symbolNames := LoadRulesFromFile("./grammars/clifford_plus_t.grmr")

	fmt.Println(rules)
	fmt.Println(symbolNames)

	a := RootNodeFromRules(rules, symbolNames)
	a.Value = 0x01
	// op
	b := RootNodeFromRules(rules, symbolNames)
	b.Value = 0x02
	a.AppendChild(b)
	// CNOT
	e := RootNodeFromRules(rules, symbolNames)
	e.Value = 0x09
	b.AppendChild(e)

	// K
	c := RootNodeFromRules(rules, symbolNames)
	c.Value = 0x03
	e.AppendChild(c)
	// op
	j := RootNodeFromRules(rules, symbolNames)
	j.Value = 0x02
	c.AppendChild(j)
	// CNOT
	o := RootNodeFromRules(rules, symbolNames)
	o.Value = 0x09
	j.AppendChild(o)

	// K
	k := RootNodeFromRules(rules, symbolNames)
	k.Value = 0x03
	o.AppendChild(k)
	// q
	f := RootNodeFromRules(rules, symbolNames)
	f.Value = 0x05
	k.AppendChild(f)
	// q0
	// h := RootNodeFromRules(rules, symbolNames)
	// h.Value = 0x0B
	// f.AppendChild(h)

	// K
	l := RootNodeFromRules(rules, symbolNames)
	l.Value = 0x03
	o.AppendChild(l)
	// q
	m := RootNodeFromRules(rules, symbolNames)
	m.Value = 0x05
	l.AppendChild(m)
	// q1
	n := RootNodeFromRules(rules, symbolNames)
	n.Value = 0x0C
	m.AppendChild(n)

	// K
	d := RootNodeFromRules(rules, symbolNames)
	d.Value = 0x03
	e.AppendChild(d)
	// const
	g := RootNodeFromRules(rules, symbolNames)
	g.Value = 0x04
	d.AppendChild(g)
	// 1-ket
	i := RootNodeFromRules(rules, symbolNames)
	i.Value = 0x14
	g.AppendChild(i)

	fmt.Println(a)
	a.Finish()
	fmt.Println(a)

	s := a.ToSyntaxTree()
	fmt.Println(s)
}
