package gp

import (
	"fmt"
)

func TestAll() {
	TestGrammar()
}

func TestGrammar() {
	rules, symbolNames := LoadRulesFromFile("./gp/grammars/clifford_plus_t.grmr")

	a := RootNodeFromRules(rules, symbolNames)
	// op
	b := RootNodeFromRules(rules, symbolNames)
	b.Value = 0x03
	a.AppendChild(b)
	// CNOT
	e := RootNodeFromRules(rules, symbolNames)
	e.Value = 0x0A
	b.AppendChild(e)

	// K
	c := RootNodeFromRules(rules, symbolNames)
	a.AppendChild(c)
	// op
	j := RootNodeFromRules(rules, symbolNames)
	j.Value = 0x03
	c.AppendChild(j)
	// CNOT
	o := RootNodeFromRules(rules, symbolNames)
	o.Value = 0x0A
	j.AppendChild(o)

	// K
	k := RootNodeFromRules(rules, symbolNames)
	c.AppendChild(k)
	// q
	f := RootNodeFromRules(rules, symbolNames)
	f.Value = 0x09
	k.AppendChild(f)
	// q0
	// h := RootNodeFromRules(rules, symbolNames)
	// h.Value = 0x0B
	// f.AppendChild(h)

	// K
	l := RootNodeFromRules(rules, symbolNames)
	c.AppendChild(l)
	// q
	m := RootNodeFromRules(rules, symbolNames)
	m.Value = 0x09
	l.AppendChild(m)
	// q1
	n := RootNodeFromRules(rules, symbolNames)
	n.Value = 0x0D
	m.AppendChild(n)

	// K
	d := RootNodeFromRules(rules, symbolNames)
	a.AppendChild(d)
	// const
	g := RootNodeFromRules(rules, symbolNames)
	g.Value = 0x08
	d.AppendChild(g)
	// 1-ket
	i := RootNodeFromRules(rules, symbolNames)
	i.Value = 0x15
	g.AppendChild(i)

	fmt.Println(a)
	a.Finish()
	fmt.Println(a)

	s := a.ToSyntaxTree()
	fmt.Println(s)
}
