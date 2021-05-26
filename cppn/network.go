package cppn

import (
	_ "fmt"
)

var nextId uint64 = 0

const fnLength = 2

type GraphPart interface {
	CalculateValue() float64
	Reset()
}

type Network struct {
	Nodes []*Node
	Edges []*Edge
}

func NewNetwork() *Network {
	n := new(Network)
	return n
}
