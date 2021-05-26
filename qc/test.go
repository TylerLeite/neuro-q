package qc

import (
	"fmt"
	"log"
	"math/rand"
	"time"
)

func TestAll() {
	rand.Seed(time.Now().UnixNano())

	h := NewHadamard()

	a, err := Operate(h, NewZero())
	if err != nil {
		log.Fatal(err)
	}

	b := NewZero()

	ab := TensorProduct(a, b)

	c := NewZero()

	abc := TensorProduct(ab, c)

	fmt.Println(abc)

	i := NewIdentity()
	cn := NewCNOT()

	cnab, err := KroneckerProduct(cn, i)
	if err != nil {
		log.Fatal(err)
	}

	cnbc, err := KroneckerProduct(i, cn)
	if err != nil {
		log.Fatal(err)
	}

	ha, _ := KroneckerProduct(h, i)
	ha, err = KroneckerProduct(ha, i)
	if err != nil {
		log.Fatal(err)
	}

	s1, err := Operate(cnab, abc)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(s1)

	s2, err := Operate(cnbc, s1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(s2)

	s3, err := Operate(ha, s2)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(s3)

	cnabbc, err := MatMul(cnab, cnbc)
	if err != nil {
		log.Fatal(err)
	}

	cnabbcha, err := MatMul(cnabbc, ha)
	if err != nil {
		log.Fatal(err)
	}

	s4, err := Operate(cnabbcha, abc)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(s4)

	cnabbcha2, err := Combine(cnab, cnbc, ha)
	if err != nil {
		log.Fatal(err)
	}

	s5, err := Operate(cnabbcha2, abc)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(s5)
}
