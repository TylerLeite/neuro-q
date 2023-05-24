package qc

import (
	"fmt"
	"log"
	"math/rand"
	"testing"
	"time"
)

func TestAll(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	h := NewHadamard()

	a, err := Operate(h, NewZero())
	if err != nil {
		t.Fatal(err)
	}

	a_test := Tensor([]complex128{ONE_OVER_ROOT_2 + 0i, ONE_OVER_ROOT_2 + 0i})
	if !a.Equals(&a_test) {
		t.Errorf("hadamard gate failed. expected %v. got %v", a_test, a)
	}

	b := NewZero()
	ab := TensorProduct(a, b)
	c := NewZero()
	abc := TensorProduct(ab, c)

	// [(0.70710678118+0i) (0+0i) (0+0i) (0+0i) (0.70710678118+0i) (0+0i) (0+0i) (0+0i)]
	abc_test := Tensor(make([]complex128, 8))
	abc_test[0] = 0.70710678118 + 0i
	abc_test[4] = 0.70710678118 + 0i
	if !abc.Equals(&abc_test) {
		t.Errorf("tensor product failed. expected %v. got %v", abc_test, abc)
	}

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
