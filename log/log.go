package log

import "fmt"

const DEBUG = false

// TODO: better logging solution
// e.g.
// log.Set(flags ...bool)
// defer log.Unset(flags ...bool)
func Book(msg string, flags ...bool) {
	for _, flag := range flags {
		if !flag {
			return
		}
	}

	fmt.Print(msg)
}

type LineBreak uint8

const (
	NL LineBreak = iota
	HR
)

func Break(typ LineBreak, flags ...bool) {
	for _, flag := range flags {
		if !flag {
			return
		}
	}

	switch typ {
	case NL:
		fmt.Println()
	case HR:
		fmt.Println("-------------------------------")
	default:
		fmt.Println()
	}
}
