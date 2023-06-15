package neat

import "fmt"

const (
	DEBUG                = false
	DEBUG_PROPAGATION    = false
	DEBUG_RESET          = false
	DEBUG_DRAW           = false
	DEBUG_COMPILE        = false
	DEBUG_ADD_CONNECTION = false
	DEBUG_ADD_NODE       = false
	DEBUG_MUTATION       = false
)

// TODO: better logging solution
func Log(msg string, flags ...bool) {
	for _, flag := range flags {
		if !flag {
			return
		}
	}

	fmt.Print(msg)
}
