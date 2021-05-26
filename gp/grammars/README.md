## Rules for the grammar
`lhs -> rhs1 [rhs2 ... rhsN]`

Means that the node with type `lhs` will have, as children, N nodes of types `rhs1 ... rhsN` if this rule is applied.

Multiple rules can exist for any given `lhs` type, and they will be chosen from at random during mutation. NOTE: Rules should be ordered such that always following the first rule will quickly construct a valid derivation tree

`!` is the start symbol, it marks what the root nodes can be. If there should be multiple options here, then have it go to a dummy starter node, and then have rules for that starter to have those options as children.

`_` is the terminal symbol. It represents a choice to not create a child node during mutation. If the only rule for a given `lhs` type is `_`, that means the node is terminal (it will never have any children)

`~` is the intron symbol. For each rule with this as the `lhs`, the first `rhs` will be considered an intron, and taken out during the conversion from derivation tree to syntax tree.
