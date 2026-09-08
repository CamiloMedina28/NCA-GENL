# Mathematical foundations 

## Vector space axioms

A vector space V is a set of objects (called vectors), with two operations called addition and multiplication by a scalar that satisfies ten axioms

1. If $x \in V$ and $y \in V \to x + y \in V$. Closure under addition.
2. $\forall x,y,z \in V, (x + y) + z = x + (y + z)$. Associativity of addition.
3. $\exists 0 \in V : \forall x \in V, x + 0 = 0 + x = x$. Identity element of addition (Additive identity).
4. If $x \in V \to \exists -x \in V : x + (-x) = 0$. Inverse element of addition (Additive inverse).
5. $\forall x, y \in V, x + y = y + x$. Commutativity of addition.
6. If $\alpha \in \mathbb{F}$ and $x \in V \to \alpha x \in V$. Closure under scalar multiplication.
7. $\forall \alpha \in \mathbb{F}, \forall x, y \in V, \alpha(x + y) = \alpha x + \alpha y$. Distributivity of scalar multiplication with respect to vector addition.
8. $\forall \alpha, \beta \in \mathbb{F}, \forall x \in V, (\alpha + \beta)x = \alpha x + \beta x$. Distributivity of scalar multiplication with respect to field addition.
9. $\forall \alpha, \beta \in \mathbb{F}, \forall x \in V, \alpha(\beta x) = (\alpha\beta)x$. Compatibility of scalar multiplication (Scalar associativity).
10. $\forall x \in V, 1x = x$. Identity element of scalar multiplication (Scalar identity).

## Logistic linear regression and sigmoid