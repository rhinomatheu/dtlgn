import jax
import jax.numpy as jnp
from functools import partial

import numpy as np

TRUE = 1
FALSE = -1
UNKNOWN = 0


def p(x: float | int, y: float | int, w: jax.Array) -> float:
    """

    """

    return (
        w[0] + w[1]*x + w[2]*y + w[3]*x*y +
        w[4]*x**2 + w[5]*y**2 + w[6]*x**2*y + w[7]*x*y**2 +
        w[8]*x**2*y**2
    )
    

@partial(jax.jit, static_argnames=("data"))
def loss(w, data):

    l = 0
    for datum in data:
        l += (p(datum[0], datum[1], w) - datum[2])**2
    
    return l

grad_loss = jax.jit(jax.grad(loss), static_argnames="data")

# XOR gate
data_static = [
    (-1, -1, 1),
    (-1, 0, 0),
    (-1, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, -1, 0),
    (1, 0, 0),
    (1, 1, 1)
]
data = tuple(data_static)
lr = 0.01
N = 5000

import random

w = jnp.array(np.random.normal(size=(9,)))

for i in range(N):
    w = w - lr*grad_loss(w, data)
    if i%100==0:
        print(f"iter: {i}")

print()
print(f"final weights: {w}")
print()
print("  A |  B |    I     |  !A^B")
print("---------------------")
k=0
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        x = p(i, j, w)
        print(f" {i:2} | {j:2} | {x:6.03f}  |  {data_static[k][2]:2}")
        k+=1
