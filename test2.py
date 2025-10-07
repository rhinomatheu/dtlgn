import jax
import jax.numpy as jnp
from functools import partial

import numpy as np

def p(x: float | int, y: float | int, w: jax.Array) -> float:
    """

    """

    return w[0] + w[1]*x + w[2]*y + w[3]*x*y

@partial(jax.jit, static_argnames=("data"))
def loss(w, data):

    l = 0
    for datum in data:
        l += (p(datum[0], datum[1], w) - datum[2])**2
    
    return l

grad_loss = jax.jit(jax.grad(loss), static_argnames="data")

# XOR gate
data = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
)

lr = 0.01
N = 1000

w = jnp.array(np.random.normal(size=(4,)))

for i in range(N):
    w = w - lr*grad_loss(w, data)
    if i%100==0:
        print(f"iter: {i}")

print()
print(f"final weights: {w}")
print()
print(" A | B | A^B")
print("------------")
for i in range(2):
    for j in range(2):
        x = p(i, j, w)
        print(f" {i} | {j} | {x:.03f}")


import matplotlib.pyplot as plt

def xor(x, y):
    return x+y-2*x*y

N=100
cc = np.linspace(0, 1, N)
xxor=np.empty((N,N))
lxor=np.empty((N,N))
for i in range(N):
    for j in range(N):
        xxor[i,j]=xor(cc[i], cc[j])
        lxor[i,j]=p(cc[i], cc[j], w)

plt.subplot(1, 2, 1)
plt.imshow(xxor, origin="lower", extent=[0,1,0,1])
plt.title("real XOR")

plt.subplot(1, 2, 2)
plt.imshow(lxor, origin="lower", extent=[0,1,0,1])
plt.title("learned XOR")

plt.show()