import jax
import jax.numpy as jnp

import numpy as np

import random

import matplotlib.pyplot as plt

GATE_NAMES=[
    "FALSE",
    "A AND B",
    "NOT(A IMPLIES B)",
    "A",
    "NOT(B IMPLIES A)",
    "B",
    "A XOR B",
    "A OR B",
    "NOT(A OR B)",
    "NOT(A XOR B)",
    "NOT(B)",
    "B IMPLIES A",
    "NOT(A)",
    "A IMPLIES B",
    "NOT(A AND B)",
    "TRUE"
]

# 3 gate circuit
data = [
#     A   B   C   D   Z
    (-1, -1, -1, -1, -1),
    (-1, -1, -1,  1, -1),
    (-1, -1,  1, -1, -1),
    (-1, -1,  1,  1,  1),

    (-1,  1, -1, -1,  1),
    (-1,  1, -1,  1,  1),
    (-1,  1,  1, -1,  1),
    (-1,  1,  1,  1, -1),

    ( 1, -1, -1, -1,  1),
    ( 1, -1, -1,  1,  1),
    ( 1, -1,  1, -1,  1),
    ( 1, -1,  1,  1, -1),

    ( 1,  1, -1, -1,  1),
    ( 1,  1, -1,  1,  1),
    ( 1,  1,  1, -1,  1),
    ( 1,  1,  1,  1, -1),
]

# polynomial circuit block
@jax.jit
def p(x, y, w):
    return w[0] + w[1]*x + w[2]*y + w[3]*x*y

# inner product on L(G)
# G here is $U_2\times U_2$
# $f:G\to S^1$
@jax.jit
def ip(w1, w2):
    d = 0
    for x in [-1, 1]:
        for y in [-1, 1]:
            d += p(x, y, w1)*p(x, y, w2)
    
    return d/4

# distance metric on L(G)
@jax.jit
def dist(w1, w2):
    d = ip(w1-w2, w1-w2)
    return jnp.sqrt(d)

# 3 gate circuit
# stand-in for what would be the network
@jax.jit
def circuit(a, b, c, d, w):

    # OR: w[:4]
    # AND: w[4:8]
    # XOR: w[8:]

    OR = p(a, b, w[:4])
    AND = p(c, d, w[4:8])
    XOR = p(OR, AND, w[8:])

    return XOR

# loss over one batch of data
@jax.jit
def loss(w, data):
    l = 0
    for datum in data:
        l += jnp.square(circuit(datum[0], datum[1], datum[2], datum[3], w) - datum[4])
    
    return l

# jit that shit
grad_loss = jax.jit(jax.grad(loss))

lr = 0.001
N = 50 # needs not more
epochs = 10

batches = [tuple(random.sample(data, len(data))) for _ in range(epochs)]

# w = jnp.array(np.random.randn(12,))
# uniform initialization works better since all gate coefs are
# actually \in {-1, -0.5, 0.5, 1}
w = jnp.array(np.random.uniform(-1, 1, (12,)))

LOSS = jnp.empty((N+1,))
LOSS = LOSS.at[0].set(loss(w, data))

for i in range(N):
    for batch in batches:

        grad = grad_loss(w, batch)
        w = w - lr*grad
    
    LOSS = LOSS.at[i+1].set(loss(w, data))
    print(i)

plt.plot(LOSS)
plt.show()

print(w)
print(loss(w, data))

# GATE COEFS
ccs=jnp.array([
    [-1.,   0.,   0.,   0. ],
    [-0.5,  0.5,  0.5,  0.5],
    [-0.5,  0.5, -0.5, -0.5],
    [ 0.,   1.,   0.,   0. ],
    [-0.5, -0.5,  0.5, -0.5],
    [ 0.,   0.,   1.,   0. ],
    [ 0.,   0.,   0.,  -1. ],
    [ 0.5,  0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5, 0.5],
    [ 0.,   0.,   0.,   1. ],
    [ 0.,   0.,  -1.,   0. ],
    [ 0.5,  0.5, -0.5,  0.5],
    [ 0.,  -1.,   0.,   0., ],
    [ 0.5, -0.5,  0.5,  0.5],
    [ 0.5, -0.5, -0.5, -0.5],
    [ 1.,   0.,   0.,   0. ],
])

w_OR = w[:4]
w_AND = w[4:8]
w_XOR = w[8:]

mn = 10
idx = 0
print("TEST")
for i, gate in enumerate(ccs):
    d = dist(w_OR, gate)
    if d < mn:
        mn = d
        idx = i

    # print(f"{i}: {ip(w_OR,gate):.04f}")
print("OR: " + GATE_NAMES[idx])
w1 = ccs[idx]

mn = 10
idx = 0
for i, gate in enumerate(ccs):
    d = dist(w_AND, gate)
    if d < mn:
        mn = d
        idx = i
    # print(f"{i}: {ip(w_AND,gate):.04f}")
print("AND: " + GATE_NAMES[idx])
w2 = ccs[idx]

mn = 10
idx = 0
for i, gate in enumerate(ccs):
    d = dist(w_XOR, gate)
    if d < mn:
        mn = d
        idx = i
    # print(f"{i}: {ip(w_AND,gate):.04f}")
print("XOR: " + GATE_NAMES[idx])
w3 = ccs[idx]

wr = jnp.hstack([w1, w2, w3])

print("  LEARNED  |   DATA    |  RECONS   ")
print("-----------|-----------|-----------")
for datum in data:
    out = circuit(datum[0], datum[1], datum[2], datum[3], w)
    recon = circuit(datum[0], datum[1], datum[2], datum[3], wr)
    print(f"    {out:>2.0f}     |  {datum[4]:>2d}       |  {recon:>2.0f}")