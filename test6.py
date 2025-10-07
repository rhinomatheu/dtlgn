import jax
import jax.numpy as jnp
import jax.lax
import matplotlib.pyplot as plt
from functools import partial
import numpy as np


FALSE = 1.
UNKNOWN = complex(-0.5, 3**0.5/2)
TRUE = complex(-0.5, -3**0.5/2)

print(f"omega: {UNKNOWN}")
print(f"omega^2: {TRUE}")

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
        l += jnp.abs(p(datum[0], datum[1], w) - datum[2])**2
    
    return l

grad_loss = jax.jit(jax.grad(loss, holomorphic=False), static_argnames="data")

@jax.jit
def inner_product(w1, w2):
    d=0
    for x in [FALSE, UNKNOWN, TRUE]:
        for y in [FALSE, UNKNOWN, TRUE]:
            d += p(x,y,w1)*jnp.conjugate(p(x,y,w2))
    return jnp.abs(d)/9.


# Indicator
data = (
    (FALSE, FALSE, 1),
    (FALSE, UNKNOWN, 0),
    (FALSE, TRUE, 0),
    (UNKNOWN, FALSE, 0),
    (UNKNOWN, UNKNOWN, 1),
    (UNKNOWN, TRUE, 0),
    (TRUE, FALSE, 0),
    (TRUE, UNKNOWN, 0),
    (TRUE, TRUE, 1)
)

# AND
data2 = (
    (FALSE, FALSE, FALSE),
    (FALSE, UNKNOWN, FALSE),
    (FALSE, TRUE, FALSE),
    (UNKNOWN, FALSE, FALSE),
    (UNKNOWN, UNKNOWN, UNKNOWN),
    (UNKNOWN, TRUE, UNKNOWN),
    (TRUE, FALSE, FALSE),
    (TRUE, UNKNOWN, UNKNOWN),
    (TRUE, TRUE, TRUE)
)

# PASSTHROUGHT, TRUE
data3 = (
    (FALSE, FALSE, TRUE),
    (FALSE, UNKNOWN, TRUE),
    (FALSE, TRUE, TRUE),
    (UNKNOWN, FALSE, TRUE),
    (UNKNOWN, UNKNOWN, TRUE),
    (UNKNOWN, TRUE, TRUE),
    (TRUE, FALSE, TRUE),
    (TRUE, UNKNOWN, TRUE),
    (TRUE, TRUE, TRUE)
)

lr = 0.01
N = 100


# You should probably use the jax.random module instead of numpy.random -Sandeep
# w = jax.random.normal(jax.random.PRNGKey(0), (4,))

w = jnp.array(np.random.randn(9))
w = jnp.array(np.random.randn(9,2).view(np.complex128).reshape(9))

for i in range(N):
    if i%100==0:
        print(f"iteration: {i}")
    w = w - lr*jnp.conjugate(grad_loss(w, data2))

@jax.jit
def log_w(x):
    x = jnp.log(x)/jnp.log(UNKNOWN)
    return jnp.real(x)

print("HERE YO")
for i in range(9):
    print(f"w_{i}: {w[i]:8.05f}")
print()
print(" A | B | A^B")
print("------------")
for i,x in  enumerate([FALSE, UNKNOWN, TRUE]):
    for j, y in enumerate([FALSE, UNKNOWN, TRUE]):
        out = p(x, y, w)
        #out = log_w(out)-1
        print(f" {i-1} | {j-1} | {out:.03f}")

print()
print(" A | B | A^B")
print("------------")
for i,x in  enumerate([FALSE, UNKNOWN, TRUE]):
    for j, y in enumerate([FALSE, UNKNOWN, TRUE]):
        out = p(x, y, w)
        out = jnp.log(out*TRUE)/jnp.log(UNKNOWN)
        out = out.real
        print(f" {i-1} | {j-1} | {out:.03f}")


N=100

X = np.linspace(-1, 1, N)

# map {-1, 0, 1} to S^1
def f2s(x):
    return jnp.pow(UNKNOWN, 1+x)

# map S^1 to {-1, 0, 1}
def s2f(x):
    x = jnp.log(x)/jnp.log(UNKNOWN)
    x = x.real
    if x<-1e-5:
        x = x+3
    return x-1

img = np.empty((N,N))

for i in range(N):
    for j in range(N):

        x = f2s(X[i])
        y = f2s(X[j])

        out = p(x,y,w)

        out = jnp.log(out*TRUE)/jnp.log(UNKNOWN)
        out = out.real

        img[i,j] = out

plt.imshow(img, origin="lower", extent=[-1, 1, -1, 1])
plt.show()