import jax
import jax.numpy as jnp

def parity(i) -> callable:
    
    w = jnp.zeros(4)
    w = w.at[i].set(1)

    @jax.jit
    def rtn(x, y):
        return jnp.dot(w, jnp.array([1, x, y, x*y]))

    return rtn

def inner_prod(f: callable, g: callable) -> float:
    d = 0.
    for x in [-1, 1]:
        for y in [-1, 1]:
            d += f(x, y)*g(x, y)
    return 0.25*d

def fourier_coef(f: callable) -> jnp.array:
    """
    """

    return jnp.array([
        inner_prod(f, parity(0)),
        inner_prod(f, parity(1)),
        inner_prod(f, parity(2)),
        inner_prod(f, parity(3))
    ])

def poly_gate(w):
    
    @jax.jit
    def rtn(x, y):
        return jnp.dot(w, jnp.array([1, x, y, x*y]))
    
    return rtn

def gate_from_truth(truth_table: jnp.array):
    """Generate an io function from truth table.

    Args:
        truth_table: The truth table (4,).
        hamming: If True, truth_table uses {-1, 1} encoding. If false, truth_table uses {0, 1} encoding.
    
    Note:
        To be consistent with the roots of unity, -1 = True, +1 = False.
        Strange, yes, but it is correct

    truth table should have the following input structure:
       X  |   Y  |   Z  
    --------------------
       1  |   1  |   *   tt[0]
      -1  |   1  |   *   tt[1]
       1  |  -1  |   *   tt[2]
      -1  |  -1  |   *   tt[3]
    """

    def rtn(x, y):
        idx = int((3-x-2*y)/2)
        return truth_table[idx]
    
    return rtn

# REMEMBER, TRUE = -1,  FALSE = 1
# | id | Operator             | AB=11   | AB=-11 | AB=1-1 |AB=-1-1|
# |----|----------------------|---------|--------|--------|-------|
# | 0  | 0                    |  1      |  1     |  1     |  1    |
# | 1  | A and B              |  1      |  1     |  1     | -1    |
# | 2  | not(A implies B)     |  1      | -1     |  1     |  1    |
# | 3  | A                    |  1      | -1     |  1     | -1    |
# | 4  | not(B implies A)     |  1      |  1     | -1     |  1    |
# | 5  | B                    |  1      |  1     | -1     | -1    |
# | 6  | A xor B              |  1      | -1     | -1     |  1    |
# | 7  | A or B               |  1      | -1     | -1     | -1    |
# | 8  | not(A or B)          | -1      |  1     |  1     |  1    |
# | 9  | not(A xor B)         | -1      |  1     |  1     | -1    |
# | 10 | not(B)               | -1      | -1     |  1     |  1    |
# | 11 | B implies A          | -1      | -1     |  1     | -1    |
# | 12 | not(A)               | -1      |  1     | -1     |  1    |
# | 13 | A implies B          | -1      |  1     | -1     | -1    |
# | 14 | not(A and B)         | -1      | -1     | -1     |  1    |
# | 15 | 1                    | -1      | -1     | -1     | -1    |

TRUTHS = [
    [1, 1, 1, 1],       # 0
    [1, 1, 1, -1],      # 1
    [1, -1, 1, 1],      # 2
    [1, -1, 1, -1],     # 3
    [1, 1, -1, 1],      # 4
    [1, 1, -1, -1],     # 5
    [1, -1, -1, 1],     # 6
    [1, -1, -1, -1],    # 7
    [-1, 1, 1, 1],      # 8
    [-1, 1, 1, -1],     # 9
    [-1, -1, 1, 1],     # 10
    [-1, -1, 1, -1],    # 11
    [-1, 1, -1, 1],     # 12
    [-1, 1, -1, -1],    # 13
    [-1, -1, -1, 1],    # 14
    [-1, -1, -1, -1]    # 15
]

coefs = jnp.array([
    [1.0, 0.0, 0.0, 0.0],       # 0
    [0.5, 0.5, 0.5, -0.5],      # 1
    [0.5, 0.5, -0.5, 0.5],      # 2
    [0.0, 1.0, 0.0, 0.0],       # 3
    [0.5, -0.5, 0.5, 0.5],      # 4
    [0.0, 0.0, 1.0, 0.0],       # 5
    [0.0, 0.0, 0.0, 1.0],       # 6
    [-0.5, 0.5, 0.5, 0.5],      # 7
    [0.5, -0.5, -0.5, -0.5],    # 8
    [0.0, 0.0, 0.0, -1.0],      # 9
    [0.0, 0.0, -1.0, 0.0],      # 10
    [-0.5, 0.5, -0.5, -0.5],    # 11
    [0.0, -1.0, 0.0, 0.0],      # 12
    [-0.5, -0.5, 0.5, -0.5],    # 13
    [-0.5, -0.5, -0.5, 0.5],    # 14
    [-1.0, 0.0, 0.0, 0.0],      # 15
])


def main2():
    
    for i, g in enumerate(TRUTHS):
        gate = gate_from_truth(g)
        cf = fourier_coef(gate)
        print(f"[{cf[0]}, {cf[1]}, {cf[2]}, {cf[3]}],\t# {i}")



def main():
    import numpy as np

    w = jnp.array(np.random.rand(4))
    print(w)

    der = poly_gate(w)
    # der = poly_gate(coefs[3])

    for _w in coefs:
        print(inner_prod(der, poly_gate(_w)))
        

def main3():
    
    import numpy as np
    import matplotlib.pyplot as plt

    w = np.array([0.8432878, 0.7175393, 0.8621536, 0.16228338])
    _w = np.array([0.5, 0.5, 0.5, -0.5])

    N = 100
    A = np.empty((N,N)) # exact
    B = np.empty((N,N)) # approx

    x = np.linspace(-1, 1, N)

    for i in range(N):
        for j in range(N):
            X = np.array([1, x[i], x[j], x[i]*x[j]])
            A[i,j] = np.dot(X, _w)
            B[i, j] = np.dot(X, w)
    
    plt.subplot(1,2,1)
    plt.imshow(A, origin="lower", extent=[-1,1,-1,1])
    plt.title("exact")

    plt.subplot(1,2,2)
    plt.imshow(B, origin="lower", extent=[-1,1,-1,1])
    plt.title("approx")
    plt.show()

if __name__ == "__main__":
    main2()