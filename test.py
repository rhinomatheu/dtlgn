#
#
#

import numpy as np


# constants
TRUE = SUCCESS = 1
FALSE = FAILURE = -1
UNKNOWN = RUNNING = 0


def NOT(x):

    return -x

def AND(x, y):

    return np.minimum(x, y)

def OR(x, y):

    return np.maximum(x, y)

def IMPLIES(x, y):

    return OR(NOT(x), y)

def DOUBLE_IMPLIES(x, y):

    return AND(IMPLIES(x, y), IMPLIES(y, x))

def SEQ(x, y):

    return 0.5*(y-1)*x**2 + 0.5*(y+1)*x

def SEL(x, y):

    return -SEQ(-x, -y)

#

def main():
    import matplotlib.pyplot as plt

    N = 1000

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)

    # AND
    plt.subplot(3, 2, 1)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = AND(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("AND(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    # OR
    plt.subplot(3, 2, 2)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = OR(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("OR(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    # IMPLIES
    plt.subplot(3, 2, 3)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = IMPLIES(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("IMPLIES(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    # DOUBLE IMPLIES
    plt.subplot(3, 2, 4)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = DOUBLE_IMPLIES(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("DOUBLE_IMPLIES(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    # SEQ
    plt.subplot(3, 2, 5)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = SEQ(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("SEQ(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    # SEL
    plt.subplot(3, 2, 6)

    im = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            im[i,j] = SEL(x[i], y[j])
    
    plt.imshow(im, origin="lower", extent=[-1,1,-1,1])
    plt.title("SEL(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()
    
def main2():
    import matplotlib.pyplot as plt

    # max vs *

    N = 100

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    nd = np.empty((N,N))
    soft_nd = np.empty((N,N))

    for i in range(N):
        for j in range(N):
            nd[i, j] = np.minimum(x[i], y[j])
            soft_nd[i,j] = x[i]*y[j]
    
    plt.subplot(1, 2, 1)
    plt.imshow(nd, origin="lower", extent=[-1,1,-1,1])
    plt.title("hard")

    plt.subplot(1, 2, 2)
    plt.imshow(soft_nd, origin="lower", extent=[-1,1,-1,1])
    plt.title("soft")

    plt.show()


if __name__ == "__main__":
    main2()