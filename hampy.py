import numpy as np
import matplotlib.pyplot as plt


# | id | Operator             | AB=-1-1 | AB=1-1 | AB=-11 | AB=11 |
# |----|----------------------|---------|--------|--------|-------|
# | 0  | 0                    | -1      | -1     | -1     | -1    |
# | 1  | A and B              | -1      | -1     | -1     |  1    |
# | 2  | not(A implies B)     | -1      |  1     | -1     | -1    |
# | 3  | A                    | -1      |  1     | -1     |  1    |
# | 4  | not(B implies A)     | -1      | -1     |  1     | -1    |
# | 5  | B                    | -1      | -1     |  1     |  1    |
# | 6  | A xor B              | -1      |  1     |  1     | -1    |
# | 7  | A or B               | -1      |  1     |  1     |  1    |
# | 8  | not(A or B)          |  1      | -1     | -1     | -1    |
# | 9  | not(A xor B)         |  1      | -1     | -1     |  1    |
# | 10 | not(B)               |  1      |  1     | -1     | -1    |
# | 11 | B implies A          |  1      |  1     | -1     |  1    |
# | 12 | not(A)               |  1      | -1     |  1     | -1    |
# | 13 | A implies B          |  1      | -1     |  1     |  1    |
# | 14 | not(A and B)         |  1      |  1     |  1     | -1    |
# | 15 | 1                    |  1      |  1     |  1     |  1    |

coefs = np.array([
    [-1.,  0.,  0.,  0.],
    [-0.5,  0.5,  0.5,  0.5],
    [-0.5,  0.5, -0.5, -0.5],
    [0., 1., 0., 0.],
    [-0.5, -0.5, 0.5, -0.5],
    [0., 0., 1., 0.],
    [0., 0., 0., -1.],
    [0.5, 0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5, 0.5],
    [0., 0., 0., 1.],
    [0., 0., -1., 0.],
    [0.5, 0.5, -0.5, 0.5],
    [0., -1., 0., 0.],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5, -0.5],
    [1., 0., 0., 0.]
])

def gate(x: float | int, y: float | int, w: np.array) -> float:
    return w[0] + w[1]*x + w[2]*y + w[3]*x*y

def inner_product(w1: np.array, w2: np.array) -> float:
    d=0
    for x in [-1, 1]:
        for y in [-1, 1]:
            d += gate(x,y,w1)*gate(x,y,w2)
    
    return 0.25*d

def main():
    
    ee = np.empty((16, 16))

    for i in range(16):
        for j in range(16):
            ee[i,j] = inner_product(coefs[i], coefs[j])
    

    plt.imshow(ee, origin="lower")
    plt.show()

if __name__ == "__main__":
    main()