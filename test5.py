import numpy as np

def chi(i: int) -> callable:

    if i==0:
        return lambda x,y: 1
    elif i==1:
        return lambda x,y: x
    elif i==2:
        return lambda x,y: y
    elif i==3:
        return lambda x,y: x*y
    elif i==4:
        return lambda x,y: x**2
    elif i==5:
        return lambda x,y: y**2
    elif i==6:
        return lambda x,y: x**2*y
    elif i==7:
        return lambda x,y: x*y**2
    elif i==8:
        return lambda x,y: x**2*y**2
    

def ip(f: callable, g: callable) -> float:
    
    d = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            d += f(x,y)*g(x,y)
    return d

def bchi(i: int) -> callable:

    if i==0:
        return lambda x,y: 1
    elif i==1:
        return lambda x,y: x
    elif i==2:
        return lambda x,y: y
    elif i==3:
        return lambda x,y: x*y
    

def bip(f: callable, g: callable) -> float:
    
    d = 0
    for x in [-1, 1]:
        for y in [-1, 1]:
            d += f(x,y)*g(x,y)
    return d/4.

#


def main():
    for i in range(9):
        for j in range(9):

            d = ip(chi(i), chi(j))
            print(f"({i}, {j}): {d}")
    
    # print()
    # for i in range(4):
    #     for j in range(4):

    #         d = bip(bchi(i), bchi(j))
    #         print(f"({i}, {j}): {d}")


if __name__ == "__main__":
    main()