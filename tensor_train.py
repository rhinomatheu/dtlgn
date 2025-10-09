import jax
import jax.numpy as jnp
import jax.random


def TensorTrain(
        features: int,
        rank: int,
        *,
        key: jax.Array = None,
        batched: bool = False,
    ):

    if key is None:
        key = jax.random.key(0)
    
    n, r = features, rank
    _weights = jax.random.normal(key, shape=( 2*(n-2)*r**2+4*r, )) * 0.1

    strides = jnp.array([2*r])
    for i in range(1, n-2):
        strides = jnp.append(strides, strides[-1] + 2*r*r)
    
    strides = strides.tolist()

    if batched:
        # TODO: do forward pass with batched data
        pass
    
    else:
        @jax.jit
        def tensor_train_impl(weights, x):
            
            G = jnp.reshape(weights[:2*r], (2, r))
            A = G[0] + x[0]*G[1]
            out = A

            for i in range(1, n-2):
                G = jnp.reshape(weights[strides[i-1]:strides[i]], (2, r, r))
                A = G[0] + x[i]*G[1]
                out = jnp.matmul(out, A)
            
            G = jnp.reshape(weights[-2*r:], (2, r))
            A = G[0] + x[-1]*G[1]

            out = jnp.dot(out, A)

            return out
        
        return _weights, tensor_train_impl


#

# 3 gate circuit
data = [
    #           A   B   C   D   Z
    -jnp.array([-1, -1, -1, -1, -1]),
    -jnp.array([-1, -1, -1,  1, -1]),
    -jnp.array([-1, -1,  1, -1, -1]),
    -jnp.array([-1, -1,  1,  1,  1]),

    -jnp.array([-1,  1, -1, -1,  1]),
    -jnp.array([-1,  1, -1,  1,  1]),
    -jnp.array([-1,  1,  1, -1,  1]),
    -jnp.array([-1,  1,  1,  1, -1]),

    -jnp.array([ 1, -1, -1, -1,  1]),
    -jnp.array([ 1, -1, -1,  1,  1]),
    -jnp.array([ 1, -1,  1, -1,  1]),
    -jnp.array([ 1, -1,  1,  1, -1]),

    -jnp.array([ 1,  1, -1, -1,  1]),
    -jnp.array([ 1,  1, -1,  1,  1]),
    -jnp.array([ 1,  1,  1, -1,  1]),
    -jnp.array([ 1,  1,  1,  1, -1]),
]

def main():
    import numpy as np
    import random

    n, r = 4, 8
    w, forward = TensorTrain(n, r)
    x = jnp.array(np.random.normal(size=(n,)))
    print(forward(w, x))

    @jax.jit
    def loss(weights, data):
        l = 0.
        for datum in data:
            l += jnp.square(forward(weights, datum[:-1]) - datum[-1])
        return l
    
    grad_loss = jax.jit(jax.grad(loss))

    lr = 0.01
    N = 10000
    epochs = 10

    print("batching data")
    batches = [tuple(random.sample(data, len(data))) for _ in range(epochs)]

    print(f"preloss: {loss(w, data)}")

    for i in range(N):
        for batch in batches:

            grad = grad_loss(w, batch)
            w = w - lr*grad
        
        if i%100==0:
            print(i)
    
    print(f"postloss: {loss(w, data)}")

    print("   PNN     |   DATA    ")
    print("-----------|-----------")
    for datum in data:
        out = forward(w, datum[:-1])
        print(f"    {out:>2.0f}     |    {datum[4]:>2d}     ")

    
    print(f"tt inputs:{4*r+2*(n-2)*r*r}, poly inputs: {2**n}")

    



if __name__ == "__main__":
    main()