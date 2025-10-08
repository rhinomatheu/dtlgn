import jax
import jax.numpy as jnp
from jax.nn import relu
import jax.random


def Linear(
        in_features: int,
        out_features: int,
        *,
        key: jax.Array = None,
        batched: bool = True  
    ):

    if key is None:
        key = jax.random.key(0)

    weight_count: int = out_features # biases
    weight_count += out_features * in_features # linear

    _weights = jax.random.normal(key, shape=(weight_count,)) * 0.1

    if batched:
        @jax.jit
        def linear_impl(weights, x):
            biases = weights[:out_features]
            W = jnp.reshape(weights[out_features:out_features+out_features*in_features], (out_features, in_features))

            out = biases + jnp.einsum("ij,bj->bi", W, x)

            return out
        
        return _weights, linear_impl
    
    else:
        @jax.jit
        def linear_impl(weights, x):
            biases = weights[:out_features]
            W = jnp.reshape(weights[out_features:out_features+out_features*in_features], (out_features, in_features))

            out = biases + jnp.matmul(W, x)

            return out
        
        return _weights, linear_impl

def MultiLinear(
        in_features: int,
        out_features: int,
        *,
        key: jax.Array = None,
        batched: bool = True
    ):

    if key is None:
        key = jax.random.key(0)

    weight_count: int = out_features # biases
    weight_count += out_features * in_features # linear
    trig = in_features * (in_features - 1)//2
    weight_count += out_features * trig # multilinear

    _weights = jax.random.normal(key, shape=(weight_count,)) * 0.1

    if batched:
        @jax.jit
        def multilinear_impl(weights, x):
            biases = weights[:out_features]
            w = jnp.reshape(weights[out_features:out_features+out_features*in_features], (out_features, in_features))
            W = jnp.zeros((out_features, in_features, in_features))
            idx = jnp.triu_indices(in_features, k=1)
            mask = (jnp.repeat(jnp.arange(out_features), trig), jnp.tile(idx[0], out_features), jnp.tile(idx[1], out_features))
            W = W.at[mask].set(weights[out_features+in_features*out_features:])

            out = biases + jnp.einsum("ij,bj->bi", w, x) + jnp.einsum("bi,mij,bj->bm", x, W, x)

            return out
        
        return _weights, multilinear_impl
    
    else:
        @jax.jit
        def multilinear_impl(weights, x):
            biases = weights[:out_features]
            w = jnp.reshape(weights[out_features:out_features+out_features*in_features], (out_features, in_features))
            W = jnp.zeros((out_features, in_features, in_features))
            idx = jnp.triu_indices(in_features, k=1)
            mask = (jnp.repeat(jnp.arange(out_features), trig), jnp.tile(idx[0], out_features), jnp.tile(idx[1], out_features))
            W = W.at[mask].set(weights[out_features+in_features*out_features:])

            out = biases + jnp.matmul(w, x) + jnp.einsum("i,mij,j->m", x, W, x)

            return out
        
        return _weights, multilinear_impl

def PNN(
        features: list[int] | tuple[int],
        *,
        key: jax.Array = None,
        batched: bool = True,
    ):

    assert len(features) >= 2

    if key is None:
        key = jax.random.key(0)
    
    key = jax.random.split(key, num=len(features)-1)

    if len(features) == 2:
        return MultiLinear(*features, key=key[0], batched=batched)

    _weights = jnp.array([])
    _weight_lengths = []
    _layers = []

    for i in range(len(features)-1):
        w, l = MultiLinear(features[i], features[i+1], key=key[i], batched=batched)
        _weights = jnp.append(_weights, w)
        _weight_lengths.append(len(w))
        _layers.append(l)

    strides = [int(x) for x in jnp.cumsum(jnp.array(_weight_lengths))]
    
    @jax.jit
    def pnn_impl(weights, x):

        out = _layers[0](weights[:_weight_lengths[0]], x)
        for i in range(1, len(_layers)-1):
            out = _layers[i](weights[strides[i-1]:strides[i]], out)
        
        out = _layers[-1](weights[-_weight_lengths[-1]:], out)

        return out
    
    return _weights, pnn_impl

def LNN(
        features: list[int] | tuple[int],
        *,
        key: jax.Array = None,
        batched = True,
    ):

    assert len(features) >= 2

    if key is None:
        key = jax.random.key(0)

    key = jax.random.split(key, num=len(features)-1)

    if len(features) == 2:
        return Linear(*features, key=key[0], batched=batched)
    
    _weights = jnp.array([])
    _weight_lengths = []
    _layers = []

    for i in range(len(features)-1):
        w, l = Linear(features[i], features[i+1], key=key[i], batched=batched)
        _weights = jnp.append(_weights, w)
        _weight_lengths.append(len(w))
        _layers.append(l)

    strides = [int(x) for x in jnp.cumsum(jnp.array(_weight_lengths))]

    @jax.jit
    def lnn_impl(weights, x):

        out = _layers[0](weights[:_weight_lengths[0]], x)
        out = relu(out)
        for i in range(1, len(_layers)-1):
            out = _layers[i](weights[strides[i-1]:strides[i]], out)
            out = relu(out)
        
        out = _layers[-1](weights[-_weight_lengths[-1]:], out)

        return out
    
    return _weights, lnn_impl

#

# 3 gate circuit
data = [
    #           A   B   C   D   Z
    jnp.array([-1, -1, -1, -1, -1]),
    jnp.array([-1, -1, -1,  1, -1]),
    jnp.array([-1, -1,  1, -1, -1]),
    jnp.array([-1, -1,  1,  1,  1]),

    jnp.array([-1,  1, -1, -1,  1]),
    jnp.array([-1,  1, -1,  1,  1]),
    jnp.array([-1,  1,  1, -1,  1]),
    jnp.array([-1,  1,  1,  1, -1]),

    jnp.array([ 1, -1, -1, -1,  1]),
    jnp.array([ 1, -1, -1,  1,  1]),
    jnp.array([ 1, -1,  1, -1,  1]),
    jnp.array([ 1, -1,  1,  1, -1]),

    jnp.array([ 1,  1, -1, -1,  1]),
    jnp.array([ 1,  1, -1,  1,  1]),
    jnp.array([ 1,  1,  1, -1,  1]),
    jnp.array([ 1,  1,  1,  1, -1]),
]

#

def main3():

    import random
    key = jax.random.key(random.randint(0, 1000))
    
    w, nn = LNN(features=(4, 10, 5, 1), batched=False, key=key)

    S = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=float
    )

    @jax.jit
    def f(weights, x):
        prod = 0.
        for I in S:
            prod += nn(weights, I)[0]*jnp.prod(jnp.pow(x, I))
        return prod
    
    @jax.jit
    def loss(weights, data):
        l = 0.
        for datum in data:
            l += jnp.square(f(weights, datum[:-1]) - datum[-1])
        return l
    
    grad_loss = jax.jit(jax.grad(loss))

    lr = 0.001
    N = 5000
    epochs = 10

    print("batching data")
    batches = [tuple(random.sample(data, len(data))) for _ in range(epochs)]

    print(f"preloss: {loss(w, data)}")

    for i in range(N):
        for batch in batches:

            grad = grad_loss(w, batch)
            w = w - lr*grad
        
        print(i)
    
    print(f"postloss: {loss(w, data)}")

    # print(w)

    print("   PNN     |   DATA    ")
    print("-----------|-----------")
    for datum in data:
        out = f(w, datum[:-1])
        print(f"    {out:>2.0f}     |    {datum[4]:>2d}     ")
    


def main2():
    import numpy as np

    n = 10
    # w, net = PNN(features=(100, 50, 20, 10, 5, 1))
    w1, net1 = MultiLinear(n, 5, batched=True)
    w2, net2 = MultiLinear(n, 5, batched=False)
    x1 = jnp.array(np.random.normal(size=(100,n)))
    x2 = jnp.array(np.random.normal(size=(n,)))
    print(net1(w1, x1))
    print(net2(w2, x2))
    print(net1(w1,x1).shape)
    print(net2(w2, x2).shape)


def main():
    import numpy as np
    import random

    features = (4, 10, 5, 1)
    
    print(f"jitting p network with feature layers: {features}")
    w, forward = PNN(features=features, batched=False)

    print(f"jitting l network with feature layers: {features}")
    w2, forward2 = LNN(features=features, batched=False)

    @jax.jit
    def loss(weights, data):
        l = 0
        for datum in data:
            l += jnp.square(forward(weights, datum[:-1]) - datum[-1])
        return l[0]
    
    @jax.jit
    def loss2(weights, data):
        l = 0
        for datum in data:
            l += jnp.square(forward2(weights, datum[:-1]) - datum[-1])
        return l[0]
    

    # jit that shit

    print("jitting grad loss")
    grad_loss = jax.jit(jax.grad(loss))

    print("jitting grad loss2")
    grad_loss2 = jax.jit(jax.grad(loss2))

    lr = 0.001
    N = 500
    epochs = 10

    print("batching data")
    batches = [tuple(random.sample(data, len(data))) for _ in range(epochs)]


    for i in range(N):
        for batch in batches:

            grad = grad_loss(w, batch)
            grad2 = grad_loss2(w2, batch)
            w = w - lr*grad
            w2 = w2 - lr*grad2
        
        print(i)

    print("   PNN     |   DATA    |    LNN    ")
    print("-----------|-----------|-----------")
    for datum in data:
        out = forward(w, datum[:-1])[0]
        out2 = forward2(w2, datum[:-1])[0]
        print(f"    {out:>2.0f}     |    {datum[4]:>2d}     |    {out2:>2.0f}     ")

if __name__ == "__main__":
    # main()
    # main2()
    main3()