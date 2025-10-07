import jax
import jax.numpy as jnp
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

    weights = jax.random.normal(key, shape=(weight_count,)) * 0.1

    if batched:
        pass

    else:
        pass

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

    weights = jax.random.normal(key, shape=(weight_count,)) * 0.1

    if batched:
        @jax.jit
        def multilinear_impl(weights, input):
            biases = weights[:out_features]
            w = jnp.reshape(weights[out_features:out_features+in_features*out_features], (out_features, in_features))
            W = jnp.zeros((out_features, in_features, in_features))
            idx = jnp.triu_indices(in_features, k=1)
            mask = (jnp.repeat(jnp.arange(out_features), trig), jnp.tile(idx[0], out_features), jnp.tile(idx[1], out_features))
            W = W.at[mask].set(weights[out_features+in_features*out_features:])

            out = biases + jnp.einsum("ij,bj->bi", w, input) + jnp.einsum("bi,mij,bj->bm", input, W, input)

            return out
        
        return weights, multilinear_impl
    
    else:
        @jax.jit
        def multilinear_impl(weights, input):
            biases = weights[:out_features]
            w = jnp.reshape(weights[out_features:out_features+in_features*out_features], (out_features, in_features))
            W = jnp.zeros((out_features, in_features, in_features))
            idx = jnp.triu_indices(in_features, k=1)
            mask = (jnp.repeat(jnp.arange(out_features), trig), jnp.tile(idx[0], out_features), jnp.tile(idx[1], out_features))
            W = W.at[mask].set(weights[out_features+in_features*out_features:])

            out = biases + jnp.matmul(w, input) + jnp.einsum("i,mij,j->m", input, W, input)

            return out
        
        return weights, multilinear_impl

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
        return MultiLinear(*features, key[0])

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

#

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

    # 3 gate circuit
    data = [
    #     A   B   C   D   Z
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

    w1, layer1 = MultiLinear(4, 2)
    w2, layer2 = MultiLinear(2, 1)

    n1 = w1.size
    n2 = w2.size

    w = jnp.append(w1, w2)

    @jax.jit
    def forward(weights, x):
        
        ww1 = weights[:n1]
        ww2 = weights[n1:]

        z = layer1(ww1, x)
        y = layer2(ww2, z)

        return y
    
    x = jnp.array(np.random.rand(4))

    features = (4, 10, 5, 1)
    print(f"jitting network with feature layers: {features}")

    w, forward = PNN(features=features, batched=False)

    print(forward(w, x))

    @jax.jit
    def loss(weights, data):
        l = 0
        for datum in data:
            l += jnp.square(forward(weights, datum[:-1]) - datum[-1])
        return l[0]
    
    print(loss(w, data))
    # jit that shit
    grad_loss = jax.jit(jax.grad(loss))

    lr = 0.001
    N = 500
    epochs = 10

    batches = [tuple(random.sample(data, len(data))) for _ in range(epochs)]


    for i in range(N):
        for batch in batches:

            grad = grad_loss(w, batch)
            w = w - lr*grad
        
        print(i)

    print(w)
    print(loss(w, data))

    print("  LEARNED  |   DATA     ")
    print("-----------|------------")
    for datum in data:
        out = forward(w, datum[:-1])[0]
        print(f"    {out:>2.0f}     |  {datum[4]:>2d}   ")

if __name__ == "__main__":
    main()
    # main2()