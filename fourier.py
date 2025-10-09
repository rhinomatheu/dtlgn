import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, static_argnames=("n",))
def _to_bin(
        n: int,
        x: int | jax.Array,
    ) -> jax.Array:
    """
    """

    num = x.size
    x = jnp.array(x, dtype=jnp.uint32)
    x = jnp.reshape(x, (num,))
    x = jnp.unpackbits(x.view(jnp.uint8), axis=-1, bitorder="little")
    x = jnp.reshape(x, (num, 32))
    x = jnp.flip(x, axis=-1)

    return jnp.squeeze(x[:, -n:])

@partial(jax.jit, static_argnames=("n",))
def get_I(n, x):
    num = x.size
    x = jnp.array(x, dtype=jnp.uint32)
    x = jnp.reshape(x, (num,))
    x = jnp.unpackbits(x.view(jnp.uint8), axis=-1, bitorder="little")
    x = jnp.reshape(x, (num, 32))

    return jnp.squeeze(x[:, :n])

@partial(jax.jit, static_argnames=("n",))
def get_hamming_string(
        n: int,
        x: int | jax.Array,
    ) -> jax.Array:
    """
    """

    bin = _to_bin(n, x).astype(jnp.float32)
    bin = 1-2*bin

    return bin

@jax.jit
def parity_I(
        I: jax.Array,
        x: jax.Array,
    ) -> float | jax.Array:
    """
    """

    return jnp.prod(jnp.power(x, I))

@partial(jax.jit, static_argnames=("n",))
def parity_k(
        n: int,
        k: int,
        x: jax.Array,
    ) -> float | jax.Array:
    """
    """

    # 0: I=00000...
    # 1: I=10000...

    # input length = I length

    I = jnp.array(k, dtype=jnp.uint32)
    I = jnp.reshape(I, (1,))
    I = jnp.unpackbits(I.view(jnp.uint8), axis=-1, bitorder="little")
    I = jnp.reshape(I, (32,))[:n]

    return parity_I(I, x)

def parity_k_factory(
        n: int,
        k: int,
    ) -> callable:

    @jax.jit
    def parity_k_impl(
            x: jax.Array,
        ) -> float | jax.Array:
        """
        """

        # 0: I=00000...
        # 1: I=10000...

        # input length = I length

        I = jnp.array(k, dtype=jnp.uint32)
        I = jnp.reshape(I, (1,))
        I = jnp.unpackbits(I.view(jnp.uint8), axis=-1, bitorder="little")
        I = jnp.reshape(I, (32,))[:n]

        return parity_I(I, x)
    
    return parity_k_impl

@partial(jax.jit, static_argnames=("n",))
def inner_product(
        n: int,
        f: callable,
        g: callable,
    ) -> float:
    d = 0.
    for i in range(2**n):
        x = get_hamming_string(n, i)
        d += jnp.dot(f(x), g(x))
    
    return 2**(-n)*d

@partial(jax.jit, static_argnames=("n", "f"))
def coef_k(
        n: int,
        k: int,
        f: callable,
    ) -> float:
    """
    """

    d = 0
    for i in range(2**n):
        x = get_hamming_string(n, i)
        d += f(x)*parity_k(n, k, x)
    
    return 2**(-n)*d

def gate_from_truth(
        n: int,
        tt: jax.Array,
    ) -> callable:
    """
    """

    @jax.jit
    def gate_impl(x: jax.Array):
        # convert input string to decimal
        x = 0.5*(1-x)
        # MSB --- LSB
        idx = jnp.dot(jnp.power(2*jnp.ones(n), jnp.arange(n-1, -1, -1)), x).astype(int)

        return tt[idx]
    
    return gate_impl

@partial(jax.jit, static_argnames=("n",))
def gate_W(n, W, x):
    sum = 0
    for i in range(2**n):
        I = get_I(n, i)
        sum += W[tuple(I)]*parity_I(I, x)
    
    return sum

#

def main():
    # for i in range(2**4):
    #     print(get_hamming_string(4, i))
    
    import numpy as np
    n = 4 # inputs
    tt = jnp.array(np.random.randint(0, 2, (2**n,)))
    tt = 1-2*tt
    tt = tt.astype(int)
    print(tt)
    # random gate
    gate = gate_from_truth(n, tt)
    
    # get fourier coefficients from the random truth table
    W = jnp.zeros(n*[2])
    for i in range(2**n):
        xxs = get_I(n, i)
        idx = tuple(xxs)
        W = W.at[idx].set(coef_k(n, i, gate))
        # print(f"{i}: {coef_k(n, i, gate)}")
    
    for i in range(2**n):
        x = get_hamming_string(n, i)
        print(gate_W(n, W, x))




if __name__ == "__main__":
    main()
