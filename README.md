# FlashAttentionWrapper.jl

Just a simple wrapper for the [Flash Attention](https://github.com/Dao-AILab/flash-attention) operation.

## Installation

```julia
using FlashAttentionWrapper

FlashAttentionWrapper.install()
```

Note that by default it will install the latest version of FlashAttention.

## Example

```julia
using FlashAttentionWrapper

# q, k, v are assumed to be 4d CuArray of size (head_dim, n_heads, seq_len, batch_size)
o = mha(q, k, v; kw...) 
```

Check the original [doc](https://github.com/Dao-AILab/flash-attention/tree/main?tab=readme-ov-file#how-to-use-flashattention) on the explanation of supported keyword arguments.

Backward is also supported:

```julia
using CUDA
using Zygote

o, back = Zygote.pullback(q, k, v) do q, k, v
    mha(q, k, v)
end

Δo = CUDA.randn(eltype(o), size(o))

Δq, Δk, Δv = back(Δo)
```

If you'd like to use it with Lux.jl, here's a handy example:

```julia
using Lux

head_dim, n_head, seq_len, batch_size = 256, 8, 1024, 4
hidden_dim = head_dim * n_head

x = CUDA.randn(Float16, (hidden_dim, seq_len, batch_size))

m = Chain(
    BranchLayer(
        Chain(
            Dense(hidden_dim => hidden_dim, use_bias=false),
            ReshapeLayer((head_dim, n_head, seq_len))
        ),
        Chain(
            Dense(hidden_dim => hidden_dim, use_bias=false),
            ReshapeLayer((head_dim, n_head, seq_len))
        ),
        Chain(
            Dense(hidden_dim => hidden_dim, use_bias=false),
            ReshapeLayer((head_dim, n_head, seq_len))
        ),
    ),
    Attention(),
    ReshapeLayer((hidden_dim, seq_len)),
    Dense(hidden_dim => hidden_dim, use_bias=false),
)

using Random
rng = Random.default_rng()
ps, st = LuxCore.setup(rng, m)
cu_ps = recursive_map(CuArray{Float16}, ps)

o, _ = m(x, cu_ps, st)
```

Or if you prefer Flux.jl:

```julia
using Flux

head_dim, n_head, seq_len, batch_size = 256, 8, 1024, 4
hidden_dim = head_dim * n_head

x = CUDA.randn(Float16, (hidden_dim, seq_len, batch_size))

m = Flux.Chain(
    Flux.Parallel(
        tuple,
        Flux.Chain(
            Flux.Dense(CUDA.randn(Float16, hidden_dim, hidden_dim), false),
            x -> reshape(x, head_dim, n_head, seq_len, batch_size),
        ),
        Flux.Chain(
            Flux.Dense(CUDA.randn(Float16, hidden_dim, hidden_dim), false),
            x -> reshape(x, head_dim, n_head, seq_len, batch_size),
        ),
        Flux.Chain(
            Flux.Dense(CUDA.randn(Float16, hidden_dim, hidden_dim), false),
            x -> reshape(x, head_dim, n_head, seq_len, batch_size),
        ),
    ),
    qkv -> reshape(mha(qkv...;), :, seq_len, batch_size),
    Flux.Dense(CUDA.randn(Float16, hidden_dim, hidden_dim), false),
)

m(x)
```

## TODO List

- [ ] Add benchmark
- [ ] Support FlexAttention?
- [ ] Support [FlashInfer](https://github.com/flashinfer-ai/flashinfer)?
