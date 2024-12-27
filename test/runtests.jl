using FlashAttentionWrapper
using NNlib: dot_product_attention
using CUDA
using Test
using Zygote

if CUDA.functional()

    FlashAttentionWrapper.install()

@testset begin
    head_dim = 256
    n_heads = 9
    seq_len = 1024
    batch_size = 4

    q = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu
    k = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu
    v = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu

    o_nn, nn_back = Zygote.pullback(q, k, v) do q, k, v
        o_nn, α = dot_product_attention(q, k, v; nheads=n_heads)
        o_nn
    end

    o_fa, fa_back = Zygote.pullback(q, k, v) do q, k, v
        q, k, v = reshape.((q, k, v), head_dim, n_heads, seq_len, batch_size)
        mha(q, k, v)
    end

    @test isapprox(vec(o_nn), vec(o_fa))

    Δo = randn(Float16, size(o_nn)) |> cu
    Δq_nn, Δk_nn, Δv_nn = nn_back(Δo)
    Δq_fa, Δk_fa, Δv_fa = fa_back(reshape(Δo, head_dim, n_heads, seq_len, batch_size))

    # is it too large?
    @test maximum(abs.(Δq_nn .- Δq_fa)) < 3.0f-3
    @test maximum(abs.(Δk_nn .- Δk_fa)) < 3.0f-3
    @test maximum(abs.(Δv_nn .- Δv_fa)) < 3.0f-3
end
end