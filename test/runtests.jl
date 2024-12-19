using FlashAttentionWrapper
using NNlib: dot_product_attention
using CUDA
using Test

@testset begin
    head_dim = 256
    n_heads = 9
    seq_len = 1024
    batch_size = 4

    q = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu
    k = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu
    v = rand(Float16, head_dim * n_heads, seq_len, batch_size) |> cu

    o_nn, α = dot_product_attention(q, k, v; nheads=n_heads)

    o_fa = mha(reshape.((q, k, v), head_dim, n_heads, seq_len, batch_size)...; is_return_softmax=true)

    @test isapprox(vec(o_nn), vec(o_fa))
end