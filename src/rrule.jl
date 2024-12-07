using DLPack
using PythonCall

export mha, mha_varlen, mha_with_kvcache

import ChainRulesCore as CRC

const torch = Ref{Py}()
const fa = Ref{Py}()

# TODO: set rng in both forward & backward.
# Currently set to nothing

@kwdef struct FlashAttentionParams
    # nothing or (num_head,) or (num_head, batch_size)
    alibi_slopes::Union{Nothing,AbstractVector,AbstractMatrix} = nothing
    dropout::Float64 = 0.0
    softmax_scale::Union{Nothing,Float64} = nothing
    is_causal::Bool = false
    window_size_left::Int = -1
    window_size_right::Int = -1
    softcap::Float64 = 0.0
    is_return_softmax::Bool = false
    deterministic::Bool = false
end

function _mha(q, k, v; softmax_scale=nothing, kw...)
    ps = FlashAttentionParams(; softmax_scale=something(softmax_scale, size(k, 1)^(-0.5)), kw...)
    o = similar(q)
    q_py, k_py, v_py, o_py = DLPack.share.((q, k, v, o), torch[].from_dlpack)
    out_py, softmax_lse_py, p_py, rng_state_py = fa[].fwd(q_py,
        k_py,
        v_py,
        o_py,
        ps.alibi_slopes,
        ps.dropout,
        ps.softmax_scale,
        ps.is_causal,
        ps.window_size_left,
        ps.window_size_right,
        ps.softcap,
        ps.is_return_softmax,
        nothing
    )
    res = ps.is_return_softmax ? (o, DLPack.from_dlpack.((softmax_lse_py, p_py))...) : o
    res, q_py, k_py, v_py, o_py, softmax_lse_py, rng_state_py, ps
end

function CRC.rrule(::typeof(_mha), q, k, v; kw...)
    output = _mha(q, k, v; kw...)

    function _mha_pullback(Δ)
        res, q_py, k_py, v_py, o_py, softmax_lse_py, rng_state_py, ps = output
        Δout = ps.is_return_softmax ? Δ[1][1] : Δ[1]
        Δq, Δk, Δv = similar(q), similar(k), similar(v)
        Δout_py, Δq_py, Δk_py, Δv_py = DLPack.share.((Δout, Δq, Δk, Δv), torch[].from_dlpack)
        fa[].bwd(
            Δout_py,
            q_py,
            k_py,
            v_py,
            o_py,
            softmax_lse_py,
            Δq_py,
            Δk_py,
            Δv_py,
            ps.alibi_slopes,
            ps.dropout,
            ps.softmax_scale,
            ps.is_causal,
            ps.window_size_left,
            ps.window_size_right,
            ps.softcap,
            ps.deterministic,
            nothing,
            rng_state_py
        )
        (CRC.NoTangent(), Δq, Δk, Δv)
    end

    output, _mha_pullback
end

function mha(q, k, v; kw...)
    res, = _mha(q, k, v; kw...)
    res
end

#####

@kwdef struct FlashAttentionVarlenParams
    seqused_k::Union{Nothing,AbstractVector} = nothing
    leftpad_k::Union{Nothing,AbstractVector} = nothing
    block_table::Union{Nothing,AbstractMatrix} = nothing
    alibi_slopes::Union{Nothing,AbstractVector,AbstractMatrix} = nothing
    dropout::Float64 = 0.0
    softmax_scale::Union{Nothing,Float64} = nothing
    zero_tensors::Bool = false
    is_causal::Bool = false
    window_size_left::Int = -1
    window_size_right::Int = -1
    softcap::Float64 = 0.0
    is_return_softmax::Bool = false
    deterministic::Bool = false
end

function _mha_varlen(q, k, v, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k; softmax_scale=nothing, kw...)
    ps = FlashAttentionVarlenParams(; softmax_scale=something(softmax_scale, size(k, 1)^(-0.5)), kw...)
    o = similar(q)
    q_py, k_py, v_py, o_py, seqlens_q_py, seqlens_k_py = DLPack.share.((q, k, v, o, seqlens_q, seqlens_k), torch[].from_dlpack)
    out_py, softmax_lse_py, S_dmask_py, rng_state_py = fa[].varlen_fwd(
        q_py,
        k_py,
        v_py,
        o_py,
        seqlens_q_py,
        seqlens_k_py,
        ps.seqused_k,
        ps.leftpad_k,
        ps.block_table,
        ps.alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        ps.dropout,
        ps.softmax_scale,
        ps.zero_tensors,
        ps.is_causal,
        ps.window_size_left,
        ps.window_size_right,
        ps.softcap,
        ps.is_return_softmax,
        nothing
    )
    res = ps.is_return_softmax ? (o, DLPack.from_dlpack.((softmax_lse_py, S_dmask_py))...) : o
    res, q_py, k_py, v_py, o_py, seqlens_q_py, seqlens_k_py, softmax_lse_py, rng_state_py, ps
end

function CRC.rrule(::typeof(_mha_varlen), q, k, v, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k; kw...)
    res, q_py, k_py, v_py, o_py, seqlens_q_py, seqlens_k_py, softmax_lse_py, rng_state_py, ps = _mha_varlen(q, k, v, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k; kw...)

    function _mha_varlen_pullback(Δ)
        Δout = ps.is_return_softmax ? Δ[1][1] : Δ[1]
        Δq, Δk, Δv = similar(q), similar(k), similar(v)
        Δout_py, Δq_py, Δk_py, Δv_py = DLPack.share.((Δout, Δq, Δk, Δv), torch[].from_dlpack)
        fa[].varlen_bwd(
            Δout_py,
            q_py,
            k_py,
            v_py,
            o_py,
            softmax_lse_py,
            Δq_py,
            Δk_py,
            Δv_py,
            seqlens_q_py,
            seqlens_k_py,
            ps.alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            ps.dropout,
            ps.softmax_scale,
            ps.zero_tensors,
            ps.is_causal,
            ps.window_size_left,
            ps.window_size_right,
            ps.softcap,
            ps.is_return_softmax,
            ps.deterministic,
            nothing,
            rng_state_py,
        )
        (CRC.NoTangent(), Δq, Δk, Δv, CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent())
    end
end

function mha_varlen(q, k, v, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k; kw...)
    res, = _mha_varlen(q, k, v, seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k; kw...)
    res
end

#####

to_py(x) = x
to_py(x::AbstractArray) = DLPack.share(x, torch[].from_dlpack)

function mha_with_kvcache(q, kcache, vcache;
    k=nothing,
    v=nothing,
    rotary_cos=nothing,
    rotary_sin=nothing,
    cache_seqlens=nothing,
    cache_batch_idx=nothing,
    cache_leftpad=nothing,
    block_table=nothing,
    softmax_scale=nothing,
    is_causal=false,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=true,
    alibi_slopes=nothing,
    num_splits=0,
    return_softmax_lse=false
)
    o = similar(q)
    o_py, softmax_lse_py = fa[].fwd_kvcache(
        to_py(q),
        to_py(kcache),
        to_py(vcache),
        to_py(k),
        to_py(v),
        to_py(cache_seqlens),
        to_py(rotary_cos),
        to_py(rotary_sin),
        to_py(cache_batch_idx),
        to_py(cache_batch_idx),
        to_py(cache_leftpad),
        to_py(block_table),
        to_py(alibi_slopes),
        to_py(o),
        something(softmax_scale, size(q, 1)^(-0.5)),
        is_causal,
        window_size[1],
        window_size[2],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return_softmax_lse ? (o, DLPack.from_dlpack(softmax_lse_py)) : o
end