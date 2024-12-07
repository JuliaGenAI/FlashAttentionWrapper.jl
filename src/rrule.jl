using DLPack
using PythonCall

import ChainRulesCore as CRC
import LuxCore

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
        println("typeof Δ: $(typeof(Δ))")
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
