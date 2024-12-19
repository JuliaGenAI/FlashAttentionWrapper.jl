using Random
import LuxCore

const Attention = FlashAttentionParams
export Attention

LuxCore.initialparameters(::AbstractRNG, ::Attention) = (;)
LuxCore.initialstates(::AbstractRNG, ::Attention) = (;)

(m::Attention)((q, k, v), ps, st) = mha(q, k, v; (f => getfield(m, f) for f in fieldnames(Attention))...)