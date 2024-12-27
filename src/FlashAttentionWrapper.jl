module FlashAttentionWrapper

include("install.jl")
include("rrule.jl")

function __init__()
    torch[] = pyimport("torch")
    fa[] = pyimport("flash_attn_2_cuda")
end

end # module FlashAttentionWrapper
