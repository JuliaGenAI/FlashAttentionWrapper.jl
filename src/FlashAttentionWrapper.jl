module FlashAttentionWrapper

include("install.jl")
include("rrule.jl")

function __init__()
    try
        torch[] = pyimport("torch")
        fa[] = pyimport("flash_attn_2_cuda")
    catch e
        @error "Python packages [torch and/or flash_attn_2_cuda] not found. Please install the dependencies by running FlashAttentionWrapper.install() and restart the REPL"
    end
end

end # module FlashAttentionWrapper
