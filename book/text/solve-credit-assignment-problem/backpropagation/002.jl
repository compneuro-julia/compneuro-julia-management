@kwdef mutable struct NN{FT}
    n_batch::UInt32 # batch size
    n_in::UInt32 # number of input units
    n_hid::UInt32 # number of hidden units
    n_out::UInt32 # number of output units
    
    params::Dict{Any, Any} # weights and bias
    grads::Dict{Any, Any} = Dict() # gradient of params
end;

function NN(n_batch, n_in, n_hid, n_out)
    params = Dict()
    params["W1"] = 2(rand(n_in, n_hid) .- 0.5) / sqrt(n_in)
    params["W2"] = 2(rand(n_hid, n_out) .- 0.5) / sqrt(n_hid)
    params["b1"] = zeros(1, n_hid)
    params["b2"] = zeros(1, n_out)
    return NN{Float32}(n_batch=n_batch, n_in=n_in, n_hid=n_hid, n_out=n_out, params=params)
end;