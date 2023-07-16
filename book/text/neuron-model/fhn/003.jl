abstract type Layer end
abstract type Neuron <: Layer end

@kwdef struct FHNParameter{FT}
    a::FT = 0.7; b::FT = 0.8; c::FT = 10.0
end

@kwdef mutable struct FHN{FT} <:Neuron
    num_neurons::UInt16
    dt::FT = 1e-2
    param::FHNParameter = FHNParameter{FT}()
    v::Vector{FT} = fill(-1.0, num_neurons) 
    u::Vector{FT} = zeros(num_neurons)
end