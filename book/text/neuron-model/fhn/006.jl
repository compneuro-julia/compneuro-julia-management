@kwdef struct FHNParameter{FT}
    a::FT = 0.7; b::FT = 0.8; c::FT = 10.0
end

@kwdef mutable struct FHN{FT}
    param::FHNParameter = FHNParameter{FT}()
    N::UInt16
    v::Vector{FT} = fill(-1.0, N); u::Vector{FT} = zeros(N)
end