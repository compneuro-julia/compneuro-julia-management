@kwdef struct HHParameter{FT}
    Cm::FT = 1 # 膜容量(uF/cm^2)
    gNa::FT = 120; gK::FT = 36; gL::FT = 0.3 # Na+, K+, leakの最大コンダクタンス(mS/cm^2)
    ENa::FT = 50; EK::FT = -77; EL::FT = -54 # Na+, K+, leakの平衡電位(mV)
    tr::FT = 0.5; td::FT = 8 # ms
    invtr::FT = 1/tr; invtd::FT = 1/td
    v0::FT = -20 # mV
end

@kwdef mutable struct HH{FT}
    param::HHParameter = HHParameter{FT}()
    N::UInt16
    v::Vector{FT} = fill(-65, N)
    m::Vector{FT} = fill(0.05, N); h::Vector{FT} = fill(0.6, N)
    n::Vector{FT} = fill(0.32, N); r::Vector{FT} = zeros(N)
end