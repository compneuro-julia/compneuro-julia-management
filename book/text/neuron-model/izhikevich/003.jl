@kwdef struct IZParameter{FT}
    C::FT = 100  # 膜容量 (pF)
    a::FT = 0.03; b::FT = -2 # 回復時定数の逆数 (1/ms), uのvに対する共鳴度合い (pA/mV)
    d::FT = 100; k::FT = 0.7 # 発火で活性化される正味の外向き電流 (pA), ゲイン (pA/mV)
    vthr::FT = -40; vrest::FT = -60; vreset::FT = -50; vpeak::FT = 35 #　閾値電位, 静止膜電位, リセット電位, ピーク電位 (mV)
end

@kwdef mutable struct IZ{FT}
    param::IZParameter = IZParameter{FT}()
    N::UInt32
    v::Vector{FT} = fill(param.vrest, N); u::Vector{FT} = zeros(N)
    fire::Vector{Bool} = zeros(Bool, N)
end