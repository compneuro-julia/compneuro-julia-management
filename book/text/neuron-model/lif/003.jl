@kwdef struct LIFParameter{FT}
    tref::FT = 2; tc_m::FT = 10 # 不応期, 膜時定数 (ms)
    vrest::FT = -60; vreset::FT = -65; vthr::FT = -40; vpeak::FT = 30 #　静止膜電位, リセット電位, 閾値電位, ピーク電位 (mV)
end

@kwdef mutable struct LIF{FT}
    param::LIFParameter = LIFParameter{FT}()
    N::UInt32 #ニューロンの数
    v::Vector{FT} = fill(-65.0, N); v_::Vector{FT} = fill(-65.0, N) # 膜電位, 発火電位も記録する膜電位 (mV)
    fire::Vector{Bool} = zeros(Bool, N) # 発火
    tlast::Vector{FT} = zeros(N) # 最後の発火時刻 (ms)
    tcount::FT = 0 # 時間カウント
end