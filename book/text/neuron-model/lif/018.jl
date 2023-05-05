R = 1.0 #膜抵抗 
tc_m, tref = 10, 2# 膜時定数, 不応期 (ms)
vrest, vthr = -60.0, -40.0 # 静止膜電位, 閾値電位 (mV)
rate_exact = zeros(N)

for i = 1:N
    z = R*Ie[i] / (R*Ie[i] + vrest - vthr)
    rate_exact[i] = (z > 0) ? 1 / (tref + tc_m * log(z)) * 1e3 : 0
end 