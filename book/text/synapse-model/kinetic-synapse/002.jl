dt = 1e-4 # タイムステップ (sec)
α, β = 1/5e-4, 1/5e-3
T = 0.05 # シミュレーション時間 (sec)
nt = Int(T/dt) # シミュレーションの総ステップ

r = zeros(nt)

for t in 1:nt-1
    spike = ifelse(t == 1, 1, 0)
    r[t+1] = r[t] + dt * (α*spike*(1-r[t]) - β*r[t])
end

time = (1:nt)*dt
figure(figsize=(4, 3))
plot(time, r)
xlabel("Time (s)"); ylabel("Post-synaptic current (pA)")
tight_layout()