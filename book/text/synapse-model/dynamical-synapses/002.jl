# ms
Uf, τfₓ, τfᵤ = 0.15, 200, 1500
Ud, τdₓ, τdᵤ = 0.45, 1500, 200
dt = 1
T = 4000
tarray = 1:dt:T;
nt = Int(T/dt);
s = zeros(nt) # stimuli
s[500:150:2000] .= 1;
s[2500:200:3000] .= 1;