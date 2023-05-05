# simulation
uf, xf, rf = stsp(dt, T, s, Uf, τfₓ, τfᵤ)
ud, xd, rd = stsp(dt, T, s, Ud, τdₓ, τdᵤ);

# compute synaptic efficacy
xuf = uf .* xf / Uf 
xud = ud .* xd / Ud; 