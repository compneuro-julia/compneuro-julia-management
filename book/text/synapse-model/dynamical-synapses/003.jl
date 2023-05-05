# short-term synaptic plasticity
function stsp(dt, T, s, U, τₓ, τᵤ, τₛ=30)
    nt = Int(T/dt);
    αₓ, αᵤ, αₛ = dt/τₓ, dt/τᵤ, dt/τₛ # 時定数を減衰率に変換

    u, x, r = zeros(nt), zeros(nt), zeros(nt)
    u[1], x[1] = U, 1
    
    for t in 1:nt-1
        x[t+1] = x[t] + αₓ*(1-x[t]) - u[t]*x[t]*s[t]
        u[t+1] = u[t] + αᵤ*(U-u[t]) + U*(1-u[t])*s[t]
        x[t+1], u[t+1] = clamp.([x[t+1], u[t+1]], 0, 1) # for numerical stability
        r[t+1] = (1-αₛ)*r[t] + u[t]*x[t]*s[t]/U
    end
    return u, x, r
end;