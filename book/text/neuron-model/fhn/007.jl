function update!(variable::FHN, param::FHNParameter, Ie::Vector, dt)
    @unpack N, v, u = variable
    @unpack a, b, c = param
    @inbounds for i = 1:N
        v[i] += dt * c * (-u[i] + v[i] - v[i]^3 / 3 + Ie[i])
        u[i] += dt * (v[i] - b*u[i] + a)
    end
end