function update!(neuron::FHN, x::Vector)
    @unpack num_neurons, dt, v, u = neuron
    @unpack a, b, c = neuron.param
    @inbounds for i = 1:num_neurons
        v[i] += dt * c * (-u[i] + v[i] - v[i]^3 / 3 + x[i])
        u[i] += dt * (v[i] - b*u[i] + a)
    end
    return v
end

(layer::Layer)(x) = update!(layer, x)