@kwdef mutable struct BayesianLinearReg
    μ̂::Array
    Σ̂::Array
end

# Training params & definition of model
function BayesianLinearReg(ϕ, y, α, β)
    Σ̂ = inv(α * I + β * ϕ' * ϕ)
    μ̂ = β * Σ̂  * ϕ' * y;
    return BayesianLinearReg(μ̂=μ̂, Σ̂=Σ̂)
end;

function predict(ϕ, blr::BayesianLinearReg, β)
    @unpack μ̂, Σ̂ = blr
    μp = ϕ * μ̂
    σp = sqrt.(1/β .+ diag(ϕ * Σ̂ * ϕ'));
    return μp, σp
end;

function sampling_func(ϕ, blr::BayesianLinearReg, num_sampling::Int)
    @unpack μ̂, Σ̂ = blr
    dist = MvNormal(μ̂, Matrix(Hermitian(Σ̂)))
    sampled_params = rand(dist, num_sampling);
    return ϕ * sampled_params 
end;