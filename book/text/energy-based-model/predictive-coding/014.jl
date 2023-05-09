function calculate_total_error(error, errorh, variable::RaoBallard1999Model, param::RBParameter)
    @unpack r, rh, U, Uh = variable
    @unpack α, αh, σ⁻², σ⁻²td, k₁, λ = param
    recon_error = σ⁻² * sum(error.^2) + σ⁻²td * sum(errorh.^2)
    sparsity_r = α * sum(r.^2) + αh * sum(rh.^2)
    sparsity_U = λ * (sum(U.^2) + sum(Uh.^2))
    return recon_error + sparsity_r + sparsity_U
end;