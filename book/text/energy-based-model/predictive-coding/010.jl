function update!(variable::RaoBallard1999Model, param::RBParameter, inputs::Array, training::Bool)
    @unpack num_units_lv0, num_units_lv1, num_units_lv2, num_lv1, k₂, r, rh, U, Uh = variable
    @unpack α, αh, σ⁻², σ⁻²td, k₁, λ = param

    r_reshaped = r[:] # (96)

    fx = r * U' # (3, 256)
    fxh = Uh * rh # (96, )

    # Calculate errors
    error = inputs - fx # (3, 256)
    errorh = r_reshaped - fxh # (96, ) 
    errorh_reshaped = reshape(errorh, (num_lv1, num_units_lv1)) # (3, 32)

    g_r = α * r ./ (1.0 .+ r .^ 2) # (3, 32)
    g_rh = αh * rh ./ (1.0 .+ rh .^ 2) # (64, )

    # Update r and rh
    dr = k₁ * (σ⁻² * error * U - σ⁻²td * errorh_reshaped - g_r)
    drh = k₁ * (σ⁻²td * Uh' * errorh - g_rh)
    
    r[:, :] += dr
    rh[:] += drh
    
    if training 
        U[:, :] += k₂ * (σ⁻² * error' * r - num_lv1 * λ * U)
        Uh[:, :] += k₂ * (σ⁻²td * errorh * rh' - λ * Uh)
    end

    return error, errorh, dr, drh
end