function calculate_total_error(error, r, λ)
    recon_error = mean(error.^2)
    sparsity_r = λ*mean(abs.(r)) 
    return recon_error + sparsity_r
end