# Deep network
for t in 1:Nt
    # Update weights
    δ = Σyx - W₂ * W₁ * Σx
    W₁ += (W₂' * δ) * dt
    W₂ += (δ * W₁') * dt
    # SVD & save results
    Σ̂yx = W₂ * W₁ * Σx
    _, b, _ = svd(Σ̂yx)
    B[t, :] += b
end