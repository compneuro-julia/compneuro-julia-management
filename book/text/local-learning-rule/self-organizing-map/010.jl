function SOM(v, init_w; α0=1.0, σ0=6, T=500, dist_mat=nothing, return_history=true)
    # α0: update rate, σ0 : width, T : training steps
    w = copy(init_w)
    M = size(init_w)[1]
    map_width = Int(sqrt(M))
    N = size(v)[1]
    
    if return_history
        w_history = [copy(init_w)] # history of w
    end
    
    if dist_mat == nothing
        pos = hcat([[i, j] for i in 1:map_width for j in 1:map_width]...)
        dist_mat = hcat([sum((pos .- pos[:, i]) .^2, dims=1)' for i in 1:M]...); #'
    end
    
    @showprogress for t in 1:T
        α = α0 * (1 - t/T); # update rate
        σ = max(σ0 * (1 - t/T), 1); # decay from large to small (linearly decreased, avoid zero)
        exp_dist_mat = exp.(-dist_mat / (2.0(σ^2)))
        exp_dist_mat ./= maximum(sum(exp_dist_mat, dims=1))
        # loop for the N inputs
        for i in 1:N
            dist = sum((v[i, :]' .- w).^2, dims=2) # distance between input and neurons
            win_idx = argmin(dist)[1] # winner index
            # update the winner & neighbor neuron
            η = α * exp_dist_mat[win_idx, :]
            w[:, :] += η .* (v[i, :]' .- w)
        end
        if return_history
            append!(w_history, [copy(w)]) # save w
        end
    end
    if return_history
        return w_history
    else
        return w
    end
end;