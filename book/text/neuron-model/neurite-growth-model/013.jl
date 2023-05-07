function neurite_growth_model(
        tree_info_init, seg_vec_init, nt, dt, B∞, E, S, τ, μₑ, σₑ,
        turn_rate=5, max_branch_angle=0.1π, max_turn_angle=5e-3π,
        history_num=3)
    tree_info = copy(tree_info_init)
    seg_vec = copy(seg_vec_init)
    num_branching = 0
    
    if history_num > 1
        tree_info_history = []
        seg_vec_history = []
        history_timing = floor.(Int, collect(range(1, nt, length=history_num+1)))[2:end] 
    end
    
    @showprogress for tt in 1:nt
        t = tt*dt # Current time
        
        n, C = 0.0, 0.0
        for j in 1:size(tree_info)[1]
            if tree_info[j][3] == 1
                n += 1
                C += 2 ^(-S*tree_info[j][2])
            end
        end
        C /= n
        
        for j in 1:size(tree_info)[1]
            if tree_info[j][3] == 1
                γ = tree_info[j][2]
                p_branch = branching_prob(t, dt, γ, n, C, B∞, E, S, τ)
                if p_branch > rand()
                    # Neurite branching
                    num_branching += 1
                    branch_lens = (μₑ .+ randn(2) * σₑ)*dt　# branch length
                    branch_angles = [1, -1] .* rand(Uniform(0, max_branch_angle), 2)
                    for k in 1:2 # add two daughter branches
                        push!(tree_info, [j, γ+1, 1]) 
                        push!(seg_vec, [branch_lens[k], seg_vec[j][2]+branch_angles[k]])
                    end
                    tree_info[j][3] = 0 # reset centrifugal order of original branch
                else
                    # Neurite elongation
                    Δlen = (μₑ .+ randn() * σₑ)*dt
                    p_turn = turn_rate*Δlen
                    if p_turn > rand()
                        push!(tree_info, [j, γ, 1]) # add segment
                        seg_angle = seg_vec[j][2] + rand(Uniform(-max_turn_angle, max_turn_angle))
                        push!(seg_vec, [Δlen, seg_angle])
                        tree_info[j][3] = 0 # reset centrifugal order of original branch
                    else
                        seg_vec[j][1] += Δlen
                    end
                end
            end
        end
        if history_num > 1
            if tt in history_timing
                push!(tree_info_history, copy(tree_info))
                push!(seg_vec_history, copy(seg_vec))
            end
        end
    end
    println("Num. branching: ", num_branching)
    if history_num > 1
        return tree_info_history, seg_vec_history
    else
        return tree_info, seg_vec
    end
end;