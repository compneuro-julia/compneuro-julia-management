history_num = 3 # 記録する状態の数; 等間隔で記録．
init_branch_num = 2 # 初めの神経突起枝の数

# 細胞体のパラメータ
tree_info_init = [[1, 0, 0]]
seg_vec_init = [[0.0, 0.0]]

# 初期神経突起のパラメータ （神経突起が放射状に出るように設定）
Random.seed!(0)
for i in 1:init_branch_num
    push!(tree_info_init, [1, 0, 1])
    push!(seg_vec_init, [(μₑ + randn()*σₑ)*dt, (i-1)/init_branch_num*2π+1e-2*randn()])
end

@time tree_info_history, seg_vec_history = neurite_growth_model(tree_info_init, seg_vec_init, nt, dt, B∞, E, S, τ, μₑ, σₑ,
                                                                turn_rate, max_branch_angle, max_turn_angle, history_num);