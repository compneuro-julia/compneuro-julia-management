# simulation
for step in 1:num_steps # 25000 steps
    # 報酬がrandomに選ばれる
    reward = μreward + randn()*σreward
     
    # 報酬誤差(step毎に更新) reward応答をlinearとする
    δ = reward .- distribution # (3, )
 
    # δが負なら1, 正なら0
    valence = δ .≤ 0 # (3, )
 
    # 予測価値分布の更新
    α = valence .* α₋ + (1. .- valence) .* α₊
    distribution += α .* response_func.(δ) .* base_lr
    dist_trans[step, :] = distribution # 予測価値分布変化の記録
end