for step in 1:num_steps # 25000 steps
    # 報酬がrandomに選ばれる
    reward = sample(juice_amounts, pweights(juice_probs), 1) #(1, ) StatsBase.jl参照
     
    # 報酬誤差(step毎に更新) reward応答をlinearとする
    δ = reward .- distribution # (200, )
 
    # deltaが負なら1, 正なら0
    valence = δ .≤ 0 # (200, )
 
    # 予測価値分布の更新
    α = valence .* α₋ + (1. .- valence) .* α₊
    distribution += α .* response_func.(δ) * base_lrate
end