response_func(r) = sign.(r) # RPEの応答関数
 
juice_amounts = [0.1, 1, 2] # reward(ジュース)の量(uL)
juice_probs = [0.3, 0.6, 0.1] # 各ジュースが出る確率

num_cells = 200 # ニューロン(ユニット)の数
num_steps = 25000 # 訓練回数
base_lrate = 0.02 # ベースラインの学習率
   
distribution = zeros(num_cells) # 価値分布を記録する配列

α₊, α₋ = rand(num_cells), rand(num_cells) # RPEが正, 負のときの学習率
τ = α₊ ./ (α₊ + α₋); # Asymmetric scaling factor