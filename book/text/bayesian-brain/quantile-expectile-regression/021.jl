# init 
response_func(r) = sign(r) # RPEの応答関数
 
num_cells = 3 # ニューロン(ユニット)の数
num_steps = 5000 # 訓練回数
base_lr = 0.02 # ベースラインの学習率(learning rate)
 
μreward = 5 # 報酬の平均(正規分布)
σreward = 2 # 報酬の標準偏差(正規分布)
 
distribution = zeros(num_cells) # 価値分布を記録する配列
dist_trans = zeros(num_steps, num_cells) # 価値分布を記録する配列
 
α₊, α₋ = [0.1, 0.2, 0.3], [0.3, 0.2, 0.1] # RPEが正, 負のときの学習率
τ = α₊ ./ (α₊ + α₋); # Asymmetric scaling factor