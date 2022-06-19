using PyPlot, StatsBase
rc("axes.spines", top=false, right=false)

# Classical TD learning
N = 20
c = get_cmap("brg") 
cmap = c(range(0, 0.5, length=N))
x = range(-1, 1, step=1e-2)
θ = range(π/6, π/3, length=N)
α = tan.(θ)
y = α * x'

# Plot
figure(figsize=(5.5, 3))
subplot(1,2,1)
axvline(x=0, color="gray", linestyle="dashed", linewidth=2)
axhline(y=0, color="gray", linestyle="dashed", linewidth=2)
for i in 1:N
    if i == Int(N/2)       
        plot(x, y[i, :], color=cmap[Int(N/2), :], alpha=1, linewidth=3, label="Neutral")
    else
        plot(x, y[i, :], color=cmap[Int(N/2), :], alpha=0.2)
    end
end

ylim(-1,1); xlim(-1,1)
xticks([]); yticks([])
legend(loc="upper left")
title("Classical TD learning")
xlabel("RPE")
ylabel("Firing")

# Distributional TD learning
α_pos = tan.(θ)
α_neg = reverse!(tan.(θ))
 
yd = (α_pos * ((x .> 0) .* x)' + (α_neg) * ((x .≤ 0) .* x)') 

# Plot
ax = subplot(1,2,2)
axvline(x=0, color="gray", linestyle="dashed", linewidth=2)
axhline(y=0, color="gray", linestyle="dashed", linewidth=2)
for i in 1:N
    if i == 1        
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Pessimistic")
    elseif i == Int(N/2)       
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Neutral")
    elseif i == N
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Optimistic")
    else
        plot(x, yd[i, :], color=cmap[i, :], alpha=0.2)
    end
end
handles, labels = ax.get_legend_handles_labels()
ax.legend(reverse!(handles), reverse!(labels), loc="upper left")
ylim(-1,1); xlim(-1,1); xticks([]); yticks([])
title("Distributional TD learning"); xlabel("RPE"); ylabel("Firing")
tight_layout()

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

# Results plot
steps = 1:num_steps
figure(figsize=(6,4))
subplot(1,2,1) # 予測価値の変化

for i in 1:num_cells   
    plot(steps, dist_trans[:, i], label=(string((i+1)*25)*"%tile ("*L"$\tau=$"*string((i+1)*0.25)*")"))
end

title("Convergence of value prediction to \n percentile of reward distribution")
xlim(0, num_steps)
ylim(0, 10)
xlabel("Learning steps")
ylabel("Learned Value")
legend()

# Gaussian kernel density estimation
function kde(data, dx=0.1, band_width=1)
    x = minimum(data):dx:maximum(data)
    y = zero(x)
    n = size(data)[1]
    for i in 1:n
        y += exp.(-(((x .- data[i])/band_width).^2)/2)
    end
    y /= (n*band_width*sqrt(2π))
    return x, y
end

# 報酬のサンプリング
rewards = μreward .+ randn(2000) * σreward
qtile = nquantile(rewards, 3); # 報酬の四分位数を取得
x, y = kde(rewards);

plot(x, y)
hist(rewards, density=true, bins=50, orientation="vertical")
title("Reward\n distribution")
xlabel("Density")
tight_layout()

response_func(r) = sign.(r) # RPEの応答関数
 
juice_amounts = [0.1, 1, 2] # reward(ジュース)の量(uL)
juice_probs = [0.3, 0.6, 0.1] # 各ジュースが出る確率

num_cells = 200 # ニューロン(ユニット)の数
num_steps = 25000 # 訓練回数
base_lrate = 0.02 # ベースラインの学習率
   
distribution = zeros(num_cells) # 価値分布を記録する配列

α₊, α₋ = rand(num_cells), rand(num_cells) # RPEが正, 負のときの学習率
τ = α₊ ./ (α₊ + α₋); # Asymmetric scaling factor

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

# τの大きさでソートする
idx = sortperm(τ)
τ = τ[idx]
α₊ = α₊[idx]
α₋ = α₋[idx]
distribution = distribution[idx];

# 報酬をサンプリング
rewards = sample(juice_amounts, pweights(juice_probs), 1000)
 
# 結果の描画(価値・報酬分布)
figure(figsize=(8,4))
subplot(1,2,1) # Ground Truth (Reward分布)
title("Reward distribution")
x, y = kde(rewards, 0.01, 0.1);
plot(x,y); fill_between(x, y, zero(x), alpha=0.1)
scatter(rewards, zero(rewards), s=50, marker="|", color="k", alpha=0.5)
xlabel("Reward")
ylabel("Density")
 
subplot(1,2,2) # 学習後のValue(Reward)の分布
title("Learned Value distribution")
x, y = kde(distribution, 0.01, 0.1);
plot(x,y); fill_between(x, y, zero(x), alpha=0.1)
scatter(distribution, zero(distribution), s=50, marker="|", color="k", alpha=0.5) # rugplot
xlabel("Value")
ylabel("Density")
tight_layout()

# 結果の描画(累積分布)
figure(figsize=(8,4))
subplot(1,2,1) # 累積分布
sns.kdeplot(distribution, cumulative=True,bw=.05, label="Learned Value")
sns.kdeplot(rewards, cumulative=True, bw=.05, label="Reward (GT)")
xlabel("Reward (Learned Value)")
ylabel("Cumulative probability")
 
subplot(1,2,2) # 累積分布
plot(tau, distribution)
xlabel("Asymmetric scaling factors ("+ r"$\tau$)")
ylabel("Learned Value")
tight_layout()
show()

#collapse-hide

import scipy.stats
import scipy.optimize
 
def expectile_loss_fn(expectiles, taus, samples):
  """Expectile loss function, corresponds to distributional TD model """
  # distributional TD model: delta_t = (r + \gamma V*) - V_i
  # expectile loss: delta = sample - expectile
  delta = (samples[None, :] - expectiles[:, None])
 
  # distributional TD model: alpha^+ delta if delta > 0, alpha^- delta otherwise
  # expectile loss: |taus - I_{delta <= 0}| * delta^2
 
  # Note: When used to decode we take the gradient of this loss,
  # and then evaluate the mean-squared gradient. That is because *samples* must
  # trade-off errors with all expectiles to zero out the gradient of the 
  # expectile loss.
  indic = np.array(delta <= 0., dtype=np.float32)
  grad = -0.5 * np.abs(taus[:, None] - indic) * delta
  return np.mean(np.square(np.mean(grad, axis=-1)))
 
def run_decoding(reversal_points, taus, minv=0., maxv=1., method=None,
                 max_samples=1000, max_epochs=10, M=100):
  """Run decoding given reversal points and asymmetries (taus)."""
   
  # sort
  ind = list(np.argsort(reversal_points))
  points = reversal_points[ind]
  tau = taus[ind]
 
  # Robustified optimization to infer distribution
  # Generate max_epochs sets of samples,
  # each starting the optimization at the best of max_samples initial points.
  sampled_dist = []
  for _ in range(max_epochs):
      # Randomly search for good initial conditions
      # This significantly improves the minima found
      samples = np.random.uniform(minv, maxv, size=(max_samples, M))
      fvalues = np.array([expectile_loss_fn(points, tau, x0) for x0 in samples])
 
      # Perform loss minimizing on expectile loss (w.r.t samples)
      x0 = np.array(sorted(samples[fvalues.argmin()]))
      fn_to_minimize = lambda x: expectile_loss_fn(points, tau, x)
      result = scipy.optimize.minimize(
              fn_to_minimize, method=method,
              bounds=[(minv, maxv) for _ in x0], x0=x0)["x"]
      sampled_dist.extend(result.tolist())
 
  return sampled_dist, expectile_loss_fn(points, tau, np.array(sampled_dist))
 
 
# reward distribution
juice_amounts = np.array([0.1, 0.3, 1.2, 2.5, 5, 10, 20])
juice_empirical_probs = np.array(
    [0.06612594, 0.09090909, 0.14847358, 0.15489467,
     0.31159175, 0.1509519 , 0.07705306])
 
# samples of reward (1000, )
sampled_empirical_dist = np.random.choice(
    juice_amounts, p=juice_empirical_probs, size=1000)
 
n_trials = 10 # num of simulation trial
n_epochs = 20000 # num of simulation epoch
num_cells = 151  # num of cells or units
n_decodings = 5 # num of decodings
 
# Global scale for learning rates
beta = 0.2
 
# Distributional TD simulation and decoding
distribution = np.zeros((n_trials, num_cells))
alpha_pos = np.random.random((num_cells))*beta
alpha_neg = np.random.random((num_cells))*beta 
# alpha_neg = beta - alpha_pos としてもよい
 
# Simulation
for trial in tqdm(range(n_trials)):
    for step in range(n_epochs):
        # Sample reward
        reward = np.random.choice(juice_amounts, p=juice_empirical_probs)
        # Compute TD error
        delta = reward - distribution[trial]
        # Update distributional value estimate
        valence = np.array(delta <= 0., dtype=np.float32)
        alpha = valence * alpha_neg + (1. - valence) * alpha_pos
        distribution[trial] += alpha * delta
 
# Decoding from distributional TD (DTD) simulation
dtd_samples = [] # 
dtd_losses = [] # decoding loss
taus = alpha_pos / (alpha_pos + alpha_neg)
 
asym_variance = 0.2
 
for t in tqdm(range(n_decodings)):
    # Add noise to the scaling, but have mean 0.5 giving symmetric updates
    scaling_noise = np.tanh(np.random.normal(size=len(taus))) * asym_variance
    noisy_tau = np.clip(taus + scaling_noise, 0., 1.) # add noise
 
    # Run decoding for distributional TD
    values = run_decoding(
      distribution.mean(0), noisy_tau, 
      minv=juice_amounts.min(), maxv=juice_amounts.max(),
      max_epochs=1, M=100, max_samples=20000, method="TNC")
 
    dtd_samples.append(values[0])
    dtd_losses.append(values[1])
    # print(t, values[1]) 
 
# results of decoding
dtd_reward_decode = np.array(dtd_samples).flatten()
 
# plot
fig = figure(figsize=(8, 5))
# Ground truth
sns.kdeplot(sampled_empirical_dist, bw=.75, color="k", lw=0., shade=True)
sns.rugplot(sampled_empirical_dist, color="red", lw=2, zorder=10, label="Empirical")
 
# decoded distribution
sns.kdeplot(dtd_reward_decode, bw=.75, color=cm.plasma(0), lw=4., zorder=5, shade=False)
sns.rugplot(dtd_reward_decode, color=cm.plasma(0), label="Decoded")
for draw in dtd_samples:
  sns.kdeplot(draw, bw=.5, color=cm.plasma(0.), alpha=.5, lw=1., shade=False)
 
tick_params(top=False, right=False, labelsize=14)
legend(loc="best", fontsize=16)
xlabel("Reward", fontsize=16)
ylabel("Density", fontsize=16)
title("Distributional TD Decoding", fontsize=18)
tight_layout()
show()
