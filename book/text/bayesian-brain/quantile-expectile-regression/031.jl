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