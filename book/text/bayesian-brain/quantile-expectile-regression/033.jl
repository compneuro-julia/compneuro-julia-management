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