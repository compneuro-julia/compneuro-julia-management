# 報酬のサンプリング
rewards = μreward .+ randn(2000) * σreward
qtile = nquantile(rewards, 3); # 報酬の四分位数を取得
x, y = kde(rewards);