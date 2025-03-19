
## Q-learning
行動価値関数 $Q(s, a)$ を


## 勾配方策法

最適化する目的関数（方策の評価指標）はエピソード全体の期待報酬として次のように定義される

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right]
$$

ここで，$\tau = (s_0, a_0, s_1, a_1, \dots)$ はエージェントの状態，$R(\tau)$ は累積報酬を表す．

$$
\delta \theta = -\eta \nabla_\theta J(\theta)
$$

$$
\nabla_\theta J(\theta)=\frac{\partial J(\theta)}{\partial \theta}
$$
方策勾配定理 (policy gradient theorem) は

$$
\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta}\left[\frac{\partial \pi_\theta (a|s)}{\partial \theta}\frac{1}{\pi_\theta (a|s)}Q^\pi (s|a)\right]\\
&=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta (a|s)Q^\pi (s|a)\right]
\end{align}
$$

と表される．

モンテカルロ近似により，$M$ をエピソード数，$T$ を時間ステップ数とすると，

$$
\nabla_\theta J(\theta) \approx \frac{1}{M} \sum_{m=1}^M \frac{1}{T} \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_t^m|s_t^m)Q^\pi (s_t^m, a_t^m)
$$

となる．

### REINFORCE
REINFORCE法では即時報酬を用いて

$$
Q^\pi (s_t^m, a_t^m) \approx R_t^m
$$

と近似する．

Actor-Critic法では


