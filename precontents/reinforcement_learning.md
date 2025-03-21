
## Q-learning
行動価値関数 $Q(s, a)$ を


## 勾配方策法

最適化する目的関数（方策の評価指標）はエピソード全体の期待報酬として次のように定義される

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right]
$$

ここで，$\tau = (s_0, a_0, s_1, a_1, \dots)$ はエージェントの軌跡 (trajectory)\footnote{ここでの$\tau$は時定数や時刻を意味しないことに注意．}，$R(\tau)$ は累積報酬を表す．勾配上昇法 (gradient ascent) により，

$$
\delta \theta = \eta \nabla_\theta J(\theta)
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

---
### 運動学習の例
方策勾配法 (REINFORCE法) により運動学習を行うことを考えよう．

in Friedrich

制御重みを$K$として，制御器を$\pi_K(u|x)$とする．軌跡$\tau$に対するコストを$c(\tau)$とする．

$$
J=\mathbb{E}_{\pi_K}[c(\tau)]=\mathbb{E}_{\pi_K}\left[\sum_{t=0}^T c_t\right]
$$

方策勾配定理により，状態は$s\to x$, 行動は$a\to u$として，

$$
\begin{align}
\nabla_\mathbf{K} J &= \mathbb{E}_{\pi_K}\left[\nabla_\mathbf{K} \log \pi_K (\tau)c(\tau)\right]\\
&= \mathbb{E}_{\pi_K}\left[\sum_{t=0}^T c_t \sum_{s=0}^T \nabla_\mathbf{K} \log \pi_K (u_s|x_s)\right]
\end{align}
$$

勾配降下法により，

$$
\Delta \mathbf{K} = -\eta \sum_{t=0}^T c_t \left(\sum_{s=0}^t \nabla_\mathbf{K} \log \pi_K (u_s|x_s)\right)
$$

とする．適格度トレースにより，

$$
\Delta \mathbf{K} = - c_t Z_t
$$

とし，

$$
\begin{align}
Z_t &:= \sum_{s=0}^t \nabla_\mathbf{K} \log \pi_K (u_s|x_s)\\
&=Z_{t-1}+\nabla_\mathbf{K} \log \pi_K (\mathbf{u}_t|\mathbf{x}_t)
\end{align}
$$

とする．確率的制御器を

$$
\mathbf{u}_t=-\mathbf{K}\mathbf{x}_t-\boldsymbol{\xi}_t
$$

とする．ただし，$\boldsymbol{\xi}_t\sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ とする．
よって，$\mathbf{u}_t\sim \mathcal{N}(-\mathbf{K}\mathbf{x}_t, \boldsymbol{\Sigma})$であるので，多変量正規分布の確率密度関数を考えると

$$
\pi_K(\mathbf{u}_t | \mathbf{x}_t) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}} \exp \left( -\frac{1}{2} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t) \right)
$$

となる．$d$ は $u, \xi$ の次元である．この確率密度の対数を取ると，

$$
\log \pi_K(\mathbf{u}_t | \mathbf{x}_t) = -\frac{1}{2} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t) - \frac{d}{2} \log (2\pi) - \frac{1}{2} \log |\boldsymbol{\Sigma}|
$$

となる．$K$ に関する勾配をとると，まず後ろの2項は消えるため，

$$
\nabla_\mathbf{K} \log \pi_K(\mathbf{u}_t | \mathbf{x}_t) = \nabla_\mathbf{K} \left( -\frac{1}{2} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{u}_t +\mathbf{K} \mathbf{x}_t) \right)
$$

となる．$B$が対称のとき，

$$
\frac{\partial}{\partial A}(x-As)^\top B(x-As)=-2B(x-As)s^\top
$$

が成り立つ (Matrix cookbook) ので，$s\to -\mathbf{x}_t$とすると，

$$
\begin{align}
\nabla_\mathbf{K} \log \pi_K(\mathbf{u}_t | \mathbf{x}_t) &= -\boldsymbol{\Sigma}^{-1}(\mathbf{u}_t +\mathbf{K} \mathbf{x}_t)\mathbf{x}_t^\top\\
&=\boldsymbol{\Sigma}^{-1} \boldsymbol{\xi}_t \mathbf{x}_t^\top\quad(\because \mathbf{u}_t = -\mathbf{K} \mathbf{x}_t - \boldsymbol{\xi}_t)
\end{align}
$$

となる．$\boldsymbol{\Sigma}=\sigma^2 I$とすると，

$$
\nabla_\mathbf{K} \log \pi_K(\mathbf{u}_t | \mathbf{x}_t)=\frac{1}{\sigma^2} \boldsymbol{\xi}_t \mathbf{x}_t^\top
$$

となる．よって，

$$
\begin{align}
Z_t&=Z_{t-1}+\frac{1}{\sigma^2}\boldsymbol{\xi}_t \mathbf{x}_t^\top\left(=\frac{1}{\sigma^2}\sum_{s=0}^t\boldsymbol{\xi}_s \mathbf{x}_s^\top\right)\\
\Delta \mathbf{K}_t&=-\eta c_tZ_t
\end{align}
$$

とできる．