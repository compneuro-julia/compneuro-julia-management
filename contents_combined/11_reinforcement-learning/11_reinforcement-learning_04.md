## 方策ベース法
### 勾配方策法

Q学習やSARSA等は価値ベース法

方策

最適化する目的関数（方策の評価指標）はエピソード全体の期待報酬として次のように定義される

$$
\begin{align}
J(\theta) &= \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right]\\
&=\mathbb{E}\left[\sum_{t=1}^\infty \gamma^{t-1}r_t\ \middle|\ s=s_0\right]
\end{align}
$$

ここで，$R(\tau)$ は累積報酬を表す．

$$
R(\tau)=\left[\right]
$$


勾配上昇法 (gradient ascent) により，

$$
\delta \theta = \eta \nabla_\theta J(\theta)
$$

$$
\nabla_\theta J(\theta)=\frac{\partial J(\theta)}{\partial \theta}
$$

方策勾配定理 (policy gradient theorem) は

$$
\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta}\left[\frac{\partial \pi_\theta (a|s)}{\partial \theta}\frac{1}{\pi_\theta (a|s)}q_\pi (s|a)\right]\\
&=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta (a|s)q_\pi (s|a)\right]
\end{align}
$$

と表される．

https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

方策を考えると

$$
p(\tau \mid \theta) = p(s_0) \prod_{t=0}^T p(s_{t+1}\mid s_t, a_t) \pi_\theta (a_t \mid s_t)
$$

となる．$\log$ を取ると，

$$
\log p(\tau \mid \theta) = \log p(s_0) + \sum_{t=0}^T \left(\log p(s_{t+1}\mid s_t, a_t) + \log \pi_\theta (a_t \mid s_t)\right)
$$

となる．勾配を計算すると，$\theta$ を含まない項は消え，

$$
\nabla_\theta \log p(\tau \mid \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t \mid s_t)
$$

となる．$\nabla_x \log f(x)=\dfrac{\nabla_x f(x)}{f(x)}$ より，$\nabla_x f(x)=f(x)\nabla_x \log f(x)$ である．これは対数微分によるトリック (log-derivative trick) と呼ばれる．これを用いると，$f(x)$ を微分演算の外に取り出すことができる．

$$
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta\mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right]\\
&=\nabla_\theta \sum_\tau p(\tau \mid \theta) R(\tau)\\
&=\sum_\tau \nabla_\theta\ p(\tau \mid \theta) R(\tau)\quad (\because \textrm{微分と総和の順序交換})\\
&=\sum_\tau p(\tau \mid \theta) \nabla_\theta\log p(\tau \mid \theta) R(\tau)\quad (\because \textrm{対数微分によるトリック})\\
&=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\nabla_\theta \log p(\tau \mid \theta) R(\tau)\right]\quad (\because \textrm{期待値での表現に変換})\\
&=\mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t \mid s_t) R(\tau)\right]
\end{align}
$$


モンテカルロ近似により，$M$ をエピソード数，$T$ を時間ステップ数とすると，

$$
\nabla_\theta J(\theta) \approx \frac{1}{M} \sum_{m=1}^M \frac{1}{T} \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_t^m|s_t^m)q_\pi (s_t^m, a_t^m)
$$

となる．

### REINFORCE
REINFORCE法では即時報酬を用いて

$$
q_\pi (s_t^m, a_t^m) \approx R_t^m
$$

と近似する．

ベースライン付きREINFORCE

Actor-Critic法では

p.286

---
### 方策勾配法による運動学習の例
方策勾配法 (REINFORCE法) により運動学習を行うことを考えよう．

in Friedrich
$\tau = (s_0, a_0, s_1, a_1, \dots)$ はエージェントの軌道 (trajectory)．ここでの$\tau$は時定数や時刻を意味しないことに注意．


制御重みを$K$として，制御器を$\pi_K(u|x)$とする．軌道$\tau$に対するコストを$c(\tau)$とする．

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

とする．確率的制御器 $\pi_K(\mathbf{u} \mid \mathbf{x})$ を

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

とできる．この，ノイズベクトル $(\boldsymbol{\xi}_t)$ と活動ベクトル ($\mathbf{x}_t$) の外積を取って重みを更新する方法はノード摂動法と同様である．

### Actor-Critic法