## 価値ベース法


Q-learning & SARSA

Q-learning ()

SARSA ()

state-action-reward-state-action

SARSA, Q学習では状態 $s$ で行動 $a$ を取るときの価値を $Q(s, a)$ とし，行動価値関数 (action value function)

Future rewardは


Q学習とSARSAはTD学習を元にしている．


$$
V(s_t)\leftarrow V(s_t)+\alpha[r_{t+1}+\gamma V(s_{t+1})-V(s_t)]
$$

SARSAでは

$$
Q(s_t, a_t)\leftarrow Q(s_t, a_t)+\alpha[r_{t+1}+\gamma Q(s_{t+1}, a_{t+1})-Q(s_t, a_t)]
$$


### SARSA

### Q学習

行動選択

価値ベース法


行動価値関数 $Q(s, a)$ を

累積報酬を

$$
R_{t}=\sum_{t'=t}^\infty \gamma^{t'-t}r_{t'}
$$

とする $(0\leq \gamma \leq 1)$．

状態と行動から価値を出力する真の写像 $Q^*$ がある場合，

$Q^*: \textrm{状態} \times \textrm{行動} \to \mathbb{R}$

$Q^*$が学習できれば，最適方策 $\pi^*$ は

$$
\pi^*(s)=\mathrm{argmax}_a Q^*(s, a)
$$

で得られる．実際には $Q^*$ は得られない．

Bellmann方程式

$$
q_\pi(s, a)=r+\gamma q_\pi(s', \pi(s'))
$$

TD誤差は

$$
\delta = Q(s, a) - (r+\gamma \max_{a'} Q(s', a'))
$$

で得られる．$\mathcal{L}=\delta^2$ とする (DQNなどでは2乗誤差ではなく，Huber損失を用いる)．$\mathcal{L}$ を小さくし，推定値 $Q(s, a)$ が TD target $r+\gamma \max_{a'} Q(s', a')$ に近づくように訓練を行う．

関数近似する場合は $\textrm{状態} \to \mathbb{R}^\textrm{行動数}$ のモデルを作成する．

単に全ての状態を入力とするのは都合が悪いので，
位置を行動に変換する．