
# 強化学習
### 強化学習の目的
本章で扱う**強化学習** (reinforcement learning, RL) では環境 (environment) と，その中で行動するエージェント (agent) という概念が導入される．環境とは，エージェントが相互作用する対象であり，エージェントの行動によってその状態が変化するものである．一方，エージェントは環境内で行動を選択し，学習を行う主体（例えば生物やロボットなど）を意味する．エージェントは環境内で行動し，状態と行動に応じて**報酬** (reward) を得る．強化学習ではエージェントには望ましい行動が教師信号として与えられない代わりに，この報酬が与えられる．強化学習の目的は，エージェントが環境との相互作用を行い，結果として得られる報酬をより多く獲得する（目標を達成する）ために行動の選択を調整することである．

### 状態と行動
環境とエージェントの状態 (state) を $s\in \mathcal{S}$とし，エージェントの行動 (action) を $a \in \mathcal{A}$ とする．ここで，$\mathcal{S}$は環境とエージェントのあらゆる可能な状態の集合であり，$\mathcal{A}$ はエージェントが選択できる行動の集合である．状態や行動は離散的または連続的であり得る．

状態と行動が離散的である例として，グリッド状の迷路の探索課題が挙げられる．この場合，環境は迷路全体を指し，状態集合 $\mathcal{S}$ は迷路内におけるエージェントの位置からなり，行動集合 $\mathcal{A}$ は，$\{上, 下, 左, 右\}$ の4つの移動方向からなる．移動しない（その場で待つ）ことが行動集合に含まれる場合もある．

状態と行動が連続的である例としては，動物の歩行が挙げられる．この場合，環境は動物を取り巻くすべての要素を指し，エージェントは動物（厳密にはその神経系）に相当する．状態集合 $\mathcal{S}$ は環境の状態（地面や大気の状態など）に加え，動物自身の状態（環境内での位置や体の各部位の配置など）が含まれる．一方，行動集合 $\mathcal{A}$ は特定の筋肉の筋緊張の強弱などで表される．

### 報酬
エージェントは行動の結果として，状態に応じた報酬を得る．この報酬は正にも負にもなり得る．望ましい行動をとった場合には正の報酬が得られ，望ましくない行動をとった場合には負の報酬，すなわち罰 (punishment) が与えられる．報酬は即時に得られることもあれば，長期的な成果としてもたらされることもある．

具体例として，動物の歩行を考えてみよう．正の報酬としては，移動先で得られる水や餌（食料）などがある．一方，負の報酬には，歩行による疲労（エネルギー消費）や痛み（筋肉痛，障害物との接触，外敵の攻撃など）が含まれる．

生物においては，環境や自身の状態からさまざまな要素が報酬として与えられ，その生物（エージェント）がすべての報酬を明示的に設定する必要はない．しかし，強化学習の枠組みでは，エージェントに課題を解かせるために，人間が適切に報酬を定義する必要がある．この過程を報酬設計 (reward design) と呼ぶ．例えば，迷路探索課題では，動物の歩行における報酬を抽象化し，ゴール到達時に正の報酬を与え，移動に伴って一定の負の報酬を課すといった形で報酬を設計することができる．

### マルコフ決定過程 (MDP)
これまで説明した状態・行動・報酬の遷移について考えよう．エージェントが状態 $s_t$ において行動 $a_t$ をとると，状態 $s_{t+1}$ に遷移し，報酬 $r_{t+1}$ を受け取る \footnote{状態 $s_t$ において行動 $a_t$ を行った後に受け取る報酬を$r_t$ とする流派もある．}．状態 $s_{t+1}$ と報酬 $r_{t+1}$ が直前の状態 $s_t$ と行動 $a_t$ のみに依存し，過去の状態や行動の履歴には依存しない場合，この過程は**マルコフ性** (Markov property) を持つと言える．このとき，環境とエージェントの状態遷移確率は

$$
\begin{equation}
p(s_{t+1}, r_{t+1} \mid s_t, a_t)
\end{equation}
$$

で表される．これは「状態 $s_t$ で行動 $a_t$ を選択した際に，次の状態が $s_{t+1}$ になり，報酬 $r_{t+1}$ を得る確率」を示している．このように状態遷移がマルコフ性を持ち，エージェントの行動が次の状態への遷移確率を決定する確率過程を**マルコフ決定過程** (Markov Decision Process, MDP) と呼ぶ．MDPが成立する，すなわち状態遷移がマルコフ性を持つためには，状態 $s_t$ が環境とエージェントの相互作用に関する十分な情報を持つ必要がある．

### 部分観測マルコフ決定過程 (POMDP)
動物は感覚器を通して外界を認識しているが，外界のすべてを認識できるわけではない．これと同様に，エージェントは環境およびエージェント自身の状態 $s_t$ を直接観測できるとは限らない．エージェントが環境およびエージェント自身から受け取る情報を**観測** (observation) $o_t$ とすると，$o_t = s_t$ の場合はMDPが成立する．

しかし，現実の多くの問題では，エージェントは $s_t$ の一部しか観測できない場合や，観測に不確実性 (uncertainty) を含む場合がある．この場合，環境は**部分観測マルコフ決定過程** (partially observable Markov decision process, POMDP) で記述される．例えば，動物が視覚経路から外部の環境を観測する場合，瞬時的には視野の範囲しか外界を観測できず，また視野の範囲の物体であっても二次元の網膜像からは物体の三次元的形状を正確に得ることはできない（形状は推論する必要があり，その過程には不確実性が含まれる）．このような状況では，エージェントは観測の不確実性を考慮し，状態に対する信念 (beliefs) を持って意思決定を行う必要がある．

### 方策
与えられた状態 $s$ に対してエージェントの行動 $a$ を決定する関数を**方策** (policy) と呼び，$\pi$ で表される．ある状態 $s$ に対して常に同じ行動 $a$ を決定する方策を決定論的方策と呼び， $a=\pi(s)$ で表される．一方で行動を確率的に決定する方策を，確率的方策と呼び，$\pi(a \mid s) = p(a \mid s)$ で表される．ここで $\pi$ のみを使用する場合は方策それ自体を意味し，$\pi(a \mid s)$ は状態 $s$ が与えられた時に $a$ を選択する確率を意味する．

### 収益
強化学習は望ましい方策を得ることが目的であるが，そのためには方策の「良さ」を評価する必要がある．単純に瞬時的な報酬 $r_t$ で方策を評価した場合，即時的には報酬が少ないが後に大きな報酬が貰えるような方策を取らなくなってしまうため，これは望ましくない．こうした，行動に対する報酬が即時に得られず，後に得られるような場合の報酬を**遅延報酬** (delayed reward) と呼ぶ．方策の評価のためには遅延報酬も含めた報酬を将来全体において累積的に評価することが必要であり，評価した値を**収益** (return) と呼ぶ．最も単純な収益 $G_t$ としては，時刻 $t+1$ 以降の報酬を加算した**累積報酬** (cumulative reward) があり，時刻 $T$ に得られる報酬までを考慮する場合は次式で表される．

$$
\begin{equation}
G_t := r_{t+1}+r_{t+2}+r_{t+3}+\cdots+r_T = \sum_{k=t+1}^{T}r_{k}
\end{equation}
$$

累積報酬は平易であるが，$T$ が大きい場合には $G_t$ が無限大に発散してしまう恐れがある．そこで，$G_t$ の発散を防ぐために**割引率** (discount factor) $\gamma\ (0\leq \gamma \leq 1)$ と呼ばれる係数で将来の報酬が減衰するようにする．

$$
\begin{equation}
G_t := r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^{T-t-1}r_T = \sum_{k=t+1}^{T}\gamma^{k-t-1} r_{k}
\end{equation}
$$

これを**割引報酬和** (discounted total reward) と呼ぶ．$T\to \infty$ の場合は $\gamma^{T-t-1}r_T \to 0$ となるため $G_t$ が発散することは防がれる．$\gamma$ が0に近い場合は短期的な報酬を重視し，1に近い場合は累積報酬のように長期的な報酬も重視して行動選択を行うこととなる．以降では，$T\to \infty$とし，無限の未来の報酬までを考慮した $G_t:=\sum_{k=t+1}^{\infty}\gamma^{k-t-1} r_{k}$ を収益として考えることとする．この場合，

$$
\begin{align}
G_t &= r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots\\
&=r_{t+1}+\gamma (r_{t+2}+\gamma r_{t+3}+\cdots)\\
&=r_{t+1} + \gamma G_{t+1}
\end{align}
$$ 

が成立する．

### 価値
方策は状態に応じて変化するため，方策 $\pi$ の収益は状態ごとに評価する必要がある．状態 $s$ から，方策 $\pi$ に従って行動を選択した場合の収益の期待値を，状態 $s$ の**価値** (value) あるいは**状態価値** (state value) と呼び，$V^\pi(s)$ で表す．MDPの場合，$V^\pi(s)$ は以下で定義される．

$$
\begin{equation}
V^\pi(s) := \mathbb{E}_\pi \left[G_t \mid s_t = s \right]=\mathbb{E}_\pi \left[\sum_{k=t+1}^{\infty}\gamma^{k-t-1} r_{k}\ \middle|\ s_t = s \right]
\end{equation}
$$

ここで，$\mathbb{E}_\pi \left[\cdot \right]$ は方策 $\pi$ に従う場合の $[\cdot]$ 内の確率変数の期待値を取ることを意味する．また，$V^\pi(\cdot)$ を**状態価値関数** (state value function) と呼ぶ．

状態価値と同様の発想で，状態 $s$ において行動 $a$ を選択した場合の価値を**行動価値** (action value)と呼ぶ．行動価値は，方策 $\pi$ に従う条件下で，状態 $s$ において行動 $a$ を選択した場合の収益の期待値として計算され，$Q^\pi (s, a)$ で表される．

$$
\begin{equation}
Q^\pi(s, a) := \mathbb{E}_\pi \left[G_t \mid s_t = s, a_t=a \right]= \mathbb{E}_\pi \left[\sum_{k=t+1}^{\infty}\gamma^{k-t-1} r_{k}\ \middle|\ s_t = s, a_t=a \right]
\end{equation}
$$

この $Q^\pi (\cdot)$ を行動価値関数 (action value function) と呼ぶ．状態 $s$における価値 $V^\pi(s)$は，状態 $s$において取る可能性のあるすべての行動 $a$ の価値 $Q^\pi(s, a)$ の期待値として次式のように表すことができる．

$$
\begin{equation}
V^\pi(s) = \sum_{a} \pi(a \mid s) Q^\pi(s, a)
\end{equation}
$$

すなわち，状態 $s$ の価値 $V^\pi(s)$ は，その状態 $s$ での各行動 $a$ の価値 $Q^\pi(s, a)$ に方策，つまり行動$a$が取られる確率 $\pi(a \mid s)$ の重みをつけた加重平均として計算できる．

### Bellman方程式

$$
\begin{align}
V^\pi(s) &= \mathbb{E}_\pi \left[G_t \mid s_t = s \right]\\
&= \mathbb{E}_\pi \left[r_{t+1} + \gamma G_{t+1} \mid s_t = s \right]\\
\end{align}
$$

## 状態価値の推定

状態 $s$ の価値の推定値を $V(s)$ とする．
モンテカルロ法とTD法の2種類がある．

$$
\begin{align}
\textrm{モンテカルロ法: } V(s_t)&\leftarrow V(s_t)+\alpha \left[G_t - V(s_t)\right]\\
\textrm{TD法: } V(s_t)&\leftarrow V(s_t)+\alpha \left[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right]
\end{align}
$$

### モンテカルロ法

$$

$$

### 時間差分学習
p.105

時間差分学習 (Temporal difference (TD) learning) は価値推定の基本的な手法である．状態 $s$ の価値 (value) を $V(s)$ で表し，状態価値関数 (state value function) と呼ぶ．

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

---
TD (Temporal difference) learningにおいて，**報酬予測誤差**(reward prediction error, **RPE**) $\delta_{i}$は次のように計算される． 

$$ 
\begin{equation}
\delta_{i}=r+\gamma V_{j}\left(x^{\prime}\right)-V_{i}(x) 
\end{equation}
$$ 

ただし，現在の状態を$x$, 次の状態を$x'$, 予測価値分布を$V(x)$, 報酬信号を$r$, 時間割引率(time discount)を$\gamma$としました．
また，$V_{j}\left(x^{\prime}\right)$は予測価値分布$V\left(x^{\prime}\right)$からのサンプルです． このRPEは脳内において主に中脳の**VTA**(腹側被蓋野)や**SNc**(黒質緻密部)における**ドパミン(dopamine)ニューロン**の発火率として表現されています．

ただし，VTAとSNcのドパミンニューロンの役割は同一ではありません．ドパミンニューロンへの入力が異なっています [(Watabe-Uchida et al., _Neuron._ 2012)](https://www.cell.com/neuron/fulltext/S0896-6273(12)00281-4)． また，細かいですがドパミンニューロンの発火は報酬量に対して線形ではなく，やや飽和する非線形な応答関数 (Hill functionで近似可能)を持ちます([Eshel et al., _Nat. Neurosci._ 2016](https://www.nature.com/articles/nn.4239))．このため著者実装では報酬 $r$に非線形関数がかかっているものもあります．

先ほどRPEはドパミンニューロンの発火率で表現されている，といいました．RPEが正の場合はドパミンニューロンの発火で表現できますが，単純に考えると負の発火率というものはないため，負のRPEは表現できないように思います．ではどうしているかというと，RPEが0（予想通りの報酬が得られた場合）でもドパミンニューロンは発火しており，RPEが正の場合にはベースラインよりも発火率が上がるようになっています．逆にRPEが負の場合にはベースラインよりも発火率が減少する(抑制される)ようになっています
    ([Schultz et al., <span style="font-style: italic;">Science.</span> 1997](https://science.sciencemag.org/content/275/5306/1593.long "https://science.sciencemag.org/content/275/5306/1593.long"); [Chang et al., <span style="font-style: italic;">Nat Neurosci</span>. 2016](https://www.nature.com/articles/nn.4191 "https://www.nature.com/articles/nn.4191"))．発火率というのを言い換えればISI (inter-spike interval, 発火間隔)の長さによってPREが符号化されている(ISIが短いと正のRPE, ISIが長いと負のRPEを表現)ともいえます ([Bayer et al., <span style="font-style: italic;">J.
    Neurophysiol</span>. 2007](https://www.physiology.org/doi/full/10.1152/jn.01140.2006 "https://www.physiology.org/doi/full/10.1152/jn.01140.2006"))．

予測価値(分布) $V(x)$ですが，これは線条体(striatum)のパッチ (SNcに抑制性の投射をする)やVTAのGABAニューロン (VTAのドパミンニューロンに投射して減算抑制をする, ([Eshel, et al., _Nature_. 2015](https://www.nature.com/articles/nature14855 "https://www.nature.com/articles/nature14855")))などにおいて表現されている． この予測価値は通常のTD learningでは次式により更新されます． 

$$ 
\begin{equation}
V_{i}(x) \leftarrow V_{i}(x)+\alpha_{i} f\left(\delta_{i}\right) 
\end{equation}
$$ 

ただし，$\alpha_{i}$は学習率(learning rate), $f(\cdot)$はRPEに対する応答関数である．生理学的には$f(\delta)=\delta$を使うのが妥当である．

TD誤差

$$
\begin{equation}
\delta_{t} = r_{t+1} + \gamma V(s_{t+1}) - V(s_{t})
\end{equation}
$$

価値の更新

$$
\begin{equation}
V(s_{t}) \leftarrow V(s_{t}) + \alpha \delta_{t}
\end{equation}
$$
---


## Q-learning & SARSA
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
Q^\pi(s, a)=r+\gamma Q^\pi(s', \pi(s'))
$$

TD誤差は

$$
\delta = Q(s, a) - (r+\gamma \max_{a'} Q(s', a'))
$$

で得られる．$\mathcal{L}=\delta^2$ とする (DQNなどでは2乗誤差ではなく，Huber損失を用いる)．$\mathcal{L}$ を小さくし，推定値 $Q(s, a)$ が TD target $r+\gamma \max_{a'} Q(s', a')$ に近づくように訓練を行う．

関数近似する場合は $\textrm{状態} \to \mathbb{R}^\textrm{行動数}$ のモデルを作成する．

単に全ての状態を入力とするのは都合が悪いので，
位置を行動に変換する．


## 勾配方策法

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
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta}\left[\frac{\partial \pi_\theta (a|s)}{\partial \theta}\frac{1}{\pi_\theta (a|s)}Q^\pi (s|a)\right]\\
&=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta (a|s)Q^\pi (s|a)\right]
\end{align}
$$

と表される．

https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html


方策が無い場合は

$$
p(\tau) = p(s_0) \prod_{t=0}^T p(s_{t+1}\mid s_t, a_t) p(a_t \mid s_t)
$$

だが，方策を考えると

$$
p(\tau \mid \theta) = p(s_0) \prod_{t=0}^T p(s_{t+1}\mid s_t, a_t) \pi_\theta (a_t \mid s_t)
$$

となる．$\log$ を取ると，

$$
\log p(\tau \mid \theta) = \log p(s_0) + \sum_{t=0}^T \left(\log p(s_{t+1}\mid s_t, a_t) + \log \pi_\theta (a_t \mid s_t)\right)
$$

となる．勾配を計算すると，

$$
\nabla_\theta \log p(\tau \mid \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t \mid s_t)
$$

となる．$\frac{d}{dx} \log(f(x))=\frac{f'(x)}{f(x)}$ より，$f'(x)=f(x)\frac{d}{dx} \log(f(x))$ である．これをLog-Derivative Trickと呼ぶ．

$$
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta\mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \right]\\
&=\nabla_\theta\int_\tau p(\tau \mid \theta) R(\tau) d\tau\\
&=\int_\tau \nabla_\theta p(\tau \mid \theta) R(\tau) d\tau\quad (\because \textrm{微分と積分の順序交換})\\
\end{align}
$$


$$
\begin{align}
&=\mathbb{E}_{\pi_\theta}\left[\frac{\partial \pi_\theta (a|s)}{\partial \theta}\frac{1}{\pi_\theta (a|s)}Q^\pi (s|a)\right]\\
&=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta (a|s)Q^\pi (s|a)\right]
\end{align}
$$



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

ベースライン付きREINFORCE

Actor-Critic法では

p.286

---
### 方策勾配法による運動学習の例
方策勾配法 (REINFORCE法) により運動学習を行うことを考えよう．

in Friedrich
$\tau = (s_0, a_0, s_1, a_1, \dots)$ はエージェントの軌跡 (trajectory)．ここでの$\tau$は時定数や時刻を意味しないことに注意．


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

とできる．この，ノイズベクトル $(\boldsymbol{\xi}_t)$ と活動ベクトル ($\mathbf{x}_t$) の外積を取って重みを更新する方法はノード摂動法と同様である．

## 内発的動機付け

maximum entropy (MaxEnt) RL
あるいは
entropy-regularized reinforcement learning

Complex behavior from intrinsic motivation to occupy future
action-state path space
https://www.nature.com/articles/s41467-024-49711-1

https://github.com/jorgeerrz/occupancy_max_paper/tree/main/code_occupancy_max

https://www.nature.com/articles/s42256-024-00829-3?

Soft Q-learning

Soft Actor-Critic

## POMDP
信念 (belief) を用いる．
belief MDP