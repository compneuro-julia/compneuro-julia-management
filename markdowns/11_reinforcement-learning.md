- [第11章：強化学習](#第11章強化学習)
  - [強化学習とマルコフ決定過程](#強化学習とマルコフ決定過程)
    - [強化学習の目的](#強化学習の目的)
    - [状態と行動](#状態と行動)
    - [報酬](#報酬)
    - [マルコフ決定過程 (MDP)](#マルコフ決定過程-mdp)
    - [部分観測マルコフ決定過程 (POMDP)](#部分観測マルコフ決定過程-pomdp)
    - [方策と軌道](#方策と軌道)
    - [収益](#収益)
    - [価値](#価値)
  - [状態価値の推定](#状態価値の推定)
    - [モンテカルロ法](#モンテカルロ法)
    - [Bellman方程式とTD学習](#bellman方程式とtd学習)
    - [報酬予測誤差とドーパミン作動性ニューロン](#報酬予測誤差とドーパミン作動性ニューロン)
    - [Rescorla-Wagnerモデル](#rescorla-wagnerモデル)
    - [eligibility traceの利用とTD($\\lambda$) 則](#eligibility-traceの利用とtdlambda-則)
  - [価値ベース法](#価値ベース法)
    - [SARSA](#sarsa)
    - [Q学習](#q学習)
  - [方策ベース法](#方策ベース法)
    - [勾配方策法](#勾配方策法)
    - [REINFORCE](#reinforce)
    - [方策勾配法による運動学習の例](#方策勾配法による運動学習の例)
    - [Actor-Critic法](#actor-critic法)
  - [分布型強化学習](#分布型強化学習)
    - [sign関数を用いたDistributional RLと分位点回帰](#sign関数を用いたdistributional-rlと分位点回帰)
  - [内発的動機付け](#内発的動機付け)
  - [POMDP](#pomdp)

---
# 第11章：強化学習
## 強化学習とマルコフ決定過程
### 強化学習の目的
本章で扱う**強化学習** (reinforcement learning, RL) では環境 (environment) と，その中で行動するエージェント (agent) という概念が導入される．環境とは，エージェントが相互作用する対象であり，エージェントの行動によってその状態が変化するものである．一方，エージェントは環境内で行動を選択し，学習を行う主体（例えば生物やロボットなど）を意味する．エージェントは環境内で行動し，状態と行動に応じて**報酬** (reward) を得る．強化学習ではエージェントには望ましい行動が教師信号として与えられない代わりに，この報酬が与えられる．強化学習の目的は，エージェントが環境との相互作用を行い，結果として得られる報酬をより多く獲得する（目標を達成する）ために行動の選択を調整することである．

### 状態と行動
環境とエージェントの状態 (state) を $s\in \mathcal{S}$ とし，エージェントの行動 (action) を $a \in \mathcal{A}$ とする．ここで，$\mathcal{S}$は環境とエージェントのあらゆる可能な状態の集合であり，$\mathcal{A}$ はエージェントが選択できる行動の集合である．状態や行動は離散的または連続的であり得る．

状態と行動が離散的である例として，グリッド状の迷路の探索課題が挙げられる．この場合，環境は迷路全体を指し，状態集合 $\mathcal{S}$ は迷路内におけるエージェントの位置からなり，行動集合 $\mathcal{A}$ は，$\{上, 下, 左, 右\}$ の4つの移動方向からなる．移動しない（その場で待つ）ことが行動集合に含まれる場合もある．

状態と行動が連続的である例としては，動物の歩行が挙げられる．この場合，環境は動物を取り巻くすべての要素を指し，エージェントは動物（厳密にはその神経系）に相当する．状態集合 $\mathcal{S}$ は環境の状態（地面や大気の状態など）に加え，動物自身の状態（環境内での位置や体の各部位の配置など）が含まれる．一方，行動集合 $\mathcal{A}$ は特定の筋肉の筋緊張の強弱などで表される．

### 報酬
エージェントは行動の結果として，状態に応じた報酬 $r \in \mathbb{R}$ を得る．この報酬は正にも負にもなり得る．望ましい行動をとった場合には正の報酬が得られ，望ましくない行動をとった場合には負の報酬，すなわち罰 (punishment) が与えられる．報酬は即時に得られることもあれば，長期的な成果としてもたらされることもある．

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

### 方策と軌道
与えられた状態 $s$ に対してエージェントの行動 $a$ を決定する関数を**方策** (policy) と呼び，$\pi$ で表される．ある状態 $s$ に対して常に同じ行動 $a$ を決定する方策を決定論的方策と呼び， $a=\pi(s)$ で表される．一方で行動を確率的に決定する方策を，確率的方策と呼び，$\pi(a \mid s) = p(a \mid s)$ で表される．ここで $\pi$ のみを使用する場合は方策それ自体を意味し，$\pi(a \mid s)$ は状態 $s$ が与えられた時に $a$ を選択する確率を意味する．

次に **軌道** (trajectory) を定義する．軌道とは，あるエージェントが環境と相互作用する中で得られる状態，行動，報酬の系列全体をまとめたものであり，

$$
\begin{equation}
\tau := \{s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T, a_T, r_{T+1}, s_{T+1}\}
\end{equation}
$$  

のように表される．ここで $T$ は任意の終端時刻を表し，$s_{T+1}$ は終端状態 (terminal state) と呼ばれる．終端時刻 $T$ が有限であり，目標の達成や失敗などの条件で明確に終了する（終端状態がある）軌道は，特に **エピソード** (episode) あるいは**試行** (trial) と呼ばれる．すなわち，エピソードは終端条件を満たして終了する軌道であり，無限に続く可能性のある軌道（例えば定常方策による継続的な制御）と区別されうる．$T$ が有限の場合，方策 $\pi$ の下で，軌道（エピソード） $\tau$ を取る確率は，マルコフ性より，

$$
\begin{equation}
p(\tau) := p(s_0) \prod_{t=0}^T p(s_{t+1}, r_{t+1}\mid s_t, a_t) \pi(a_t \mid s_t)
\end{equation}
$$

と表される．ただし，$p(s_0)$ は初期状態 $s_0$ を取る確率である．

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
方策は状態に応じて変化するため，方策 $\pi$ の収益は状態ごとに評価する必要がある．状態 $s$ から，方策 $\pi$ に従って行動を選択した場合の収益の期待値を，状態 $s$ の**価値** (value) あるいは**状態価値** (state value) と呼び，$v_\pi(s)$ で表す．MDPの場合，$v_\pi(s)$ は以下で定義される．

$$
\begin{equation}
v_\pi(s) := \mathbb{E}_\pi \left[G_t \mid s_t = s \right]=\mathbb{E}_\pi \left[\sum_{k=t+1}^{\infty}\gamma^{k-t-1} r_{k}\ \middle|\ s_t = s \right]
\end{equation}
$$

ここで，$\mathbb{E}_\pi \left[\cdot \right]$ は方策 $\pi$ に従う場合の $[\cdot]$ 内の確率変数の期待値を取ることを意味する．また，$v_\pi(\cdot)$ を**状態価値関数** (state value function) と呼ぶ．

状態価値と同様の発想で，状態 $s$ において行動 $a$ を選択した場合の価値を**行動価値** (action value)と呼ぶ．行動価値は，方策 $\pi$ に従う条件下で，状態 $s$ において行動 $a$ を選択した場合の収益の期待値として計算され，$q_\pi (s, a)$ で表される．

$$
\begin{equation}
q_\pi(s, a) := \mathbb{E}_\pi \left[G_t \mid s_t = s, a_t=a \right]= \mathbb{E}_\pi \left[\sum_{k=t+1}^{\infty}\gamma^{k-t-1} r_{k}\ \middle|\ s_t = s, a_t=a \right]
\end{equation}
$$

この $q_\pi (\cdot)$ を行動価値関数 (action value function) と呼ぶ．状態 $s$ における価値 $v_\pi(s)$は，状態 $s$ において取る可能性のあるすべての行動 $a$ の価値 $q_\pi(s, a)$ の期待値として次式のように表すことができる．

$$
\begin{equation}
v_\pi(s) = \sum_{a} \pi(a \mid s) q_\pi(s, a)
\end{equation}
$$

すなわち，状態 $s$ の価値 $v_\pi(s)$ は，その状態 $s$ での各行動 $a$ の価値 $q_\pi(s, a)$ に方策，つまり行動 $a$ が取られる確率 $\pi(a \mid s)$ の重みをつけた加重平均として計算できる．

## 状態価値の推定
状態価値 $v_\pi(s)$ や行動価値 $q_\pi(s, a)$ は，環境との相互作用を通して推定を行う必要がある．まずは状態価値の推定について考えよう．以下では（方策 $\pi$ に従った際の）状態 $s$ の価値の推定値を $V(s)$ とする．また，終端時刻 $T$ が有限である場合のみを考える．

### モンテカルロ法
期待値を近似的に推定する手法として**モンテカルロ法** (Monte-Carlo method) がある．モンテカルロ法を用いると，一般に確率変数 $X$ および関数 $f$ がある場合，$\mathbf{E}[f(X)]$ の推定値 $\mu$ は，サンプル平均 $\mu=\frac{1}{N}\sum_{n=1}^N f(x_n)$ として与えられる．ただし，$x_n$ は$X$ の実現値（観測値）である．$x_n$ を全て保持せず，逐次的に（オンラインで）モンテカルロ推定を行う場合，

$$
\begin{equation}
\mu_{n}= \mu_{n-1}+\frac{1}{n} \left[f(x_n)-\mu_{n-1}\right]
\end{equation}
$$

と表される（$\mu_n$ は $n$ 回目の更新時の推定値である）．サンプル平均を取る手法は $X$ の分布 $p(X)$ が定常 (stationary) である場合はよいが，非定常 (non-stationary) である場合，すなわち $X$ の分布が時刻 $n$ に伴って変化する場合，過去と現在のサンプルに同様の重みを与えることは推定が悪くなる要因となりうる．このような非定常環境では$1/n$ の代わりに固定の学習率 $\alpha\ (0\leq \alpha \leq 1)$ を用い，現在のサンプルに大きな重みを与える，すなわち指数移動平均 (exponential moving average; EMA) を取る手法がより適している．

$$
\begin{equation}
\mu_{n}= \mu_{n-1}+\alpha \left[f(x_n)-\mu_{n-1}\right]
\end{equation}
$$

強化学習では，時間あるいは方策の変化に伴って状態価値や行動価値が変化する非定常環境を仮定することが多く，基本的には指数移動平均による推定を使用する．この手法を用いて状態価値 $v_\pi(s)$ を推定することを考えよう．$v_\pi(s)$ は状態 $s$ における 収益 $G_t$ の期待値であるため，1試行ごとに終端時刻まで軌道（エピソード）を記録し，各状態における $G_t$ を計算して，それにより推定値を次のように更新する方法が考えられる．

$$
\begin{equation}
V(s_t)\leftarrow V(s_t)+\alpha \left[G_t - V(s_t)\right]
\end{equation}
$$

強化学習の文脈では，この価値推定手法を指して（狭義の）モンテカルロ法と呼ぶ．モンテカルロ法には，$G_t$ が試行が終了するまで得られないという問題点がある．

### Bellman方程式とTD学習
モンテカルロ法を改善し，効率よく価値推定を行うために**Bellman方程式** (Bellman equation) を導入する．Bellman方程式は現在の価値と1ステップ将来の価値の関係を表す方程式であり，状態価値 $v_\pi(s)$ については以下が成立する．

$$
\begin{align}
v_\pi(s) &:= \mathbb{E}_\pi \left[G_t \mid s_t = s \right]\\
&= \mathbb{E}_\pi \left[r_{t+1} + \gamma G_{t+1} \mid s_t = s \right]\\
&= \mathbb{E}_\pi \left[r_{t+1} + \gamma v_\pi(s_{t+1}) \mid s_t = s \right]\\
\end{align}
$$

モンテカルロ法で使用した収益 $G_t$ の代わりに $r_{t+1} + \gamma v_\pi(s_{t+1})$ の推定値 $r_{t+1} + \gamma V(s_{t+1})$ を使用すると，$V(s)$ を推定する更新則は 

$$
\begin{equation}
V(s_t)\leftarrow V(s_t)+\alpha \left[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right]
\end{equation}
$$

となる．これを**時間差分学習** (temporal difference learning) と呼び，今後はTD学習と表記する．この $r_{t+1} + \gamma V(s_{t+1})$ と $V(s_t)$ の誤差を**報酬予測誤差** (reward prediction error, RPE) あるいは**TD誤差** (TD error) と呼び，

$$
\begin{equation}
\delta_t := r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\end{equation}
$$

で表す．報酬予測誤差 $\delta_t$ を用いると，TD学習の更新則は $V(s_t)\leftarrow V(s_t)+\alpha \delta_t$ と表記できる．

### 報酬予測誤差とドーパミン作動性ニューロン
ここでTD学習と神経科学の対応について紹介する．

大脳基底核

報酬に応答するニューロンがあるとは知られていた．

理論的枠組みは
https://www.jneurosci.org/content/16/5/1936/tab-article-info

ドーパミン作動性ニューロン (dopaminergic neurons) あるいは ドーパミンニューロン (dopamine neurons) は神経伝達物質の一種であるドーパミン (dopamine) を分泌する神経細胞であり，主に中脳の腹側被蓋野 (Ventral tegmental area, VTA) や黒質緻密部 (substantia nigra pars compacta, SNc) に分布している．

TD学習における報酬予測誤差がドーパミン作動性ニューロン (dopaminergic neurons; DA) により符号化されていることがSchultzらにより報告されている {cite:p}`Schultz1997-ih`. 

サルのVTA


ドーパミンニューロンであるとどう同定したか？

https://www.nature.com/articles/nature03015

条件刺激 (conditioned stimulus, CS) と無条件刺激 (unconditioned stimulus, US)


シミュレーションをここに入れる．Schlutz, 
北澤の再現．


A Unified Framework for Dopamine Signals across Timescales


https://www.nature.com/articles/s41593-023-01566-3

ただし，VTAとSNcのドーパミンニューロンの役割は同一ではない．ドーパミンニューロンへの入力が異なっています [(Watabe-Uchida et al., _Neuron._ 2012)](https://www.cell.com/neuron/fulltext/S0896-6273(12)00281-4)． また，細かいですがドーパミンニューロンの発火は報酬量に対して線形ではなく，やや飽和する非線形な応答関数 (Hill functionで近似可能)を持ちます([Eshel et al., _Nat. Neurosci._ 2016](https://www.nature.com/articles/nn.4239))．このため著者実装では報酬 $r$に非線形関数がかかっているものもあります．

先ほどRPEはドーパミンニューロンの発火率で表現されている，といいました．RPEが正の場合はドーパミンニューロンの発火で表現できますが，単純に考えると負の発火率というものはないため，負のRPEは表現できないように思います．ではどうしているかというと，RPEが0（予想通りの報酬が得られた場合）でもドーパミンニューロンは発火しており，RPEが正の場合にはベースラインよりも発火率が上がるようになっています．逆にRPEが負の場合にはベースラインよりも発火率が減少する(抑制される)ようになっています

https://www.nature.com/articles/nature12475



ドーパミンニューロンの短時間の光遺伝学的抑制は内因性の負の報酬予測誤差を模倣する
https://www.nature.com/articles/nn.4191 "https://www.nature.com/articles/nn.4191

発火率というのを言い換えればISI (inter-spike interval, 発火間隔)の長さによってPREが符号化されている(ISIが短いと正のRPE, ISIが長いと負のRPEを表現)ともいえます ([Bayer et al., <span style="font-style: italic;">J.
Neurophysiol</span>. 2007](https://www.physiology.org/doi/full/10.1152/jn.01140.2006 "https://www.physiology.org/doi/full/10.1152/jn.01140.2006"))．

ドーパミンニューロンの活動は報酬予測誤差のみを符号化しているわけではなく，運動 (movement), Salience, Threat等の他の要素に関しても予測誤差を計算していると報告されています．
これを一般化予測誤差という

https://www.nature.com/articles/s41593-024-01705-4


予測価値(分布) $V(x)$ですが，これは線条体(striatum)のパッチ (SNcに抑制性の投射をする)やVTAのGABAニューロン (VTAのドーパミンニューロンに投射して減算抑制をする, ([Eshel, et al., _Nature_. 2015](https://www.nature.com/articles/nature14855 "https://www.nature.com/articles/nature14855")))などにおいて表現されている．

### Rescorla-Wagnerモデル
TD学習は古典的条件付け (Classical conditioning) のモデルである，Rescorla-Wagner (RW) モデル {cite:p}`rescorla1972theory` と予測誤差に基づいて学習を進めるという点で関連がある．RWモデルは条件刺激 (CS) と無条件刺激 (US) の間

$$
\Delta V_i = \eta \left(\lambda - \sum_j V_j\right)
$$

https://www.jstage.jst.go.jp/article/janip/66/2/66_66.2.4/_pdf

### eligibility traceの利用とTD($\lambda$) 則
eligibility trace

---

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

---

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



---
## 分布型強化学習
分布型強化学習 (Distributional reinforcement learning)

TD学習の拡張である．

Dabneyら

$$
Z^\pi (s, a) = 
$$

分布型Bellmann方程式 (distributional Bellman equation) 

$$
Z(s_t) = R(s_t) + \gamma Z(s_{t+1})
$$

期待値をとると，$V(s_t)=\mathbb{E}[Z(s_t)]$ となる．

Quantileはノンパラ
PPCやDPCはパラメトリック


https://arxiv.org/abs/1710.10044

### sign関数を用いたDistributional RLと分位点回帰

それでは，なぜ予測価値 $V_i$は$\tau_i$ 分位点に収束するのでしょうか．Extended Data Fig.1のように平衡点で考えてもよいのですが，後のために分位点回帰との関連について説明します．分位点回帰については記事を書いたので先にそちらを読んでもらうと分かりやすいと思います

実はDistributional RL (かつ，RPEの応答関数にsign関数を用いた場合)における予測報酬 $V_i$の更新式は，分位点回帰(Quantile
regression)を勾配法で行うときの更新式とほとんど同じです．分位点回帰では$\delta$の関数$\rho_{\tau}(\delta)$を次のように定義します． 

$$ \rho_{\tau}(\delta)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right|\cdot |\delta|=\left(\tau-\mathbb{I}_{\delta
\leq 0}\right)\cdot \delta 
$$ 

そして，この関数を最小化することで回帰を行います．ここで$\tau$は分位点です．また$\delta=r-V$としておきます．今回，どんな行動をしても未来の報酬に影響はないので$\gamma=0$としています．

ここで， 

$$ 
\frac{\partial \rho_{\tau}(\delta)}{\partial \delta}=\rho_{\tau}^{\prime}(\delta)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot \operatorname{sign}(\delta) 
$$ 

なので，$r$を観測値とすると， 

$$
\frac{\partial \rho_{\tau}(\delta)}{\partial V}=\frac{\partial \rho_{\tau}(\delta)}{\partial \delta}\frac{\partial \delta(V)}{\partial V}=-\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\operatorname{sign}(\delta) 
$$ 

となります．ゆえに$V$の更新式は 

$$ 
V \leftarrow V - \beta\cdot\frac{\partial \rho_{\tau}(\delta)}{\partial V}=V+\beta \left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\operatorname{sign}(\delta) 
$$ 

です．ただし，$\beta$はベースラインの学習率です．個々の$V_i$について考え，符号で場合分けをすると


$$ 
\begin{cases} V_{i} \leftarrow V_{i}+\beta\cdot |\tau_i|\cdot\operatorname{sign}\left(\delta_{i}\right)
&\text { for } \delta_{i}>0\\ V_{i} \leftarrow V_{i}+\beta\cdot |\tau_i-1|\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i} \leq 0 \end{cases} 
$$ 

となります．$0 \leq
\tau_i \leq 1$であり，$\tau_i=\alpha_{i}^{+} / \left(\alpha_{i}^{+} + \alpha_{i}^{-}\right)$であることに注意すると上式は次のように書けます． 

$$ 
\begin{cases} V_{i} \leftarrow V_{i}+\beta\cdot
\frac{\alpha_{i}^{+}}{\alpha_{i}^{+}+\alpha_{i}^{-}}\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i}>0\\ V_{i} \leftarrow V_{i}+\beta\cdot
\frac{\alpha_{i}^{-}}{\alpha_{i}^{+}+\alpha_{i}^{-}}\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i} \leq 0 \end{cases} 
$$ 

これは前節で述べたDistributional
RLの更新式とほぼ同じです．いくつか違う点もありますが，RPEが正の場合と負の場合に更新される値の比は同じとなっています．

このようにRPEの応答関数にsign関数を用いた場合，報酬分布を上手く符号化することができます．しかし実際のドーパミンニューロンはsign関数のような生理的に妥当でない応答はせず，RPEの大きさに応じた活動をします．そこで次節ではRPEの応答関数を線形にしたときの話をします．

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

https://blog.dileeplearning.com/p/a-critique-of-successor-representations