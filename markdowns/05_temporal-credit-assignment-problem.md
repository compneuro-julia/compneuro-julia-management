# 第5章：再帰型ニューラルネットワークと経時的貢献度分配問題
本章では再帰型ニューラルネットワーク（recurrent neural networks; RNN），すなわち再帰的結合を持った発火率モデルの学習則について取り扱う．前章で扱った前方向結合のみのニューラルネットワークでは通常，入力と出力の間の遅延は考慮せず，即時的に学習することを想定する。しかしRNNや運動制御，強化学習などの動的な系では，シナプス結合，神経活動や行動の変化が，誤差や報酬という形で観測されるまでに時間的な遅れを伴う場合がある。このような状況において，「ある時点のシナプス結合，神経活動，行動等の変化が，後になって得られる誤差や報酬にどれだけ寄与したか」を明らかにし，適切に割り当てる問題を経時的貢献度分配問題（temporal credit assignment problem）と呼ぶ。時間のずれによって評価信号が遅れて到来する場合，因果の流れを遡って「いつ，どの変化が，どれほど貢献したか」を正確に見積もることが，学習の要となる。

勾配法に基づいて経時的貢献度分配をする代表的な手法として，経時的誤差逆伝播法  (backpropagation through time; BPTT) と実時間再帰学習 (real-time recurrent learning; RTRL) の2種類がある．本章ではまず，BPTTとRTRLを統合的な視点から説明し，なぜ勾配法から2種類の学習則が生じるのかについて説明する．次に，BPTTとRTRLを実装に落とし込むための詳細を説明する．最後に，BPTTとRTRLを踏まえたうえで，生理学的妥当性の高い学習則について説明を行う．

## 勾配法に基づく経時的貢献度分配：BPTTとRTRL
A Practical Sparse Approximation for Real Time Recurrent Learning

A Unified Framework of Online Learning Algorithms for
Training Recurrent Neural Networks

### RNNの構造と損失関数
まず，本節で扱うRNNの定義を行う．時刻 $t$ における入力を $\mathbf{x}_t \in \mathbb{R}^{n}$，隠れ状態を $\mathbf{h}_t \in \mathbb{R}^{d}$，出力を $\mathbf{y}_t \in \mathbb{R}^{m}$ とすると，隠れ状態と出力は

$$
\begin{align}
\mathbf{u}_t &= \mathbf{W}_{\mathrm{rec}}\mathbf{h}_{t-1} + \mathbf{W}_{\mathrm{in}}\mathbf{x}_t + \mathbf{b}\\
\mathbf{h}_t &= \left(1-\alpha\right)\mathbf{h}_{t-1} + \alpha f(\mathbf{u}_t)\\
\mathbf{a}_t &= \mathbf{W}_{\mathrm{out}}\mathbf{h}_t\\
\mathbf{y}_t &= g(\mathbf{a}_t)
\end{align}
$$  

で与えられる。ただし，$\mathbf{W}_{\mathrm{in}} \in \mathbb{R}^{d\times n}, \mathbf{W}_{\mathrm{rec}} \in \mathbb{R}^{d\times d}, \mathbf{W}_{\mathrm{out}} \in \mathbb{R}^{m\times d}$ はシナプス結合重み，$\mathbf{b} \in \mathbb{R}^{d}$ は定常項，$f(\cdot), g(\cdot)$ は活性化関数であり，$\alpha:=\frac{1}{\tau}$ は状態の更新率（時定数 $\tau$ の逆数）である \footnote{$\alpha < 1$であるRNNは，重み共有をした残差結合 (residual/skip connection) のある順伝播モデル (ResNetなど) に展開することが可能である \citep{liao2016bridging}．}．
また，状態の初期値を $\mathbf{h}_{0}=\mathbf{0}$ とする．時刻 $t$ での教師信号を $\mathbf{y}_t^*$ とすると，損失 $\mathcal{L}$ は各時刻における損失 $\mathcal{L}_t$ の和を取り，

$$
\begin{equation}
\mathcal{L} = \sum_t \mathcal{L}_t\left(\mathbf{y}_t,\mathbf{y}_t^*\right)
\end{equation}
$$  

として与えられる．

### 過去向き・未来向きの勾配和
このRNNにおける目標は損失 $\mathcal{L}$ を最小化するようにパラメータ $\theta \ \left(\in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{W}_{\mathrm{out}},\mathbf{b}\}\right)$ を最適化することである．勾配法の観点では，損失のパラメータに対する勾配 $\dfrac{\partial \mathcal{L}}{\partial \theta}$ が求まれば最適化が可能である．損失は $\mathcal{L} = \sum_t \mathcal{L}_t$ と時間方向に分解できるので，勾配は

$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t}\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}\underbrace{\frac{\partial \theta_s}{\partial \theta}}_{= \mathbf{I}}=\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}
\end{equation}
$$

と時間方向に分解できる．ここで便宜的に「時刻 $s$ に用いられたパラメータ $\theta$」を $\theta_s$ と表記した。従って，$\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ は「時刻 $t$ における損失 $\mathcal{L}_t$ の時刻 $s$ （$s\leq t$）に用いられたパラメータ $\theta_s$ に対する勾配」を意味する。また，たとえオンライン学習でパラメータを毎時刻更新する場合であっても，勾配計算においては$\theta_s$ の微小変化 $\delta \theta_s$ はそのまま現在の $\theta$ の微小変化  $\delta \theta$ に等しいと見なせるため，$\frac{\partial \theta_s}{\partial \theta} = \mathbf{I}$ が成立し，これを上式に適用している。

ここで，なぜBPTTとRTRLの2種類の学習法があるのかといえば，$\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ の二重和においてどちらの和を先に計算するかが2通りあるためである．勾配の和を取る2通りの方法は，その和の向きが過去向き (past facing) と未来向き (future facing) と呼ばれ，次のように表せる：

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta}=
\begin{dcases}
\sum_{s=1}^T\frac{\partial \mathcal{L}}{\partial \theta_s}=\sum_{s=1}^T\sum_{t=s}^T\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{未来向き; BPTT})\\
\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t=1}^T\sum_{s=1}^t\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{過去向き; RTRL})
\end{dcases}
\end{align}
$$

ただし，時刻の範囲を $1 \leq s, t \leq T\quad (s\leq t)$ と明示的にした．
現在時刻 $s$ におけるパラメータ $\theta_s$ が未来のすべての損失 $\mathcal{L}_t\ (t \geq s)$ に与える影響を合算するのが未来向き勾配であり，この形は「現在のパラメータの変更が将来の損失にどれだけ寄与するか」を評価する．一方で，現在時刻 $t$ における損失 $\mathcal{L}_t$ に対して，過去のすべてのパラメータ $\theta_s\ (s\leq t)$ が与えた影響を合算するのが過去向き勾配であり，この形は「現在の損失が過去のパラメータの適用にどれだけ依存するか」を評価する．

BPTTは未来向き勾配を用い，RTRLは過去向き勾配を用いるが，いずれの場合でも二重和をそのまま計算するのは，計算量 $\mathcal{O}(T^2)$ を要するため非効率である．このため，動的計画法 (dynamic programming; DP) を使用し，計算量を $\mathcal{O}(T)$ に削減する．なお，動的計画法はある問題を部分問題に分割し、それらの解を再利用して計算量を削減するアルゴリズム設計手法である．誤差逆伝播法も動的計画法の一種であり，運動制御や強化学習等でも動的計画法は頻繁に使用される重要な考え方である．ここで注意すべきは，「勾配評価における時間軸の方向」と「動的計画法を進める方向」は逆になる点である。例えば，未来の損失や報酬を含めた評価を動的計画法で求めるときは，未来から過去の方向に漸化式を遡る。これは第11章で説明する強化学習におけるBellman方程式やTD学習においても同様である．

動的計画法を用いるために，勾配を展開する．勾配を展開するときのルールとして，現在の状態やパラメータは過去の損失，状態，パラメータに影響しないということである．すなわち，過去の変数の，現在の変数に対する勾配は0となる．逆に，未来の変数の，現在の変数に対する勾配は意味を持つので，これと連鎖律を用いて勾配を展開する．

未来向きの場合，外側の総和の中身は

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta_t}&=\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\left(\sum_{s \geq t} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \sum_{s \geq t+1} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \sum_{s \geq t+1} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}
\end{align}
$$

となり，$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$ を未来の勾配 $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}$ で表すことができる．次に，過去向きの場合，外側の総和の中身は

$$
\begin{align}
\frac{\partial \mathcal{L}_t}{\partial \theta}&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\sum_{s\leq t} \frac{\partial \mathbf{h}_t}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \sum_{s\leq t-1} \frac{\partial \mathbf{h}_t}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \sum_{s\leq t-1} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
\end{align}
$$

となり，$\frac{\partial \mathbf{h}_t}{\partial \theta}$ を過去の勾配 $\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}$ を用いて表すことができる．これらをまとめると以下のように表すことができる：

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta}=
\begin{dcases}
\sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\sum_{t=1}^T\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}\right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
\end{dcases}
\end{align}
$$


$$
\begin{align}
\text{BPTT:}\quad \frac{\partial \mathcal{L}}{\partial \theta}&=\sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\sum_{t=1}^T\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}\right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
\text{RTRL:}\quad \frac{\partial \mathcal{L}}{\partial \theta}&=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
\end{align}
$$

### 前向き・後ろ向き自動微分
先ほどの自動微分における前後と過去未来を混同しないように注意してほしい．

前向きモード自動微分 (forward-mode differentiation) がRTRLに対応し，後ろ向きモード自動微分がBPTTに対応する．

入力から損失の向きに計算するか，損失から入力の向きに計算するか．

## 経時的誤差逆伝播法 (BPTT)
ここからはBPTTについて，各パラメータごとの具体的な勾配を求める．

出力層の誤差信号を

$$
\begin{equation}
\boldsymbol{\delta}_t^{\mathrm{out}}
:=\frac{\partial \mathcal{L}_t}{\partial \mathbf{a}_t}
=\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\frac{\partial \mathbf{y}_t}{\partial \mathbf{a}_t}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\odot g'(\mathbf{a}_t)^\top\quad \left(\in \mathbb{R}^{1\times m}\right)
\end{equation}
$$  

と定義する。ここで $\odot$ は要素積 (Hadamard product) を表す。また中間層に逆伝播する誤差は時間方向の再帰関係から

$$
\begin{equation}
\boldsymbol{\delta}_t
:=\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}=\underbrace{\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}}_{\mathclap{\substack{\text{現在時刻の}\\\text{直接寄与}}}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}}_{\mathclap{\substack{\text{次時刻以降への}\\\text{間接寄与}}}}
\quad \left(\in \mathbb{R}^{1\times d}\right)
\end{equation}
$$  

が成り立つ．ここで直接寄与項は

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{a}_t}\frac{\partial \mathbf{a}_t}{\partial \mathbf{h}_t}=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}}
\end{equation}
$$

であり，間接寄与項は

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}&=\boldsymbol{\delta}_{t+1}\left[(1-\alpha)\mathbf{I}_d+\alpha \frac{\partial f(\mathbf{u}_{t+1})}{\partial \mathbf{u}_{t+1}}\frac{\partial \mathbf{u}_{t+1}}{\partial \mathbf{h}_{t}}\right]\\
&=\left(1-\alpha\right)\boldsymbol{\delta}_{t+1} +\alpha \boldsymbol{\delta}_{t+1} \odot f'(\mathbf{u}_{t+1})^\top \mathbf{W}_{\mathrm{rec}}
\end{align}
$$

である．ここで，$\delta_{t}^\mathrm{h} := \boldsymbol{\delta}_t \odot f'(\mathbf{u}_t)^\top\ \left(\in \mathbb{R}^{1\times d}\right)$ とすると，

$$
\begin{equation}
\boldsymbol{\delta}_t
=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}} +\left(1-\alpha\right)\boldsymbol{\delta}_{t+1} +\alpha \delta_{t+1}^\mathrm{h} \mathbf{W}_{\mathrm{rec}}
\end{equation}
$$  

が成立する。ただし，境界条件として $\boldsymbol{\delta}_{T+1}=\mathbf{0}$ とする。これらを用いて各重み行列の勾配を時刻方向に和をとる形で求める。

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{out}}}
&=\sum_t \mathbf{h}_t\boldsymbol\delta_t^{\mathrm{out}}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{rec}}}
&=\alpha \sum_t \mathbf{h}_{t-1}\delta_{t}^\mathrm{h}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{in}}}
&=\alpha \sum_t \mathbf{x}_t\delta_{t}^\mathrm{h}
\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}}
&=\alpha\sum_t \delta_{t}^\mathrm{h}\\
\end{align}
$$

以上が BPTT による重み更新の基本式である。BPの時と同様に，実装時には$\delta_{t}$ は列ベクトルとなり，バッチ処理も考慮するため，転置の有無や行列積の順序は変化する．

## 実時間リカレント学習 (RTRL)
RTRL では各パラメータ $\theta\in\{\mathbf{W}_{\mathrm{rec}},\mathbf{W}_{\mathrm{in}},\mathbf{b}\}$ に対して時刻 $t$ での状態感度行列  
$$\begin{equation}
\mathbf{P}_t^{(\theta)} = \frac{\partial \mathbf{h}_t}{\partial \theta}
\end{equation}$$  
を逐次的に保持し，出力誤差と組み合わせて各時刻ごとに重み更新を行う。まず状態感度は以下の再帰式で更新される：  
$$\begin{equation}
\mathbf{P}_t^{(\theta)}
=\left(1-\alpha\right)\mathbf{P}_{t-1}^{(\theta)}
+\alpha\mathbf{D}_f(\mathbf{a}_t)
\left(\mathbf{W}_{\mathrm{rec}}\mathbf{P}_{t-1}^{(\theta)} + \mathbf{Q}_t^{(\theta)}\right),
\end{equation}
$$  

ここでパラメータ依存の入力感度 $\mathbf{Q}_t^{(\theta)}$ は  

$$
\begin{equation}
\mathbf{Q}_t^{(\mathbf{W}_{\mathrm{rec}})}=\mathbf{h}_{t-1},\quad
\mathbf{Q}_t^{(\mathbf{W}_{\mathrm{in}})}=\mathbf{x}_t,\quad
\mathbf{Q}_t^{(\mathbf{b})}=\mathbf{1}
\end{equation}
$$  

とし，$\mathbf{P}_t^{(\mathbf{W}_{\mathrm{out}})}=\mathbf{0}$ とする。一方，出力層の誤差は BPTT と同様に  
$\boldsymbol\delta_t^{\mathrm{out}}=\partial\mathcal{L}_t/\partial\mathbf{u}_t$ であるから，時刻 $t$ における各パラメータの勾配は  
$$\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \theta}
=\left(\boldsymbol\delta_t^{\mathrm{out}}\right)^\top
\frac{\partial \mathbf{u}_t}{\partial \theta}
=\left(\boldsymbol\delta_t^{\mathrm{out}}\right)^\top
\mathbf{W}_{\mathrm{out}}\mathbf{P}_t^{(\theta)},
\end{equation}$$  
ただし $\theta=\mathbf{W}_{\mathrm{out}}$ の場合は  
$$\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{\mathrm{out}}}
=\boldsymbol\delta_t^{\mathrm{out}}\mathbf{h}_t^\top.
\end{equation}$$  
このように RTRL では時刻ごとに $\mathbf{P}_t^{(\theta)}$ を更新し，それを用いて逐次的に勾配を計算するため，オンライン学習が可能となる。

以上，BPTT と RTRL の学習則を同一モデルに適用した形でまとめた。どちらも同じ損失 $\mathcal{L}$ を最小化するが，BPTT は全時刻を遡ってまとめて誤差を伝播させる一方，RTRL は逐次的に感度を保持しリアルタイムで勾配を得る点が異なる。


## 両者のトレードオフ  
過去向き方式はオンライン性が強く，一度に扱うパラメータ依存を１つの損失にまとめるため，リアルタイム更新が可能であるが，その分、「過去→現在」の微分を保持する大きなテンソル（感度行列）を圧縮する工夫が必要となる。未来向き方式は「現在→未来」の影響を直接扱うため，パラメータ感度の保持は不要だが，未来の損失を参照する逆伝播がオンラインでは難しく，しばしばトランケート（打ち切り）を伴う。  


損失に対する状態感度
状態に対するパラメータ感度

BPTTは


いずれの手法も，時系列モデルの状態更新則が  

$$
\mathbf{h}_t = F\bigl(\mathbf{h}_{t-1},\,\mathbf{x}_t;\,\theta\bigr)
$$  

のように，状態は過去から未来への一方向性を持つため，過去の状態を未来の状態で微分する操作 $\partial \mathbf{h}_{t-1}/\partial \mathbf{h}_t$ は常にゼロとなる．

$\frac{\partial \mathbf{h}_t}{\partial \theta} \in \mathbb{R}^{d \times |\theta|}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \in \mathbb{R}^{d\times d}$

状態感度 (state sensitivity) $\frac{\partial \mathbf{h}_t}{\partial \theta}$

---

## 経時的誤差逆伝播法 (BPTT)

Backpropagation through time and the brain
https://www.sciencedirect.com/science/article/pii/S0959438818302009


$$
\begin{align}
\text{入力層 : }&\mathbf{h}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{a}_\ell=\mathbf{W}_\ell \mathbf{h}_\ell +\mathbf{b}_\ell\\
&\mathbf{h}_{\ell+1}=f_\ell\left(\mathbf{a}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{h}_{L+1}
\end{align}
$$


RNN

入力を $\mathbf{x}_{t}$, 状態を $\mathbf{h}_t$, 出力を $\mathbf{y}_t$ とする．活性化関数を $f, g$ とし，時定数を $\tau$，重みを $\mathbf{W}$ とする． 
状態遷移を

$$
\mathbf{h}_{t}=\left(1-\alpha\right)\cdot \mathbf{h}_{t-1} +\alpha\cdot f(\mathbf{W}_{\mathrm{rec}}\mathbf{h}_{t-1} +\mathbf{W}_{\mathrm{in}}\mathbf{x}_{t}+\mathbf{b})
$$

出力を
$$
\mathbf{y}_t = g(\mathbf{W}_{\mathrm{out}}\mathbf{h}_t)
$$

とする．モデルを訓練するために正解 $\mathbf{y}_t$ が与えられる．

BPTT (Backpropagation through time)
backpropagation through time (BPTT) (Rumelhart et al., 1985) in order to compare it with the learning rules presented above. The derivation here follows Lecun (1988).

RTRL (Real-time recurrent learning)
Williams RJ, Zipser D. 1989. A learning algorithm for continually running fully recurrent neural networks. Neural Computation 1:270–280. DOI: https://doi.org/10.1162/neco.1989.1.2.270



RFRO (Random feedback local online learning)
Check the implementation

$$
\frac{\partial h_{j} (t )}{\partial W_{a b}} = (1 - \alpha ) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}} ,
$$

Output error

$$
\begin{align}
\epsilon (t)=\mathbf{y}(t)-\hat{\mathbf{y}}(t)\\
\mathcal{L}=\frac{1}{2T}\sum_{t=1}^T \|\epsilon (t)\|^2
\end{align}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=-\frac{1}{T}\sum_{t=1}^T \mathbf{W}_{out}^\top \epsilon (t)\frac{\partial \mathbf{h}(t)}{\partial \mathbf{W}}
$$

Update rule
$$
\begin{align}
\Delta \mathbf{W}^{out}_t&=\eta \epsilon_{t} \mathbf{h}_t\\
\Delta \mathbf{W}_{rec}(t)&=\eta \mathbf{B}\epsilon(t) \mathbf{P}(t)\\
\Delta \mathbf{W}_{in}(t)&=\eta \mathbf{B}\epsilon (t) \mathbf{Q}(t)\\
\end{align}
$$

Eligibility trace $\mathbf{P}\in \mathbb{R}^{N_{rec}\times N_{rec}}, \mathbf{Q}\in \mathbb{R}^{N_{rec}\times N_{in}}$

$$
\begin{align}
\mathbf{P}_t&=\alpha f'(\mathbf{u}_t)\mathbf{h}_{t-1}^\top+\left(1-\alpha\right)\mathbf{P}_{t-1}\\
\mathbf{Q}_t&=\alpha f'(\mathbf{u}_t)\mathbf{x}_{t-1}^\top+\left(1-\alpha\right)\mathbf{Q}_{t-1}
\end{align}
$$

----
## **BPTT（Backpropagation Through Time）とRTRL（Real-Time Recurrent Learning）の数式表現**

### **1. BPTT（Backpropagation Through Time）**
BPTTはRNN（Recurrent Neural Network）の学習において誤差逆伝播法（Backpropagation）を時間方向に適用する手法である．時系列データに対する損失関数の勾配を効率的に計算するために，時間展開したネットワーク上で逆伝播を行う．

#### **(1) 順伝播**
考えるRNNの状態更新と出力の式は、次のように書けます。

$$
\mathbf{h}_t = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\mathbf{y}_t = g(\mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y)
$$

ここで、
- $h_t$は時刻$t$の隠れ状態
- $x_t$は入力
- $y_t$は出力
- $W_h, W_x, W_y$はそれぞれの重み行列
- $b_h, b_y$はバイアス
- $f$は活性化関数（例: tanh, ReLU）
- $g$は出力関数（例: softmax）

#### **(2) 損失関数**
一般的な損失関数$\mathcal{L}$は、時系列全体の誤差の合計として定義されます。

$$
\mathcal{L} = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

ここで$\ell(y_t, \hat{y}_t)$は時刻$t$における損失。

#### **(3) 誤差逆伝播**
BPTTでは、勾配を時間方向に逆伝播させます。

##### **出力層の勾配**
出力層の重み$W_y$に関する勾配：

$$
\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial y_t} \frac{\partial y_t}{\partial W_y}
$$

##### **隠れ層の勾配**
隠れ層の重み$W_h$に関する勾配は、時間的な依存関係を考慮して連鎖律を適用します。

$$
\delta_t = \frac{\partial L}{\partial h_t} = \frac{\partial \ell_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} + \delta_{t+1} \frac{\partial h_{t+1}}{\partial h_t}
$$

重み$W_h$に関する勾配は以下のようになります。

$$
\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \delta_t \frac{\partial h_t}{\partial W_h}
$$

BPTTでは、計算量削減のために「一定の時間範囲（トランケーション）」のみを考慮する **Truncated BPTT** もよく用いられます。

---

### **2. RTRL（Real-Time Recurrent Learning）**
RTRLは、RNNの各時刻の重み更新をリアルタイムに行う学習アルゴリズムです。BPTTとは異なり、時間展開をせずに、各時刻での勾配を逐次更新します。

#### **(1) 隠れ状態の再掲**
$$
h_t = f(W_h h_{t-1} + W_x x_t + b_h)
$$

#### **(2) 隠れ状態の勾配伝播**
RTRLでは、すべての時刻で重み$W_h$に対する状態の変化率を直接追跡します。

$$
S_t = \frac{\partial h_t}{\partial W_h}
$$

この$S_t$を逐次的に更新する式は以下のようになります。

$$
S_t = f'(W_h h_{t-1} + W_x x_t + b_h) \left( W_h S_{t-1} + \frac{\partial h_t}{\partial W_h} \right)
$$

これを利用して、損失$L$に関する勾配を計算します。

$$
\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} S_t
$$

---

### **BPTTとRTRLの比較**
| 項目 | BPTT | RTRL |
|------|------|------|
| 勾配計算 | 時間展開後に逆伝播 | リアルタイムに勾配更新 |
| 計算コスト | 高い（長期依存の計算が可能） | 非常に高い（$O(T n^4)$の計算量） |
| メモリ使用量 | 長期依存を扱う場合多い | 少ない（即時更新） |
| 応用 | 深層学習で一般的（LSTM, GRU） | ほぼ使われない（計算コストの問題） |

通常、計算コストの問題でRTRLはほとんど使用されず、BPTTやTruncated BPTTが主流です。


##

A Practical Sparse Approximation for
Real Time Recurrent Learning

BPTT

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\sum_{t=1}^T\left(\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}=\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}
$$


RTRL

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
$$

$$
\frac{\partial \mathbf{h}_t}{\partial \theta}=\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}
$$

$$
\frac{d\mathcal{L}_t}{d \theta}=\frac{d\mathcal{L}_t}{d h_t}\frac{d h_t}{d \theta}
$$

$h_t=f(x_t, h_{t-1}, \theta)$を全微分すると，

$$
\frac{dh_t}{d\theta}=\frac{\partial h_t}{\partial h_{t-1}}\frac{d h_{t-1}}{d \theta}+\frac{\partial h_t}{\partial x_t}\frac{d x_t}{d \theta}+\frac{\partial h_t}{\partial\theta}=\frac{\partial h_t}{\partial h_{t-1}}\frac{d h_{t-1}}{d \theta}+\frac{\partial h_t}{\partial\theta}
$$

$x_t$は$\theta$に依存しない．全微分と偏微分に注意すると，


RFLOに記載

$$
\Delta \mathbf{W}_{\textrm{out}} = \frac{\eta}{T} \sum_{t=1}^T \epsilon(t) h(t)^\top 
$$

$$
\Delta \mathbf{W}_{\textrm{rec}} = \frac{\eta}{T} \sum_{t=1}^T \mathbf{W}_{\textrm{out}}^\top \epsilon(t) \frac{\partial h(t)}{\partial \mathbf{W}_{\textrm{rec}}}
$$


## 実時間リカレント学習 (RTRL)
## 適格度トレースによるRTRLの近似※

http://frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full

## 合成勾配学習
DNI

Online Learning via Synthetic Gradients

Decoupled Neural Interfaces (DNI)
Jaderberg et al. (2017)

合成勾配，すなわち損失に対するパラメータの勾配を予測することと同一ではないが，
運動において，結果が分かる前に誤差を推定することがある．
この場合は，損失に対する状態の勾配である．

これはBellman方程式と同じである．

誤差や勾配を推定することで

DNI(B), DNI(lambda), 小脳・大脳連関

Decoupled Neural Interfaces（DNI）とは，深層ニューラルネットワークの層間における誤差逆伝播（backpropagation）の依存関係を緩和し，各層の重み更新を部分的に「分離」して行うことで，並列化および生体適合性を高めようとする学習手法である。従来の誤差逆伝播法では，出力層から入力層に向かって誤差信号を逐次的に伝搬させる必要があり，すべての中間層は前段の勾配情報を待ってから自らのパラメータ更新を行う。この逐次依存性は計算グラフの層深が深くなるほど同期のボトルネックとなり，特に生体ニューラル回路や大規模分散システムへの実装に際して問題となる。  

DNIはこの問題に対して，各中間層に「合成勾配（synthetic gradient）」と呼ばれる局所的な誤差信号を予測する小さな補助モデルを付与する。具体的には，ある層ℓの出力 $h^{(\ell)}$ を受けて，補助モデル $M^{(\ell)}$ がその後段で生じるであろう真の勾配 $\partial \mathcal{L}/\partial h^{(\ell)}$ を予測し，これを用いて層ℓの重みを即時に更新する。すなわち，真の勾配が上流から到着するのを待たず，予測された合成勾配 $\hat g^{(\ell)} = M^{(\ell)}(h^{(\ell)})$ に基づいてパラメータ $\theta^{(\ell)}$ を更新することで，学習プロセスの層間同期を解消する。  

合成勾配モデル $M^{(\ell)}$ は通常小規模な多層パーセプトロンで実装され，真の勾配が利用可能になった後にその予測を教師信号として自身も学習する。すなわち，補助モデルは損失関数  
$$
\mathcal{L}_{\rm synth}^{(\ell)} = \bigl\|\hat g^{(\ell)} - g^{(\ell)}\bigr\|^2
$$ 
を最小化するように訓練される。この二重学習構造により，各メインモデルの層は独立に,—「decoupled」— 自らの補助モデルから供給される勾配情報だけで重み更新できるため，層間の待ち時間が排除され，完全な非同期分散学習が可能となる。  

DNIのメリットは第一に計算効率の向上であり，層深ネットワークのスケーラビリティが改善される点である。各層は逐次的な勾配伝搬の待機を必要としないため，ハードウェアパイプラインや分散ノード間で並列に学習更新を実行できる。第二に生物的妥当性の向上が期待される。生体神経回路では長距離の逆伝播による誤差信号の伝送機構は実証されておらず，DNIは局所的予測によって学習信号を得る点で神経回路の活動様式に近い可能性を示す。  

一方で合成勾配の予測誤差が大きい場合，メインモデルの学習が不安定化するリスクがあるため，補助モデルの設計や真の勾配との整合性をいかに保つかが重要となる。また，補助モデル自身の追加パラメータがオーバーヘッドとなるため，メモリおよび計算コストのトレードオフを慎重に評価する必要がある。これらの課題に対しては，合成勾配の正則化や補助モデルの軽量化手法，さらには真の勾配とのハイブリッド学習スケジュールの導入などが提案されている。  

まとめると，Decoupled Neural Interfacesは誤差逆伝播の逐次的制約を局所予測によって解除し，同期なし非同期学習を可能にする枠組みであり，大規模分散学習および生物的学習メカニズムの解明に向けた有力な手法として注目されている。