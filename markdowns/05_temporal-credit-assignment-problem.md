# 第5章：再帰型ニューラルネットワークと経時的貢献度分配問題
本章では**再帰型ニューラルネットワーク**（recurrent neural networks; RNN），すなわち再帰的結合を持った発火率モデルの学習則について取り扱う．前章で扱った前方向結合のみのニューラルネットワークでは通常，入力と出力の間の遅延は考慮せず，即時的に学習することを想定する。しかしRNNや運動制御，強化学習などの動的な系では，シナプス結合，神経活動や行動の変化が，誤差や報酬という形で観測されるまでに時間的な遅れを伴う場合がある。このような状況において，「ある時点のシナプス結合，神経活動，行動等の変化が，後になって得られる誤差や報酬にどれだけ寄与したか」を明らかにし，各変化に対して貢献度を適切に割り当てる問題を**経時的貢献度分配問題**（temporal credit assignment problem; TCAP）と呼ぶ。

勾配法に基づいて経時的貢献度分配をする代表的な手法として，**実時間再帰学習** (real-time recurrent learning; RTRL) と**経時的誤差逆伝播法**  (backpropagation through time; BPTT) の2種類がある．本章ではまず，RTRLとBPTTを勾配和の方向に基づいた統合的な視点から説明し，なぜ勾配法から2種類の学習則が得られるのかについて説明する．次に，RTRLとBPTTを用いてパラメータの勾配を具体的に計算する．最後に，RTRLとBPTTを踏まえたうえで，生理学的妥当性の高い学習則について説明を行う．

## 勾配法に基づく経時的貢献度分配：RTRLとBPTT
### RNNの構造と損失関数
まず，本節で扱うRNNの定義を行う．時刻 $t\ (1\leq t \leq T)$\footnote{隠れ状態 $\mathbf{h}_t$ についてのみ $t=0$ を定義する．} における入力を $\mathbf{x}_t \in \mathbb{R}^{n}$，隠れ状態を $\mathbf{h}_t \in \mathbb{R}^{d}$，出力を $\mathbf{y}_t \in \mathbb{R}^{m}$ とすると，隠れ状態と出力は

$$
\begin{align}
\mathbf{u}_t &= \mathbf{W}_{\mathrm{rec}}\mathbf{h}_{t-1} + \mathbf{W}_{\mathrm{in}}\mathbf{x}_t + \mathbf{b}_h\\
\mathbf{h}_t &= \left(1-\alpha\right)\mathbf{h}_{t-1} + \alpha f(\mathbf{u}_t)\\
\mathbf{a}_t &= \mathbf{W}_{\mathrm{out}}\mathbf{h}_t+ \mathbf{b}_y\\
\mathbf{y}_t &= g(\mathbf{a}_t)
\end{align}
$$  

で与えられる。ただし，$\mathbf{W}_{\mathrm{in}} \in \mathbb{R}^{d\times n}, \mathbf{W}_{\mathrm{rec}} \in \mathbb{R}^{d\times d}, \mathbf{W}_{\mathrm{out}} \in \mathbb{R}^{m\times d}$ はシナプス結合重み，$\mathbf{b} \in \mathbb{R}^{d}$ は定常項，$f(\cdot), g(\cdot)$ は活性化関数であり，$\alpha:=\frac{1}{\tau}$ は状態の更新率（時定数 $\tau$ の逆数）である \footnote{$\alpha < 1$であるRNNは，重み共有をした残差結合 (residual/skip connection) のある順伝播モデル (ResNetなど) に展開することが可能である \citep{liao2016bridging}．}．
また，状態の初期値を $\mathbf{h}_{0}=\mathbf{0}$ とする．$\alpha = 1$ の場合はElmanネットワークと同一である \citep{elman1990finding}．時刻 $t$ での教師信号を $\mathbf{y}_t^*$ とすると，損失 $\mathcal{L}$ は各時刻における損失 $\mathcal{L}_t$ の和を取り，

$$
\begin{equation}
\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t\left(\mathbf{y}_t,\mathbf{y}_t^*\right)
\end{equation}
$$  

として与えられる．

### 過去方向・未来方向の勾配和
このRNNを学習させる際の目標は，損失 $\mathcal{L}$ を最小化するようにパラメータ $\theta \in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{W}_{\mathrm{out}},\mathbf{b}_h, \mathbf{b}_y\}$ を最適化することである．勾配法の観点では，損失のパラメータに対する勾配 $\dfrac{\partial \mathcal{L}}{\partial \theta}$ が求まれば最適化が可能である．損失は $\mathcal{L} = \sum_t \mathcal{L}_t$ と時間方向に分解できるので，勾配は

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta}&=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t=1}^T\sum_{s=1}^T\frac{\partial \mathcal{L}_t}{\partial \theta_s}\underbrace{\frac{\partial \theta_s}{\partial \theta}}_{= \mathbf{I}}=\sum_{t=1}^T\sum_{s=1}^t\frac{\partial \mathcal{L}_t}{\partial \theta_s}
\end{align}
$$

と時間方向に分解できる．ここで，$s, t$ はいずれも時刻を表し，$1 \leq s, t \leq T$ である．また，便宜的に「時刻 $s$ に用いられたパラメータ $\theta$」を $\theta_s$ と表記した。従って，$\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ は「時刻 $t$ における損失 $\mathcal{L}_t$ の時刻 $s$ に用いられたパラメータ $\theta_s$ に対する勾配」を意味する。また，オンライン学習でパラメータを毎時刻更新する場合であっても，勾配計算においては $\theta_s$ の微小変化 $\delta \theta_s$ はそのまま現在の $\theta$ の微小変化  $\delta \theta$ に等しいと見なせるため，$\frac{\partial \theta_s}{\partial \theta} = \mathbf{I}$ が成立する。さらに現在のパラメータの状態は過去の損失に影響を与えないため，$s>t$ では $\frac{\partial \mathcal{L}_t}{\partial \theta_s}=\mathbf{0}$ となり，上式では $s\leq t$ の範囲の勾配のみが残っている．

ここで，なぜRTRLとBPTTの2種類の学習法が存在するのかを考えると、$\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ という二重和において、どちらの変数に対して先に和を取るかに2通りの方法があるためである。すなわち、現在時刻を $t$ または $s$ のいずれかを基準にとるかによって、内側の和を過去方向 (past-facing) または未来方向 (future-facing) に進めることができ、これに応じて勾配の和の取り方も区別される \citep{marschall2020unified}：

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta}=
\begin{dcases}
\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t=1}^T\sum_{s=1}^t\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{過去方向勾配和; e.g., RTRL})\\
\sum_{s=1}^T\frac{\partial \mathcal{L}}{\partial \theta_s}=\sum_{s=1}^T\sum_{t=s}^T\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{未来方向勾配和; e.g., BPTT})
\end{dcases}
\end{align}
$$

過去方向勾配和では、「現在時刻 $t$ における損失 $\mathcal{L}_t$ に対して、過去のすべてのパラメータ $\theta_s\ (s \leq t)$ が及ぼした影響」を合算する。すなわち、この形式では、「現在の損失が過去のパラメータの微小な変化にどれだけ応答するか」を評価することになる。一方、未来方向勾配和では、「現在時刻 $s$ におけるパラメータ $\theta_s$ が、未来のすべての損失 $\mathcal{L}_t\ (t \geq s)$ に与える影響」を合算する。したがって、この形式では、「現在のパラメータの微小な変化が将来の損失にどれだけ影響を及ぼすか」を評価することになる。結論から言えば，RTRLは過去方向勾配和，BPTTは未来方向勾配和を利用する．

勾配和をいずれの方向で取る場合であっても、二重和をそのまま計算すれば、計算量は $\mathcal{O}(T^2)$ となり、非効率である。このため、動的計画法 (dynamic programming; DP) を用いて、計算量を $\mathcal{O}(T)$ に削減するのが一般的である。ここで，動的計画法とは、問題を部分問題に分割し、それらの部分問題の解を再利用することで、全体の計算量を削減するアルゴリズム設計手法である。誤差逆伝播法（backpropagation）も動的計画法の一種に位置づけられ、また、運動制御や強化学習など、さまざまな分野で動的計画法は頻繁に用いられている。

なお，RTRLとBPTTをある程度ご存じの読者であれば，「RTRLは未来方向で，BPTTは過去方向ではないのか」という疑問が浮かぶであろう．ここで「勾配和の評価において時間を進める方向」と「動的計画法において計算を進める方向」は逆になることに注意していただきたい．例えば、未来向き勾配和のように，将来に発生する損失や報酬を考慮して損失関数を評価する場合、動的計画法を適用すると、計算は未来から過去へと逆向きに進める必要がある。このように、将来の結果をもとに現在の値を推定する構図は、強化学習のTD学習における状態価値の推定にも見られる。すなわち、状態価値を将来の累積報酬の期待値として定義した上で、その推定は未来から過去へと逆向きに進められる\footnote{TD学習を含め，強化学習は第11章で詳解する．}。

以上を踏まえ，勾配和を動的計画法を用いて計算しよう．動的計画法を適用するためには、まず勾配を適切に展開し，現在の勾配を、ひとつ前（または後）の時刻における勾配との関係で再帰的に表現する必要がある。このとき留意すべき基本原則は、「現在の状態やパラメータは、過去の損失、状態、パラメータに影響を及ぼさない」という事実である。すなわち、現在の変数に対する過去の変数の勾配は常に0となる（例えば $\frac{\partial \mathbf{h}_{t-1}}{\partial \mathbf{h}_t}=\mathbf{0}$ である）。一方で、未来の変数は現在の変数に依存するため、現在の変数が未来の損失に及ぼす影響は意味を持つ。したがって、勾配を展開する際には、未来の損失に向かう方向で連鎖律を適用していくことになる。

まず，過去方向勾配和の場合を考える．パラメータに対する即時的な損失の勾配は

$$
\begin{align}
\frac{\partial \mathcal{L}_t}{\partial \theta}&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\sum_{s\leq t} \frac{\partial \mathbf{h}_t}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \sum_{s\leq t-1} \frac{\partial \mathbf{h}_t}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \sum_{s\leq t-1} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta_s}\right)\\
&=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t} + \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
\end{align}
$$

となる．ここで感度行列 (sensitivity matrix, influence matrix) を $\mathbf{P}_t:=\dfrac{\partial \mathbf{h}_t}{\partial \theta}\in \mathbb{R}^{d \times |\theta|}$，即時的感度行列を $\tilde{\mathbf{P}}_t:=\dfrac{\partial \mathbf{h}_t}{\partial \theta_t}\in \mathbb{R}^{d \times |\theta|}$，状態遷移のヤコビ行列を $\mathbf{J}_t := \dfrac{\partial \mathbf{h}_{t}}{\partial \mathbf{h}_{t-1}} \in \mathbb{R}^{d\times d}$ とする．ただし，$|\theta|$ はパラメータの次元数を意味する．この場合，$\mathbf{P}_t$ は次のように過去から未来に向かう再帰的な関係式で表せる：

$$
\begin{equation}
\mathbf{P}_t=\tilde{\mathbf{P}}_t + \mathbf{J}_{t}\mathbf{P}_{t-1}
\end{equation}
$$

ただし，境界条件として $\mathbf{P}_{0}=\mathbf{0}$ とする。この式を用いて，$\mathbf{P}_t$ を逐次的に求め，$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}$ を即時的に計算して $\mathbf{P}_t$ に乗じれば，$\frac{\partial \mathcal{L}_t}{\partial \theta}$ が求まる．

次に，未来方向勾配和の場合を考える．即時的なパラメータに対する損失の勾配は，

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta_t}&=\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\left(\sum_{s \geq t} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \sum_{s \geq t+1} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \sum_{s \geq t+1} \frac{\partial \mathcal{L}_s}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
&=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}
\end{align}
$$

となる．ここで貢献度分配ベクトル (credit assignment vector) を $\boldsymbol{\delta}_t := \dfrac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \in \mathbb{R}^{1\times d}$，即時的貢献度分配ベクトルを $\tilde{\boldsymbol{\delta}}_t := \dfrac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \in \mathbb{R}^{1\times d}$とすると，$\boldsymbol{\delta}_t$ は次のように未来から過去に向かう再帰的な関係式で表せる：

$$
\begin{equation}
\boldsymbol{\delta}_t=\tilde{\boldsymbol{\delta}}_t + \boldsymbol{\delta}_{t+1}\mathbf{J}_{t+1}
\end{equation}
$$

ただし，境界条件として $\boldsymbol{\delta}_{T+1}=\mathbf{0}$ とする。この式を用いて，$\boldsymbol{\delta}_t$ を逐次的に求め，$\frac{\partial \mathbf{h}_t}{\partial \theta_t}$ を即時的に計算して $\boldsymbol{\delta}_t$ に乗じれば，$\frac{\partial \mathcal{L}}{\partial \theta_t}$ が求まる．

連鎖律に基づき理論的な勾配導出を行ったが、数値計算においては、誤差逆伝播法における処理と同様に、自動微分（automatic differentiation, AD）の枠組みを適用して勾配を効率的に計算することができる。自動微分では、微分の累積過程に応じて2つのモードがあり，それぞれが勾配和の2つの方向と対応する。すなわち、過去方向に対する勾配和の計算は順方向累積（forward accumulation）または順モード（forward-mode）自動微分によって、未来方向に対する勾配和の計算は逆方向累積（reverse accumulation）または逆モード（reverse-mode）自動微分によって、それぞれ実行される．

本節の最後に、パラメータに対する損失勾配の展開を整理する。すなわち、過去方向（RTRLに相当）および未来方向（BPTTに相当）での勾配和は、それぞれ次のように表される。

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta} =
\begin{dcases}
\sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \left( \frac{\partial \mathbf{h}_t}{\partial \theta_t} + \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \frac{\partial \mathbf{h}_{t-1}}{\partial \theta} \right) &(\text{過去方向勾配和; e.g., RTRL})\\
\sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial \theta_t} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \theta_t} = \sum_{t=1}^T \left( \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) \frac{\partial \mathbf{h}_t}{\partial \theta_t} &(\text{未来方向勾配和; e.g., BPTT})
\end{dcases}
\end{align}
$$

次節および次々節では、ここで導出した関係式に基づいて、各パラメータの勾配を具体的に計算する。

## 実時間再帰学習 (RTRL)
まず，過去方向勾配和を利用する，**実時間再帰学習** (real-time recurrent learning; RTRL) \citep{williams1989learning} を用いて，各パラメータの勾配を計算する．

まず，前節と同様に，感度行列を $\mathbf{P}_t^{(\theta)}:=\dfrac{\partial \mathbf{h}_t}{\partial \theta}\in \mathbb{R}^{d \times |\theta|}$，即時的感度行列を $\tilde{\mathbf{P}}_t^{(\theta)}:=\dfrac{\partial \mathbf{h}_t}{\partial \theta_t}\in \mathbb{R}^{d \times |\theta|}$ とする．出力に関わるパラメータ $\mathbf{W}_{\mathrm{out}}$ および $\mathbf{b}_y$ は状態 $\mathbf{h}_t$ に影響しないため，$\mathbf{P}_t^{(\theta)}=\tilde{\mathbf{P}}_t^{(\theta)}=\mathbf{0}\ (\theta\in\{\mathbf{W}_{\mathrm{out}}, \mathbf{b}_y\})$ である．よって（即時的）感度行列は  $\theta \in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{b}_h\}$ において考える．即時的感度行列を具体的に書き下すと，次のようになる：

$$
\begin{equation}
\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{in}})} = \alpha\cdot \mathrm{diag}(f'(\mathbf{u}_t))\otimes \mathbf{x}_t, \quad 
\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{rec}})} = \alpha\cdot \mathrm{diag}(f'(\mathbf{u}_t))\otimes \mathbf{h}_{t-1}, \quad
\tilde{\mathbf{P}}_t^{(\mathbf{b}_h)} = \alpha\cdot \mathrm{diag}(f'(\mathbf{u}_t))
\end{equation}
$$

となる．ここで各パラメータの入力感度 $\mathbf{Q}_t^{(\theta)}$ を

$$
\begin{equation}
\mathbf{Q}_t^{(\mathbf{W}_{\mathrm{in}})}:=\mathbf{x}_t^\top,\quad
\mathbf{Q}_t^{(\mathbf{W}_{\mathrm{rec}})}:=\mathbf{h}_{t-1}^\top,\quad
\mathbf{Q}_t^{(\mathbf{b}_h)}:=\mathbf{I}
\end{equation}
$$  

とおくと，$\tilde{\mathbf{P}}_t^{(\theta)}=\alpha\cdot \mathrm{diag}(f'(\mathbf{u}_t))\mathbf{Q}_t^{(\theta)}$ と表せる．次に，状態遷移のヤコビ行列は

$$
\begin{equation}
\mathbf{J}_t := \dfrac{\partial \mathbf{h}_{t}}{\partial \mathbf{h}_{t-1}}=(1-\alpha)\cdot \mathbf{I} + \alpha\cdot \mathrm{diag}(f'(\mathbf{u}_t))\mathbf{W}_{\mathrm{rec}}
\end{equation}
$$ 

であるので，（分子レイアウトで書き直す）
https://en.wikipedia.org/wiki/Matrix_calculus

$$
\begin{align}
\mathbf{P}_t^{(\theta)} &=\tilde{\mathbf{P}}_t^{(\theta)}  + \mathbf{J}_{t}\mathbf{P}_{t-1}^{(\theta)}\\
&=
\end{align}
$$

と求められる．

ただし，境界条件として $\mathbf{P}_{0}=\mathbf{0}$ とする。この式を用いて，$\mathbf{P}_t$ を逐次的に求め，$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}$ を即時的に計算して $\mathbf{P}_t$ に乗じれば，$\frac{\partial \mathcal{L}_t}{\partial \theta}$ が求まる．

を逐次的に保持し，出力誤差と組み合わせて各時刻ごとに重み更新を行う。まず状態感度は以下の再帰式で更新される：  

$$
\begin{equation}
\mathbf{P}_t^{(\theta)}
=\left(1-\alpha\right)\mathbf{P}_{t-1}^{(\theta)}
+\alpha\mathbf{D}_f(\mathbf{a}_t)
\left(\mathbf{W}_{\mathrm{rec}}\mathbf{P}_{t-1}^{(\theta)} + \mathbf{Q}_t^{(\theta)}\right),
\end{equation}
$$  



とし，$\mathbf{P}_t^{(\mathbf{W}_{\mathrm{out}})}=\mathbf{0}$ とする。一方，出力層の誤差は BPTT と同様に  
$\boldsymbol\delta_t^{\mathrm{out}}=\partial\mathcal{L}_t/\partial\mathbf{u}_t$ であるから，時刻 $t$ における各パラメータの勾配は  

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \theta}
=\left(\boldsymbol\delta_t^{\mathrm{out}}\right)^\top
\frac{\partial \mathbf{u}_t}{\partial \theta}
=\left(\boldsymbol\delta_t^{\mathrm{out}}\right)^\top
\mathbf{W}_{\mathrm{out}}\mathbf{P}_t^{(\theta)},
\end{equation}
$$  

ただし $\theta=\mathbf{W}_{\mathrm{out}}$ の場合は  

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{\mathrm{out}}}
=\boldsymbol\delta_t^{\mathrm{out}}\mathbf{h}_t^\top.
\end{equation}
$$

このように RTRL では時刻ごとに $\mathbf{P}_t^{(\theta)}$ を更新し，それを用いて逐次的に勾配を計算するため，オンライン学習が可能となる。

## 経時的誤差逆伝播法 (BPTT)
次に，未来方向勾配和を利用する，**経時的誤差逆伝播法** (backpropagation through time; BPTT) \citep{werbos1988generalization,werbos1990backpropagation} を用いて，各パラメータの勾配を計算する．BPTTはRNNにおける時間方向の処理を空間的に展開してBPを適用するのと同じであるが，どのような処理が行われており，生理学的に妥当性のある処理であるのかを検証するために，ここでは具体的な勾配を計算する．

まず，出力層の誤差信号を

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

## RTRLとBPTTの生理学的実装の困難点
時空間的に局所
時間的に局所 (local)

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

RTRLはパラメータを保持できない．
BPTTは未来から過去へ戻る必要がある．

脳は過去の状態を全て保存して逆向きに再生することは困難である．
再活性化などで可能となっている部分もあるが，全ての状態を保存しておくのは難しい．

海馬においては状態の逆再生 (reverse replay) が行われることが報告されている．

https://pubmed.ncbi.nlm.nih.gov/16474382/
https://www.nature.com/articles/nature04587

https://pmc.ncbi.nlm.nih.gov/articles/PMC6013068/
https://www.science.org/doi/10.1126/science.ads4760
https://www.biorxiv.org/content/10.1101/2023.02.19.529130v4

## 適格度トレースによるRTRLの近似

RFRO (random feedback local online learning) \citep{murray2019local}

$$
\frac{\partial h_{j}(t)}{\partial W_{a b}} = (1 - \alpha) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
$$


$$
\frac{\partial h_{j} (t )}{\partial W_{a b}} = (1 - \alpha ) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
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

RFLOに記載

$$
\Delta \mathbf{W}_{\textrm{out}} = \frac{\eta}{T} \sum_{t=1}^T \epsilon(t) h(t)^\top 
$$

$$
\Delta \mathbf{W}_{\textrm{rec}} = \frac{\eta}{T} \sum_{t=1}^T \mathbf{W}_{\textrm{out}}^\top \epsilon(t) \frac{\partial h(t)}{\partial \mathbf{W}_{\textrm{rec}}}
$$

http://frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full

## 合成勾配学習によるBPTTの近似
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

accumulate BP(λ) アルゴリズムは，強化学習における accumulate TD(λ) に着想を得て，RNNの出力誤差に基づく**将来の勾配（合成勾配）** を，BPTT を用いずに逐次的かつオンラインに学習する手法である。本節では，このアルゴリズムの各ステップを時間順に追い，教科書的な流れで逐次的に解説する。

適格度トレース

---

### 準備：モデル構造と定義

- 時刻 $t$ における RNN の隠れ状態を $h_t$，RNN パラメータを $\Psi$，損失を $L_t$ とする。
- 合成勾配 $\hat{G}_t \approx \frac{\partial L_{>t}}{\partial h_t}$ を出力する**synthesiser** $g(h_t; \theta)$ を学習する。
- 目的：$g(h_t; \theta)$ が正しい未来勾配を予測できるように，$\theta$ を更新する。

---

### ステップ 0: 初期化

\[
\Psi \leftarrow \Psi_0,\quad \theta \leftarrow \theta_0,\quad h \leftarrow 0,\quad \partial h \leftarrow 0,\quad e \leftarrow 0
\]

- $h$：RNN の現在の隠れ状態
- $\partial h$：Jacobian（$\partial h_{t+1}/\partial h_t$）
- $e$：synthesiser の**eligibility trace**

---

### ステップ 1: 新しい入力を処理

\[
h' \leftarrow f(x_t, h; \Psi),\quad L \leftarrow \mathcal{L}(h', y_t)
\]

- 入力 $x_t$ を受けて，RNN は次の状態 $h'$ と出力を生成し，損失 $L$ を計算する。

---

### ステップ 2: 勾配のローカル成分を計算

\[
\partial h \leftarrow \frac{\partial h'}{\partial h},\quad \frac{\partial L}{\partial h'}\quad\text{および}\quad \frac{\partial h'}{\partial \Psi}
\]

- この時点で得られるのは現在の状態 $h_t$ における**局所損失** $L_t$ の勾配のみであり，将来損失 $L_{>t}$ の勾配はまだ得られない。

---

### ステップ 3: 時間差誤差（TD-error）を計算

\[
\delta_t := \left(\frac{\partial L}{\partial h'} + \gamma\,g(h'; \theta)\right)^\top \frac{\partial h'}{\partial h} - g(h; \theta)
\]

- 合成勾配によって将来の勾配を推定し，**誤差 $\delta_t$** として TD 誤差に類似した量を構成する。
- これは「現在の予測 $g(h; \theta)$」と「次の状態での推定値を反映したターゲット値」の誤差を表す。

---

### ステップ 4: eligibility trace を更新

\[
e \leftarrow \gamma\lambda\,\partial h\,e + \nabla_\theta g(h; \theta)
\]

- 時間方向に前向きに伝播する形で，$\theta$ に関する**パラメータごとのトレース**を更新する。
- $\lambda$ によって短期記憶と長期記憶の加重が決まる。

---

### ステップ 5: パラメータ更新

**Synthesiser（θ）の更新**：

\[
\theta \leftarrow \theta + \alpha\,\delta_t^\top e
\]

**RNN パラメータ（Ψ）の更新**：

\[
\Psi \leftarrow \Psi + \eta\,\left(\frac{\partial L}{\partial h'} + g(h'; \theta)\right)^\top \frac{\partial h'}{\partial \Psi}
\]

- ここでの合成勾配を含む勾配が $\Psi$ の更新にも使われるため，synthesiser が誤っていると RNN 自身も誤った方向に学習される点に注意が必要である。

---

### ステップ 6: 状態更新

\[
h \leftarrow h'
\]

- 状態を更新して次の時刻へ進む。

---

### 特徴とポイント

- **λ = 0**：一歩先のブートストラップ合成勾配を使う元の手法（Jaderberg et al., 2017）に相当。
- **λ = 1**：将来のすべての損失を反映した完全な勾配（理論的に BPTT と同等）に一致。
- **ただし BPTT 不要**：いかなる時刻にも「過去の状態に遡る必要がなく」、**逐次的かつオンラインで**更新が可能。

---

このように，accumulate BP(λ) は，TD(λ) の構造と学習理論に基づいて，BPTT の近似を計算的に軽量な形で実現する枠組みである。特に生物学的実装の観点からも，後方パスを用いず，forward trace と局所勾配のみで学習を行う点において有望とされている。

ステップ 2 における「ローカル勾配の計算」では，時刻 $t$ における RNN の状態遷移 $h_t \mapsto h_{t+1}$ および出力 $y_{t+1}$ に関して，以下の3つの勾配を計算する必要があります：

1. $\dfrac{\partial h_{t+1}}{\partial h_t}$  
2. $\dfrac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$  
3. $\dfrac{\partial h_{t+1}}{\partial \Psi}$  

以下ではそれぞれを順に展開する。

---

## 1. $\dfrac{\partial h_{t+1}}{\partial h_t}$：RNN状態のヤコビアン

RNN の状態更新は，一般に次のような形式をとる：

\[
h_{t+1} = f(x_t, h_t; \Psi)
\]

ここで，$f$ は例えば以下のような非線形関数であることが多い（tanh RNN の場合）：

\[
h_{t+1} = \tanh(W_{in} x_t + W_{rec} h_t + b)
\]

このとき，$h_t$ による $h_{t+1}$ のヤコビアンは：

\[
\frac{\partial h_{t+1}}{\partial h_t} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot W_{rec}
\quad \text{ただし} \quad a_t := W_{in} x_t + W_{rec} h_t + b
\]

ここで $\operatorname{diag}[v]$ はベクトル $v$ を対角成分に持つ対角行列である。

---

## 2. $\dfrac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$：ローカル損失の勾配

タスクにおける出力が $y_{t+1} = W_{out} h_{t+1}$ で，目標出力が $\hat{y}_{t+1}$，損失が MSE の場合：

\[
\mathcal{L}_{t+1} = \frac{1}{2} \| \hat{y}_{t+1} - y_{t+1} \|^2
= \frac{1}{2} \| \hat{y}_{t+1} - W_{out} h_{t+1} \|^2
\]

このとき，$h_{t+1}$ に関する損失勾配は：

\[
\frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} = -W_{out}^\top (\hat{y}_{t+1} - W_{out} h_{t+1})
\]

---

## 3. $\dfrac{\partial h_{t+1}}{\partial \Psi}$：パラメータに対する勾配

ここでは RNN パラメータ $\Psi = \{W_{in}, W_{rec}, b\}$ に対して偏微分をとる。

それぞれの成分について：

- $\dfrac{\partial h_{t+1}}{\partial W_{in}} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot x_t^\top$
- $\dfrac{\partial h_{t+1}}{\partial W_{rec}} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot h_t^\top$
- $\dfrac{\partial h_{t+1}}{\partial b} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr]$

これらはテンソル形式でまとめて記述されるか，各パラメータに対してベクトル形式で記録される。

---

## まとめ：すべての項の役割

ステップ 2 は合成勾配更新に必要な各種偏微分を局所的に計算するステップであり，すべて forward pass の情報のみに基づいて，かつ $t+1$ 時点までの情報だけで完結する。

したがってこのステップは完全にオンラインかつ BPTT 非依存であり，合成勾配の正確さと伝搬に必要な中間量を準備する要である。

## 摂動を用いた学習則
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439155/full