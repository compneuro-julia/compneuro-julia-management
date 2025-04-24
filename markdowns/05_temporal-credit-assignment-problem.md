# 第5章：再帰型ニューラルネットワークと経時的貢献度分配問題
1. **過去向き vs. 未来向きの概念図（cartoon）**  
   - Marschall et al. の模式図などを用い，矢印の向きで違いを直感的に示す。  
2. **BPTT（未来向き）の詳細導出**  
   - 実際の逆伝播式，数式レベルで \(\varepsilon_t\) の再帰や重み勾配を示す。  
3. **RTRL（過去向き）の詳細導出**  
   - 状態感度行列の再帰式，感度保持の仕組みと勾配計算を解説。  
4. **トレードオフのまとめ**  
   - 計算量・メモリ・オンライン性・生物学的妥当性の比較表で振り返る。  

BPTTとRTRLの比較
A Practical Sparse Approximation for Real Time Recurrent Learning

A Unified Framework of Online Learning Algorithms for
Training Recurrent Neural Networks

## 勾配法に基づく経時的貢献度分配
本章では再帰型ニューラルネットワーク（recurrent neural networks; RNN），すなわち再帰的結合を持った発火率モデルの学習則について取り扱う．前章で扱った前方向結合のみのニューラルネットワークでは通常，入力と出力の間の遅延は考慮せず，即時的に学習することを想定する。しかしRNNや運動制御，強化学習などの動的な系では，ある神経活動や行動の変化が，誤差や報酬という形で観測されるまでに時間的な遅れを伴う場合がある。このような状況において，「ある時点の神経活動または行動の変化が，後になって得られる誤差や報酬にどれだけ寄与したか」を明らかにし，適切に割り当てる問題を経時的貢献度分配問題（temporal credit assignment problem）と呼ぶ。時間のずれによって評価信号が遅れて到来する場合，因果の流れを遡って「いつ，どの変化が，どれほど貢献したか」を正確に見積もることが，学習の要となる。

勾配法に基づいて経時的貢献度分配をする代表的な手法として，経時的誤差逆伝播法  (backpropagation through time; BPTT) と実時間再帰学習 (real-time recurrent learning; RTRL) の2種類がある．本節ではBPTTとRTRLを統合的な視点から説明し，なぜ勾配法から2種類の学習則が生じるのかについて詳説する．実装に落とし込むための詳細については次節 (BPTT) と次々節 (RTRL) で触れる．

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

### 過去向き・未来向きの勾配計算
このRNNにおける目標は損失 $\mathcal{L}$ を最小化するようにパラメータ $\theta \ \left(\in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{W}_{\mathrm{out}},\mathbf{b}\}\right)$ を最適化することである．勾配法の観点では，パラメータの勾配 $\dfrac{\partial \mathcal{L}}{\partial \theta}$ が求まれば最適化が可能である．損失は $\mathcal{L} = \sum_t \mathcal{L}_t$ と時間方向に分解できるので，パラメータの勾配は

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t}\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}\frac{\partial \theta_s}{\partial \theta}=\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}
$$

と計算できる．ここで時刻 $s$ におけるパラメータ $\theta$ を特に $\theta_s$ と表記した．ゆえに，$\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ は時刻 $t$ における損失 $\mathcal{L}_t$ に対する，時刻 $s\ (s\leq t)$ でのパラメータ $\theta_s$ の勾配を意味する．なお，オンライン学習をしない限り，通常は全時刻において同じパラメータを使用する，すなわち，全ての $s$ について $\theta = \theta_s$ である．この場合，$\frac{\partial \theta_s}{\partial \theta}=\mathbf{I}$ が成立し，上式ではこれを用いた．

なぜ，BPTTとRTRLの2種類の学習法があるのかといえば，$\sum_{t}\sum_{s\leq t}\frac{\partial \mathcal{L}_t}{\partial \theta_s}$ の二重和においてどちらの和を先に計算するかが2通りあるためである．勾配の和を取る2通りの方法は，その和の向きが過去向き (past facing) と未来向き (future facing) であるといえる．

パラメータ $\theta$ への勾配は次のような2種類の二重和で書くことができる．
時刻の範囲は$1 \leq s, t \leq T\quad (s\leq t)$ である．

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \theta}=
\begin{dcases}
\sum_{s=1}^T\frac{\partial \mathcal{L}}{\partial \theta_s}=\sum_{s=1}^T\sum_{t=s}^T\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{未来向き; BPTT})\\
\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{t=1}^T\sum_{s=1}^t\frac{\partial \mathcal{L}_t}{\partial \theta_s}\quad(\text{過去向き; RTRL})
\end{dcases}
\end{align}
$$

#### 過去向き（past-facing）勾配  
現在時刻 $t$ における損失 $\mathcal{L}_t$ に対して，過去のすべてのパラメータ $\theta_s\ (s\leq t)$ が与えた影響を合算するのが過去向き勾配である。すなわち  

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \theta}=\sum_{s=1}^t\frac{\partial \mathcal{L}_t}{\partial \theta_s}
\end{equation}
$$

この形は「現在の損失が過去のパラメータ適用にどれだけ依存するか」を評価するため，RTRLのように損失 $\mathcal{L}_t$ に対する状態の感度行列を逐次更新しながら勾配を得る．

#### 未来向き（future-facing）勾配  
現在時刻 $s$ におけるパラメータ $\theta_s$ が未来のすべての損失 $\mathcal{L}_t\ (t \geq s)$ に与える影響を合算するのが未来向き勾配である。すなわち  

$$
\begin{equation}
\nabla_w\mathcal{L}(s)
=\sum_{t=s}^T\frac{\partial\mathcal{L}(t)}{\partial w(s)}
=\frac{\partial\mathcal{L}}{\partial w(s)}\,.
\end{equation}
$$

この形は「現在のパラメータ変更が将来の損失にどれだけ寄与するか」を評価するため，出力誤差を未来方向に逆伝播させる BPTT（Back-Propagation Through Time）が典型例である。式 (3) に対応する。  

先ほどの自動微分における前後と過去未来を混同しないように注意してほしい．

### BPTTとRTRL

BPTTとRTRLの学習則をより深く理解するために比較をする．

$$
\begin{align}
\fbox{\text{BPTT}}\quad \frac{\partial \mathcal{L}}{\partial \theta}&=\sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\sum_{t=1}^T\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}\right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}\\
\fbox{\text{RTRL}}\quad \frac{\partial \mathcal{L}}{\partial \theta}&=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
\end{align}
$$

BPTTとRTRLを以下の2つの観点から見よう．
前向き(RTRL)・後ろ向き自動微分(BPTT) に対応する．
勾配の和を過去向き(RTRL)・未来向き(BPTT)に取る．

### 前向き・後ろ向き自動微分

前向きモード自動微分 (forward-mode differentiation) がRTRLに対応し，後ろ向きモード自動微分がBPTTに対応する．

入力から損失の向きに計算するか，損失から入力の向きに計算するか．

## 経時的誤差逆伝播法 (BPTT)
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