# 第5章：再帰型ニューラルネットワークと経時的貢献度分配問題
## 再帰型ニューラルネットワーク
再帰型ニューラルネットワーク (recurrent neural network; RNN)
時刻 $t$ における入力を $\mathbf{x}_t \in \mathbb{R}^{n}$，隠れ状態を $\mathbf{h}_t \in \mathbb{R}^{d}$，出力を $\mathbf{y}_t \in \mathbb{R}^{m}$ とすると，隠れ状態と出力は

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


## 経時的貢献度分配問題

**時間的貢献度分配問題 (Temporal Credit Assignment Problem)** は、強化学習やリカレントニューラルネットワーク（RNN）のような動的システムにおいて、時間的に遅延のある報酬に対して、どのタイミングでどの行動がどれだけ貢献したのかを特定する問題です。具体的には、ある行動が取られた後、その結果として報酬が遅れて現れる場合、報酬がどの行動に対してどれだけ寄与したのかを明確に評価する必要があります。このような問題は、時間的遅延のある状況において、個々の行動の貢献度を割り当てることができなければ、エージェントが適切に学習することは難しくなります。

この時間的貢献度分配問題を解決するための手法の一つが、**バックプロパゲーション・スルー・タイム（BPTT: Backpropagation Through Time）** です。BPTTは、リカレントニューラルネットワーク（RNN）において、時間的依存関係を学習するためのアルゴリズムであり、通常のバックプロパゲーションと同様に、ネットワークの誤差を逆方向に伝播させる方法ですが、時間的に連続する情報を考慮して誤差を伝播させます。BPTTのプロセスは、まずネットワークを順方向に実行して各時刻での出力を計算し、その後、出力層から誤差を計算し、逆方向に誤差を伝播させて、全てのパラメータに対する勾配を求めます。これにより、時間的に関連する状態やアクションがどのように出力に影響を与えたかを学習することができます。

RNNは、出力が次の入力に影響を与えるという再帰的な構造を持つため、BPTTを使用することで、過去の入力やアクションが現在の出力にどれだけ影響を与えているかを適切に学習することができます。しかし、BPTTには計算コストが高くなるという欠点や、長期的な依存関係において勾配消失問題や勾配爆発問題が発生することがあるため、これらの問題を解決するために、長短期記憶（LSTM）やゲート付き再帰ユニット（GRU）のような改良型RNNが使用されることが多いです。

時間的貢献度分配問題とBPTTは、特に強化学習において重要です。強化学習では、エージェントが環境とインタラクションを行い、遅延した報酬を得ることがあります。このような環境では、エージェントがどの行動に対して報酬を得たのか、またその行動が全体の目標達成にどれだけ貢献したのかを評価する必要があります。BPTTを使用することで、時間的な依存関係を適切に学習し、遅延した報酬を適切に割り当てることが可能となり、エージェントは長期的な報酬を最大化するために必要な行動を学習することができます。

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

## BPTTとRTRLの比較
A Practical Sparse Approximation for Real Time Recurrent Learning

A Unified Framework of Online Learning Algorithms for
Training Recurrent Neural Networks

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

### 過去向き・未来向きの勾配計算
次に勾配の和を取る方向が過去向き (past facing)・未来向き (future facing) という話である．先ほどの自動微分における前後と過去未来を混同しないように注意してほしい．

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