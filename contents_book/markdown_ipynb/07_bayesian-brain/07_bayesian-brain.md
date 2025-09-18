Bayesian network

NoProp(2025),DeeperForward(2025),DFA(2016),DRTP(2021),DNI/Synthetic Gradients(2017)

ベイズ線形回帰におけるエネルギーベースモデルとしての解釈
多変量正規分布を用いているので，エネルギーベースモデルとしての解釈も可能である．
入力 $\mathbf{x}$ とラベル $y$ があったとして，その場合のエネルギー状態を下げる働きがある．これは当たり前でもあるが，エネルギーベースモデルにおいては一般にデータに適したエネルギー状態が下がることが重要である．


https://arxiv.org/abs/2306.02572

---


### スパース符号化モデル


### 予測符号化モデル

$u$ を $w$ に変更．


Annotated Bibliographyはもう一度確認する．

Pece, AEC (1992) Redundancy reduction of a Gabor representation: A possible computational role for feedback from primary visual cortex to lateral geniculate nucleus. In I Aleksander, & J Taylor, eds., Artificial Neural Networks, 2, 865–868. Amsterdam: Elsevier

Kawato, M, Hayakama, H, & Inui, T (1993) A forward-inverse optics model of reciprocal connections between visual cortical areas. Network: Computation in Neural Systems 4:415–422.


https://arxiv.org/abs/2011.07464
https://arxiv.org/abs/2112.10048

https://arxiv.org/abs/2410.19315
Predictive coding as variational inference

Srinivasan, M. V., Laughlin, S., & Dubs, A. (1982). Predictive coding: a fresh view of
inhibition in the retina. Proceedings of the Royal Society of London. Series B. Biological
Sciences, 216(1205), 427–459.

Dong, D. W., & Atick, J. J. (1995). Temporal decorrelation: a theory of lagged and
nonlagged responses in the lateral geniculate nucleus. Network: Computation in Neural
Systems, 6(2), 159–178.

A forward-inverse optics model of reciprocal connections between visual cortical areas
https://www.tandfonline.com/doi/abs/10.1088/0954-898X_4_4_001

https://pmc.ncbi.nlm.nih.gov/articles/PMC1569488/#bib45
https://pubmed.ncbi.nlm.nih.gov/15937014/


https://arxiv.org/abs/2212.00720

Kalman filter

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal
of Basic Engineering, 82(1), 35–45.

#### 観測世界の階層的予測
階層的予測符号化(hierarchical predictive coding; HPC) は\citep{`Rao1999-zv`により導入された．構築するネットワークは入力層を含め，3層のネットワークとする．LGNへの入力として画像 $\mathbf{x} \in \mathbb{R}^{n_0}$を考える．画像 $\mathbf{x}$ の観測世界における隠れ変数，すなわち潜在変数 (latent variable) を $\mathbf{r} \in \mathbb{R}^{n_1}$ とし，ニューロン群によって発火率で表現されているとする (真の変数と $\mathbf{r}$ は異なるので文字を分けるべきだが簡単のためにこう表す)．このとき，

$$
\begin{equation}
\mathbf{x} = f(\mathbf{U}\mathbf{r}) + \boldsymbol{\epsilon}
\end{equation}
$$

が成立しているとする．ただし，$f(\cdot)$は活性化関数 (activation function)，$\mathbf{U} \in \mathbb{R}^{n_0 \times n_1}$は重み行列である．$\boldsymbol{\epsilon} \in \mathbb{R}^{n_0}$ は $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ からサンプリングされるとする．潜在変数 $\mathbf{r}$はさらに高次 (higher-level)の潜在変数 $\mathbf{r}^h$により，次式で表現される．

$$
\begin{equation}
\mathbf{r} = \mathbf{r}^{td}+\boldsymbol{\epsilon}^{td}=f(\mathbf{U}^h \mathbf{r}^h)+\boldsymbol{\epsilon}^{td}
\end{equation}
$$

ただし，Top-downの予測信号を $\mathbf{r}^{td}:=f(\mathbf{U}^h \mathbf{r}^h)$ とした．また，$\mathbf{r}^{td} \in \mathbb{R}^{n_1}$, $\mathbf{r}^{h} \in \mathbb{R}^{n_2}$, $\mathbf{U}^h \in \mathbb{R}^{n_1 \times n_2}$ である．$\boldsymbol{\epsilon}^{td} \in \mathbb{R}^{n_1}$は$\mathcal{N}(\mathbf{0}$, $\sigma_{td}^2 \mathbf{I}$) からサンプリングされるとする．

話は飛ぶが，Predictive codingのネットワークの特徴は
- 階層的な構造
- 高次による低次の予測 (Feedback or Top-down信号)
- 低次から高次への誤差信号の伝搬 (Feedforward or Bottom-up 信号)

である．ここまでは高次表現による低次表現の予測，というFeedback信号について説明してきたが，この部分はSparse codingでも同じである．それではPredictive codingのもう一つの要となる，低次から高次への予測誤差の伝搬というFeedforward信号はどのように導かれるのだろうか．結論から言えば，これは復元誤差 (reconstruction error)の最小化を行う再帰的ネットワーク (recurrent network)を考慮することで自然に導かれる．

### 損失関数と学習則
#### 事前分布の設定
$\mathbf{r}$の事前分布$p(\mathbf{r})$はCauchy分布を用いる．$p(\mathbf{r})$の負の対数事前分布を$g(\mathbf{r}):=-\ln p(\mathbf{r})$としておく．

$$
\begin{align}
p(\mathbf{r})&=\prod_i p(r_i)=\prod_i \exp\left[-\alpha \ln(1+r_i^2)\right]\\
g(\mathbf{r})&=-\ln p(\mathbf{r})=\alpha \sum_i \ln(1+r_i^2)\\
g'(\mathbf{r})&=\frac{\partial g(\mathbf{r})}{\partial \mathbf{r}}=\left[\frac{2\alpha r_i}{1+r_i^2}\right]_i
\end{align}
$$

次に重み行列$\mathbf{U}$の事前分布 $p(\mathbf{U})$はGaussian分布とする．$p(\mathbf{U})$の負の対数事前分布を$h(\mathbf{U}):=-\ln p(\mathbf{U})$とすると，次のように表される．

$$
\begin{align}
p(\mathbf{U})&=\exp(-\lambda\|\mathbf{U}\|^2_F)\\
h(\mathbf{U})&=-\ln p(\mathbf{U})=\lambda\|\mathbf{U}\|^2_F\\
h'(\mathbf{U})&=\frac{\partial h(\mathbf{U})}{\partial \mathbf{U}}=2\lambda \mathbf{U}
\end{align}
$$

ただし，$\|\cdot \| _ F^2$はフロベニウスノルムを意味する．

#### 損失関数の設定
Sparse codingと同様に考えることにより，損失関数 $E$を次のように定義する．

$$
\begin{align}
E=\underbrace{\frac{1}{\sigma^{2}}\|\mathbf{x}-f(\mathbf{U} \mathbf{r})\|^2+\frac{1}{\sigma_{t d}^{2}}\left\|\mathbf{r}-f(\mathbf{U}^h \mathbf{r}^h)\right\|^2}_{\text{reconstruction error}}+\underbrace{g(\mathbf{r})+g(\mathbf{r}^{h})+h(\mathbf{U})+h(\mathbf{U}^h)}_{\text{sparsity penalty}}
\end{align}
$$

潜在変数 $\mathbf{r}, \mathbf{r}^h$ と 重み行列 $\mathbf{U}, \mathbf{U}^h$ のそれぞれに事前分布を仮定しているため，これらについてのMAP推定を行うことに相当する．

#### 再帰ネットワークの更新則
簡単のために$\mathbf{z}:=\mathbf{U}\mathbf{r}, \mathbf{z}^h:=\mathbf{U}^h\mathbf{r}^h$とする．

$$
\begin{align}
\frac{d \mathbf{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} \mathbf{U}^{\top}\bigg[\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\odot\underbrace{(\mathbf{x}-f(\mathbf{z}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\mathbf{r})\Bigg)\\
\frac{d \mathbf{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(\mathbf{U}^h)^\top\bigg[\frac{\partial f(\mathbf{z}^h)}{\partial \mathbf{z}^h}\odot\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\mathbf{r}^h)\Bigg)
\end{align}
$$

ただし，$k_1$は更新率 (updating rate)である．または，発火率の時定数を$\tau:=1/k_1$として，$k_1$は発火率の時定数$\tau$の逆数であると捉えることもできる．ここで1番目の式において，中間表現 $\mathbf{r}$ のダイナミクスはbottom-up errorとtop-down errorで記述されている．このようにbottom-up errorが $\mathbf{r}$ への入力となることは自然に導出される．なお，top-down errorに関しては高次からの予測 (prediction)の項 $f(\mathbf{x}^h)$とleaky-integratorとしての項 $-\mathbf{r}$に分割することができる．また$\mathbf{U}^\top, (\mathbf{U}^h)^\top$は重み行列の転置となっており，bottom-upとtop-downの投射において対称な重み行列を用いることを意味している．$-g'(\mathbf{r})$は発火率を抑制してスパースにすることを目的とする項だが，無理やり解釈をすると自己再帰的な抑制と言える．

#### 予測符号化による活動と結合の共調整

本節では予測符号化による

##### 予測符号化による訓練
PCには"Standard" Generative PC と "Reverse" Discriminative PCが存在する．
Millidge, B., Seth, A., & Buckley, C. L. (2021). Predictive Coding: a Theoretical and Experimental Review. In arXiv [cs.AI]. arXiv. http://arxiv.org/abs/2107.12979


ここでのPCは"Reverse" Discriminative PC

状態をdecayすることで，generativeにもdiscriminativeにもすることが可能．
A Predictive-Coding Network That Is Both Discriminative and Generative
https://direct.mit.edu/neco/article/32/10/1836/95621/A-Predictive-Coding-Network-That-Is-Both


入出力を固定 (clamp) する．電位固定法のようなものか？predictive codingと文字を合わせる．(Song et al., 2023)

$x_0=s_{in}, x_{L+1}=s_{target}$とする．状態$x_l(t=0)=\mathbf{0} (l=2, \ldots, L)$に初期化する．予測誤差 $\mathbf{\epsilon}_l$ を次式で計算する．

$$
\begin{equation}
\mathbf{\epsilon}_l(t)=\mathbf{z}_l(t)-\mathbf{w}_{l-1}f(\mathbf{z}_{l-1}(t))\quad(l=1, \ldots, L)
\end{equation}
$$

次に状態 $\mathbf{z}_l(t)\ (t=0, \ldots, \mathcal{T}-1)$ を次式で更新する．

$$
\begin{equation}
\mathbf{z}_l(t+1)=\mathbf{z}_l(t)+\gamma (-\mathbf{\epsilon}_l + f'(\mathbf{z}_l(t))) \circ (\mathbf{w}_l^\top \mathbf{\epsilon}_{l+1}(t))
\end{equation}
$$

収束後，重みを次式で更新する．$n$を一つのsampleの番号として，

$$
\begin{equation}
\mathbf{w}_l(n+1)=\mathbf{w}_l(n)+\eta \mathbf{\epsilon}_l(\mathcal{T}) f(\mathbf{z}_l(\mathcal{T}))^\top
\end{equation}
$$

として重みを更新する．
### 順伝播 (forward propagation)
$f(\cdot)$を活性化関数とする．順伝播(feedforward propagation)は以下のようになる．$(\ell=1,\ldots,L)$

$$
\begin{align}
\text{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{u}_\ell=W_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{u}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{z}_{L+1}
\end{align}
$$

#### 予測符号化による訓練
入出力を固定 (clamp) する．電位固定法のようなものか？predictive codingと文字を合わせる．(Rosebvbaum 2022, Song et al., 2023)

Rosenbaum, R. (2022). On the relationship between predictive coding and backpropagation. PloS One, 17(3), e0266102.

固定点解析によりbackpropと同等であることがわかる．
$\mathbf{z}_1=\mathbf{x}_{\textrm{in}}, \mathbf{z}_{L+1}=\mathbf{x}_{\textrm{target}}$とする．状態$\mathbf{z}_\ell(t=0)=\mathbf{0}\ (\ell=2, \ldots, L)$に初期化する．予測誤差 $\boldsymbol{\epsilon}_\ell(t)$ を次式で計算する．

$$
\begin{equation}
\boldsymbol{\epsilon}_{\ell}(t)=\mathbf{z}_{\ell+1}(t)-\mathbf{W}_{\ell}f(\mathbf{z}_{\ell}(t))\quad(\ell=1, \ldots, L-1)
\end{equation}
$$

$$
\boldsymbol{\epsilon}_{L} = \frac{\partial \mathcal{L} (\mathbf{z}_{L+1}, \mathbf{x}_{\textrm{target}})}{\partial \mathbf{z}_{L+1}}
$$

次に状態 $\mathbf{z}_\ell(t)\quad (\ell=2, \ldots, L;\  t=0, \ldots, \mathcal{T}-1)$ を次式で更新する．

$$
\begin{equation}
\mathbf{z}_\ell(t+1)=\mathbf{z}_\ell(t)+\gamma (-\boldsymbol{\epsilon}_{\ell-1} + f'(\mathbf{z}_\ell(t))) \circ (\mathbf{w}_\ell^\top \boldsymbol{\epsilon}_{\ell}(t))
\end{equation}
$$

収束後，重みを次式で更新する．$n$を一つのsampleの番号として，

$$
\begin{equation}
\mathbf{w}_l(n+1)=\mathbf{w}_l(n)+\eta \mathbf{\epsilon}_l(\mathcal{T}) f(\mathbf{z}_l(\mathcal{T}))^\top
\end{equation}
$$

として重みを更新する．


fixed prediction assumptionという (Millidge etal., 2022. Rosebvbaum 2022) 修正もある．

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}}\\
\boldsymbol{\delta}_L&:=\frac{\partial \mathcal{L}}{\partial \mathbf{u}_L}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{u}_L}\\
\boldsymbol{\delta}_\ell&:=\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell}}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell+1}}\frac{\partial \mathbf{u}_{\ell+1}}{\partial \mathbf{z}_{\ell+1}}\right)\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_{\ell}}\\
&={\mathbf{W}_{\ell+1}}^\top \boldsymbol{\delta}_{\ell+1} \odot f_\ell^{\prime}\left(\mathbf{u}_{\ell}\right)\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{W}_\ell}=\boldsymbol{\delta}_\ell \mathbf{z}_\ell^\top\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{b}_\ell}=\boldsymbol{\delta}_\ell
\end{align}
$$

---

## マルコフ連鎖モンテカルロ法

### マルコフ連鎖モンテカルロ法 (MCMC)
前節では解析的に事後分布の計算をした．事後分布を近似的に推論する方法の1つにマルコフ連鎖モンテカルロ法 (Markov chain Monte Carlo methods; MCMC) がある．他の近似推論の手法としてはLaplace近似や変分推論（variational inference）などがある．MCMCは他の手法に比して，事後分布の推論だけでなく，確率分布を神経活動で表現する方法を提供するという利点がある．

データを$X$とし，パラメータを$\theta$とする．

$$
\begin{equation}
p(\theta\mid X)=\frac{p(X\mid \theta)p(\theta)}{\int p(X\mid \theta)p(\theta)d\theta}
\end{equation}
$$

分母の積分計算$\int p(X\mid \theta)p(\theta)d\theta$が求まればよい．


## エネルギーベースモデルとサンプリング

ポテンシャルエネルギー関数$E$を下に凸の曲面，高次元の神経活動$\mathbf{x}$をその曲面を転がる球としよう．エネルギーの最小化に勾配降下を用いるエネルギーベースモデルでは球は斜面の勾配に沿って運動し，最小のエネルギー状態に到達する．Hopfieldモデルは単なる勾配降下であり，単純な勾配降下を用いるために極小解に陥りやすい．このために各ニューロンが確率的に0,1の値を取るBoltzmannマシンが考案された(Ackley, Hinton, & Sejnowski, 1985)．(制限)BoltzmannマシンではGibbsサンプリングを用い，各ユニットの活動を決める．制限Boltzmannマシンの問題点としては隠れ層間における結合を認めないため感覚入力の無い自発発火を仮定できない点にある．よりモデル構築の柔軟性が高い発火率モデルあるいはspikingモデルにおけるRNNにおいて効率的にサンプリングを行うには，ノイズや振動を用いる (Fig. 4)．なお，点推定を行うには収束時に一定の発火率を保ち続ける必要があり，難しいと考えられる．

Fig. 4. 勾配法と勾配法にノイズ，振動を加えた場合の神経活動のダイナミクスの違い．（左上）2つの細胞の活動$x_{1},\ x_{2}$に対するポテンシャルエネルギー．（右上段）ポテンシャルエネルギー局面上の神経活動の変化．左から勾配法，Langevinダイナミクス，Hamiltonian (+Langevin)ダイナミクス．（右下段）各ダイナミクスにおける$x_{1},\ x_{2}$の経時的変化．Hamiltonianダイナミクスでは振動（+ノイズ）を用いて効率的にサンプリングしている．

Boltzmanマシンでも使用した～などとする．

## ベイズ線形回帰

### モンテカルロ法

### マルコフ連鎖

### Metropolis-Hastings法

### ランジュバン・モンテカルロ法 (LMC)
拡散過程

$$
\begin{equation}
{\frac{d\theta}{dt}}=\nabla \ln p (\theta)+{\sqrt 2}{d{W}}
\end{equation}
$$

Euler–Maruyama法により，

### ハミルトニアン・モンテカルロ法 (HMC法)


LMCよりも一般的なMCMCの手法としてHamiltonianモンテカルロ法(Hamiltonian Monte Calro; HMC)あるいはハイブリッド・モンテカルロ法(Hybrid Monte Calro)がある．エネルギーポテンシャルの局面上をHamilton力学に従ってパラメータを運動させることにより高速にサンプリングする手法である．

一般化座標を$\mathbf{q}$, 一般化運動量を$\mathbf{p}$とする．ポテンシャルエネルギーを$U(\mathbf{q})$としたとき，古典力学（解析力学）において保存力のみが作用する場合のハミルトニアン (Hamiltonian) $\mathcal{H}(\mathbf{q}, \mathbf{p})$は

$$
\begin{equation}
\mathcal{H}(\mathbf{q}, \mathbf{p}):=U(\mathbf{q})+\frac{1}{2}\|\mathbf{p}\|^2
\end{equation}
$$

となる．このとき，次の2つの方程式が成り立つ．

$$
\begin{equation}
\frac{d\mathbf{q}}{dt}=\frac{\partial \mathcal{H}}{\partial \mathbf{p}}=\mathbf{p},\quad\frac{d\mathbf{p}}{dt}=-\frac{\partial \mathcal{H}}{\partial \mathbf{q}}=-\frac{\partial U}{\partial \mathbf{q}}
\end{equation}
$$

これをハミルトンの運動方程式(hamilton's equations of motion) あるいは正準方程式 (canonical equations) という．

リープフロッグ(leap frog)法により離散化する．

1. 共役事前分布を用いた解析的（閉形式）解  
   - ノイズがガウス，かつ回帰係数に対して共役なガウス事前分布を仮定すると，事後分布もガウスとなり，平均・分散を閉形式で得られる．  
   - 具体的には，  
     \[
       p(\boldsymbol\beta\mid X,y)=\mathcal{N}\left(\Sigma_n(X^TX)\beta_0 + \Sigma_n X^Ty,\;\Sigma_n\right),\quad
       \Sigma_n=(X^TX+\Sigma_0^{-1})^{-1},
     \]  
     のように書ける（PRML より）  ([Bayesian linear regression - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_linear_regression?utm_source=chatgpt.com))．  

2. ラプラス近似（Laplace’s method）  
   - 事後分布を最尤解（MAP）まわりの２次多項展開でガウス近似する手法．高次モーメントは捨象されるが，簡便かつ高速に適用可能．  
   - LaplacesDemon などのソフトウェアでも標準的に実装されている  ([LaplacesDemon - Wikipedia](https://en.wikipedia.org/wiki/LaplacesDemon))．  

3. 変分ベイズ（Variational Inference; VI）  
   - 事後分布をパラメトリックな簡易分布族 \(q(\theta;\phi)\) で近似し，KLダイバージェンスを最小化する最適化問題として解く．  
   - 平均場近似，α-divergence 最小化，Amortized VB など多様な拡張がある  ([[PDF] Bayesian inference for latent variable models](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-724.pdf?utm_source=chatgpt.com))．  

4. 期待値伝播（Expectation Propagation; EP）  
   - 近似ファクタを逐次更新し，各因子が除かれた「残差分布」を moment-matching によりガウスで再近似する手法．VI より精度良く，ラプラス近似より堅牢とされる  ([[PDF] Bayesian inference for latent variable models](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-724.pdf?utm_source=chatgpt.com))．  

5. マルコフ連鎖モンテカルロ（MCMC）  
   - 事後分布をターゲットとするマルコフ連鎖を構築しサンプルを得る手法．  
   - 代表的アルゴリズムに Gibbs sampling，Metropolis–Hastings，Hamiltonian Monte Carlo（HMC／NUTS）などがある  ([LaplacesDemon - Wikipedia](https://en.wikipedia.org/wiki/LaplacesDemon))．  

### ベイズ脳仮説
ベイズ脳仮説はより広い枠組みである．

変分ベイズ推論は

https://arxiv.org/abs/1901.07945

Knill, David C., and Alexandre Pouget. 2004. “The Bayesian Brain: The Role of Uncertainty in Neural Coding and Computation.” Trends in Neurosciences 27 (12): 712–19.

### 神経回路における不確実性の表現
ここまでは最尤推定やMAP推定などにより，パラメータ(神経活動，シナプス結合)の点推定を行ってきた．不確実性(uncertainty) を神経回路で表現する方法として主に2つの符号化方法，サンプリングに基づく符号化(sampling-based coding; SBC or neural sampling model) および確率的集団符号化(probabilistic population coding; PPC) が提案されている．SBCは神経活動が元の確率分布のサンプルを表現しており，時間的に多数の活動を集めることで元の分布の情報が得られるというモデルである．PPCは神経細胞集団により，確率分布を表現するというモデルである．

- (Walker et al., 2022)がまとめ．
- (Fiser et al., 2010)の比較表を入れる．
- 神経活動の変動性 (neural variability)
- 自発活動が事前分布であるという説 {cite:p}`Fiser2010-kw`, {cite:p}`Berkes2011-it`.
- {cite:p}`Hoyer2002-ci`
- {cite:p}`Sanborn2016-en`


　神経細胞あるいは細胞集団が確率分布を表現するにはどうすればよいだろうか．神経細胞の活動がある変数を表現していると仮定しよう．単一の細胞の瞬時的な活動がある変数の点推定に対応していると考えれば，単一の細胞の多数の活動あるいは多数の細胞の瞬時的な活動により分布は表現できると考えられる (Fig.2)．

Fig. 2. 神経活動による確率分布表現の2種類の方法．(Fiser, Berkes, Orbán, & Lengyel, 2010)より引用．(a)多数の細胞の瞬時的な活動により分布を表現する符号化 (e.g. probabilistic population codes; PPCs)．(b)単一の細胞の多数の活動により分布を表現する符号化 (e.g. neural sampling codes; NSCs)．Table1は両者の比較．著者らはSampling-based codeの方が優れていると考えている．

多数の細胞の瞬時的な活動により分布を表現する符号化としてはprobabilistic population codes (Ma, Beck, Latham, & Pouget, 2006)やdistributional TD learning (Dabney et al., 2020; Lowet, Zheng, Matias, Drugowitsch, & Uchida, 2020)などが該当する．一方で単一の細胞の多数の活動により分布を表現する符号化はサンプリングに基づいた符号化 (sampling-based coding) あるいは神経サンプリング (neural sampling) と呼ぶ．神経サンプリングの基盤となる現象は神経活動の変動性 (neural variability) である．これは感覚を処理する皮質領野（例えば視覚野）において同じ入力であっても神経細胞の活動が時間や試行に応じて変動する現象のことである (Stein, Gossen, & Jones, 2005)．これが単なるノイズなのか機能があるのかに関しては様々な説が提案されているが，神経活動の変動性によりMCMCが行われているという仮説は(Hoyer & Hyvärinen, 2002)において（自分の知る限り）初めて提案された．(Sanborn & Chater, 2016)は”Bayesian Brains without Probabilities”というキャッチーな題だが，MCMCとBayesian Brainの勉強にはなる．

ここで外界の状態を$x$, それによって生まれた感覚刺激を$y$, 脳内の神経結合を$W$としよう．事前分布 (prior) を$p(x|W)$とし，尤度 (likelihood) を$p(y|x,\ W)$とすると，事後分布 (posterior)は

$$
\begin{equation}
p\left( x \middle| y \right) = \frac{p\left( y \middle| x,\ W \right)p(x|W)}{p(y|W)}
\end{equation}
$$

しかし，ここでの問題は次の2点である．すなわち，

1.  神経回路で確率分布を如何にして表現するか．

2.  規格化定数 $Z = p\left( y \middle| W \right) = \int p\left( y \middle| x,\ W \right)p\left( x \middle| W \right)\ dx$をどう計算するか．

- Neural Sampling Codes
- Probabilistic Population Coding
- Distributed distributional code
RS Zemel, P Dayan, and A Pouget. Probabilistic interpretation of population codes. Neural Computation, 10(2):403–430, 1998. [8] MSahani and P Dayan. Doubly distributional population codes: Simultaneous representation of uncertainty and multiplicity. Neural Computation, 15(10):2255–2279, 2003.


## 神経サンプリング

サンプリングに基づく符号化(sampling-based coding; SBC or neural sampling model)をガウス尺度混合モデルを例にとり実装する．


これまでの大きな流れとして，潜在変数モデルの導入からMAP推定での近似解，そして不確実性を考慮した変分推論とMCMCを扱ってきた．

## 確率的集団符号化
### 確率的集団符号化 (probabilistic population coding)
サンプリング仮説は必要なニューロンの数は少なくてよいが，反面サンプリングに時間を要するという欠点がある．時間と空間のトレードオフになるが，空間を要するが，時間はそれほど要さない手法として確率的集団符号化がある．

Distributional Population Coding or distributed distributional codes (DDCs)

ポアソン分布
https://www.nature.com/articles/nn1790

$$
\begin{equation}
P(X=k)={\frac  {e^{-\lambda} \lambda^k}{k!}}
\end{equation}
$$

より，

$$
\begin{equation}
p(y \mid \mathbf{x}) \propto \prod_{i} \frac{\exp({-f_{i}(y)}) [f_{i}(y)]^{x_{i}}}{x_{i} !} p(y)
\end{equation}
$$

とする．こうすることで事後分布の計算が可能である．

こうした確率的集団符号化はパラメトリックモデルである．神経細胞集団が分布を表現するが，tuning curveを保持しなくてよい枠組みとして，分布型強化学習がある．こちらに関しては強化学習の項目で説明を行う．

## 変分推論
近似分布 $q$ を用意する．近似分布族を $\mathcal{Q}$ とすると，$q \in \mathcal{Q}$ において，最適な分布を探すこととなる．


\citep{Aitchison2016-xu} では

\begin{equation}
\mathcal{H}(\mathbf{u}, \mathbf{v}) = \ln p \left(\mathbf{u}, \mathbf{v} \right) + \textrm{Const.} = \ln p \left(\mathbf{v} \middle| \mathbf{u} \right) + \ln p\left(\mathbf{u} \right) + \textrm{Const.}
\end{equation}
とし，$p\left( \mathbf{v} \middle| \mathbf{u} \right)\mathcal{= N}\left( \mathbf{v};\mathbf{Bu},\ \mathbf{M}^{- 1} \right),\ \ p\left( \mathbf{u} \right) = \mathcal{N}\ (\mathbf{0},\ \mathbf{C}^{- 1})$としている．この場合，
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{v}} = \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{v}} = \ \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}}\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{u}} = - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} = \  - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}}
\end{align}
となる．このままでは等値線上を運動することになるので，Langevinダイナミクスを付け加える．
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u},\mathbf{v} \right)}}{\partial\mathbf{v}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{v}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
となる．それぞれの項は
\begin{align}
\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} &= \mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{Cu}
\end{align}
であるので，
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) + \frac{1}{\tau_{L}}\mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) + \frac{1}{\tau}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
となる．$\mathbf{B} = \mathbf{I}$ とすると，

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \left\lbrack \left( \frac{1}{\tau} - \frac{1}{\tau_{L}} \right)\mathbf{M} - \frac{1}{\tau_{L}}\mathbf{C} \right\rbrack\mathbf{u} - \left( \frac{1}{\tau} - \frac{1}{\tau_{L}} \right)\mathbf{Mv} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) + \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) + \frac{1}{\tau}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \left\lbrack \left( \frac{1}{\tau} + \frac{1}{\tau_{L}} \right)\mathbf{M} + \frac{1}{\tau_{L}}\mathbf{C} \right\rbrack\mathbf{u} - \left( \frac{1}{\tau} + \frac{1}{\tau_{L}} \right)\mathbf{Mv} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
$$


