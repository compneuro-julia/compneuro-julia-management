# 第9章：神経回路網によるベイズ推論
## ベイズ脳仮説と不確実性の表現

### ベイズ脳仮説
Knill, David C., and Alexandre Pouget. 2004. “The Bayesian Brain: The Role of Uncertainty in Neural Coding and Computation.” Trends in Neurosciences 27 (12): 712–19.

### 神経活動による不確実性の表現
ここまでは最尤推定やMAP推定などにより，パラメータ(神経活動，シナプス結合)の点推定を行ってきた．**不確実性(uncertainty)** を神経回路で表現する方法として主に2つの符号化方法，**サンプリングに基づく符号化(sampling-based coding; SBC or neural sampling model)** および**確率的集団符号化(probabilistic population coding; PPC)** が提案されている．SBCは神経活動が元の確率分布のサンプルを表現しており，時間的に多数の活動を集めることで元の分布の情報が得られるというモデルである．PPCは神経細胞集団により，確率分布を表現するというモデルである．

- (Walker et al., 2022)がまとめ．
- (Fiser et al., 2010)の比較表を入れる．
- 神経活動の変動性 (neural variability)
- 自発活動が事前分布であるという説 {cite:p}`Fiser2010-kw`, {cite:p}`Berkes2011-it`.
- {cite:p}`Hoyer2002-ci`
- {cite:p}`Sanborn2016-en`

## ベイズ脳仮説と神経活動による不確実性の表現

が外界の状態を推定する際には**不確実性 (uncertainty)** を考慮する必要がある．例えば外界は3次元なのに対し，網膜像は2次元であり，脳は不良設計問題を解かねばならない．時間の推定においては時間経過を直接的に示す感覚情報はないため，不確実性を常に含む．これらのような不確実性を含んだ推定において脳がベイズ推定を用いているというのが**ベイズ脳仮説 (Bayesian brain hypothesis)** である (Knill & Pouget, 2004)．ここで外界の状態を$x$, それによって生まれた感覚刺激を$y$, 脳内の神経結合を$W$としよう．**事前分布 (prior)** を$p(x|W)$とし，**尤度 (likelihood)** を$p(y|x,\ W)$とすると，**事後分布 (posterior)**は

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

## 神経回路における不確実性の表現

　神経細胞あるいは細胞集団が確率分布を表現するにはどうすればよいだろうか．神経細胞の活動がある変数を表現していると仮定しよう．単一の細胞の瞬時的な活動がある変数の点推定に対応していると考えれば，単一の細胞の多数の活動あるいは多数の細胞の瞬時的な活動により分布は表現できると考えられる (Fig.2)．

**Fig. 2**. 神経活動による確率分布表現の2種類の方法．(Fiser, Berkes, Orbán, & Lengyel, 2010)より引用．(a)多数の細胞の瞬時的な活動により分布を表現する符号化 (e.g. probabilistic population codes; PPCs)．(b)単一の細胞の多数の活動により分布を表現する符号化 (e.g. neural sampling codes; NSCs)．Table1は両者の比較．著者らはSampling-based codeの方が優れていると考えている．

多数の細胞の瞬時的な活動により分布を表現する符号化としては**probabilistic population codes** (Ma, Beck, Latham, & Pouget, 2006)や**distributional TD learning** (Dabney et al., 2020; Lowet, Zheng, Matias, Drugowitsch, & Uchida, 2020)などが該当する．一方で単一の細胞の多数の活動により分布を表現する符号化は**サンプリングに基づいた符号化 (sampling-based coding)** あるいは**神経サンプリング (neural sampling)** と呼ぶ．神経サンプリングの基盤となる現象は**神経活動の変動性 (neural variability)** である．これは感覚を処理する皮質領野（例えば視覚野）において同じ入力であっても神経細胞の活動が時間や試行に応じて変動する現象のことである (Stein, Gossen, & Jones, 2005)．これが単なるノイズなのか機能があるのかに関しては様々な説が提案されているが，神経活動の変動性によりMCMCが行われているという仮説は(Hoyer & Hyvärinen, 2002)において（自分の知る限り）初めて提案された．(Sanborn & Chater, 2016)は”Bayesian Brains without Probabilities”というキャッチーな題だが，MCMCとBayesian Brainの勉強にはなる．

## エネルギーベースモデル

エネルギーベースモデルではネットワークの状態をスカラー値に変換するエネルギー関数 (あるいはコスト関数) を定義し，推論時と学習時の双方においてエネルギーを最小化するようにネットワークの状態を更新する (LeCun, Chopra, Hadsell, Ranzato, & Huang, 2006)．エネルギーベースモデルとしてはIsingモデルや(Amari-)Hopfieldモデル，Boltzmannマシン等が該当する．モデルの神経活動を$\mathbf{x} \in \mathbb{R}^{n}$，パラメータ$\theta$, （ポテンシャル）エネルギー関数 $E_{\theta}:\ \mathbb{R}^{n}\mathbb{\rightarrow R}$とすると，$\mathbf{x}$の分布はGibbs-Boltzmann分布を用いて次のように表せる．

$$
\begin{equation}
p_{\theta}(\mathbf{x})\  = \frac{\exp\left( - {\beta E}_{\theta}\left( \mathbf{x} \right) \right)}{Z_{\theta}}
\end{equation}
$$

ただし，$Z_{\theta}$は規格化定数であり，$Z_{\theta} = \ \int_{}^{}{- \beta E_{\theta}\left( \mathbf{x} \right)d\mathbf{x}}$ である．定義した任意の $E_{\theta}(\mathbf{x})$ を神経活動$\mathbf{x}$やパラメータ$\theta$で微分することで，推論と学習ダイナミクスを定義できる (Fig. 3)．逆に神経活動のダイナミクスを積分することでエネルギーを定義することもできる (Isomura & Friston, 2020)．

Fig. 3. (上) エネルギー，神経活動の確率分布，推論・学習ダイナミクスの関係．簡単のため$\beta = 1$とした．いずれかを定義すれば他が導出できる．確率分布は直接保持されず，神経活動のダイナミクスによるサンプリングで表現される．（下）神経活動のダイナミクスからエネルギーと学習ダイナミクスを導出する例．

## エネルギーベースモデルとサンプリング

ポテンシャルエネルギー関数$E$を下に凸の曲面，高次元の神経活動$\mathbf{x}$をその曲面を転がる球としよう．エネルギーの最小化に勾配降下を用いるエネルギーベースモデルでは球は斜面の勾配に沿って運動し，最小のエネルギー状態に到達する．Hopfieldモデルは単なる勾配降下であり，単純な勾配降下を用いるために極小解に陥りやすい．このために各ニューロンが確率的に0,1の値を取るBoltzmannマシンが考案された(Ackley, Hinton, & Sejnowski, 1985)．(制限)BoltzmannマシンではGibbsサンプリングを用い，各ユニットの活動を決める．制限Boltzmannマシンの問題点としては隠れ層間における結合を認めないため感覚入力の無い自発発火を仮定できない点にある．よりモデル構築の柔軟性が高い発火率モデルあるいはspikingモデルにおけるRNNにおいて効率的にサンプリングを行うには，ノイズや振動を用いる (Fig. 4)．なお，点推定を行うには収束時に一定の発火率を保ち続ける必要があり，難しいと考えられる．

Fig. 4. 勾配法と勾配法にノイズ，振動を加えた場合の神経活動のダイナミクスの違い．（左上）2つの細胞の活動$x_{1},\ x_{2}$に対するポテンシャルエネルギー．（右上段）ポテンシャルエネルギー局面上の神経活動の変化．左から勾配法，Langevinダイナミクス，Hamiltonian (+Langevin)ダイナミクス．（右下段）各ダイナミクスにおける$x_{1},\ x_{2}$の経時的変化．Hamiltonianダイナミクスでは振動（+ノイズ）を用いて効率的にサンプリングしている．

## ベイズ線形回帰
ベイズ線形回帰 (Bayesian linear regression)
共役事前分布 (conjugate prior) を

$$
\begin{equation}
p(\mathbf{w})=\mathcal{N}(\mathbf{w}|\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
\end{equation}
$$

と定義し，事後分布 (posterior) を

$$
\begin{equation}
p(\mathbf{w}|\mathbf{Y}, \mathbf{X})=\mathcal{N}(\mathbf{w}|\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\Sigma}})
\end{equation}
$$

とする．ただし，

$$
\begin{align}
\hat{\boldsymbol{\Sigma}}^{-1}&= \boldsymbol{\Sigma}_0^{-1}+ \beta \Phi^\top\Phi\\
\hat{\boldsymbol{\mu}}&=\hat{\boldsymbol{\Sigma}} (\beta \Phi^\top \mathbf{y}+\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0)
\end{align}
$$

である．また，$\Phi=\phi.(\mathbf{x})$であり，$\phi(x)=[1, x, x^2, x^3]$, $\boldsymbol{\mu}_0=\mathbf{0}, \boldsymbol{\Sigma}_0= \alpha^{-1} \mathbf{I}$とする．テストデータを$\mathbf{x}^*$とした際，予測分布は

$$
\begin{equation}
p(y^*|\mathbf{x}^*, \mathbf{Y}, \mathbf{X})=\mathcal{N}(y^*|\boldsymbol{\mu}^*, \boldsymbol{\Sigma}^*)
\end{equation}
$$

となる．ただし，

$$
\begin{align}
\boldsymbol{\mu}^*&=\hat{\boldsymbol{\mu}}^\top \phi(\mathbf{x}^*)\\
\boldsymbol{\Sigma}^* &= \frac{1}{\beta} +  \phi(\mathbf{x}^*)^\top\hat{\boldsymbol{\Sigma}}\phi(\mathbf{x}^*)\\
\end{align}
$$

である．

## マルコフ連鎖モンテカルロ法

### マルコフ連鎖モンテカルロ法 (MCMC)
前節では解析的に事後分布の計算をした．事後分布を近似的に推論する方法の1つに**マルコフ連鎖モンテカルロ法 (Markov chain Monte Carlo methods; MCMC)** がある．他の近似推論の手法としてはLaplace近似や変分推論（variational inference）などがある．MCMCは他の手法に比して，事後分布の推論だけでなく，確率分布を神経活動で表現する方法を提供するという利点がある．

データを$X$とし，パラメータを$\theta$とする．

$$
\begin{equation}
p(\theta\mid X)=\frac{p(X\mid \theta)p(\theta)}{\int p(X\mid \theta)p(\theta)d\theta}
\end{equation}
$$

分母の積分計算$\int p(X\mid \theta)p(\theta)d\theta$が求まればよい．

### モンテカルロ法

### マルコフ連鎖

### Metropolis-Hastings法

### ランジュバン・モンテカルロ法 (LMC)
拡散過程

$$
\begin{equation}
{\frac{d\theta}{dt}}=\nabla \log p (\theta)+{\sqrt 2}{d{W}}
\end{equation}
$$

Euler–Maruyama法により，

### ハミルトニアン・モンテカルロ法 (HMC法)

LMCよりも一般的なMCMCの手法としてHamiltonianモンテカルロ法(Hamiltonian Monte Calro; HMC)あるいはハイブリッド・モンテカルロ法(Hybrid Monte Calro)がある．エネルギーポテンシャルの局面上をHamilton力学に従ってパラメータを運動させることにより高速にサンプリングする手法である．

一般化座標を$\mathbf{q}$, 一般化運動量を$\mathbf{p}$とする．ポテンシャルエネルギーを$U(\mathbf{q})$としたとき，古典力学（解析力学）において保存力のみが作用する場合の**ハミルトニアン (Hamiltonian)** $\mathcal{H}(\mathbf{q}, \mathbf{p})$は

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

これを**ハミルトンの運動方程式(hamilton's equations of motion)** あるいは**正準方程式 (canonical equations)** という．

リープフロッグ(leap frog)法により離散化する．

## 神経サンプリング

サンプリングに基づく符号化(sampling-based coding; SBC or neural sampling model)をガウス尺度混合モデルを例にとり実装する．

## ガウス尺度混合モデル
**ガウス尺度混合 (Gaussian scale mixture; GSM) モデル**は確率的生成モデルの一種である{cite:p}`Wainwright1999-cl`{cite:p}`Orban2016-tm`．GSMモデルでは入力を次式で予測する：

$$
\begin{equation}
\text{入力}={z}\left(\sum \text{神経活動} \times \text{基底} \right) + \text{ノイズ}
\end{equation}
$$

前節までのスパース符号化モデル等と同様に，入力が基底の線形和で表されるとしている．ただし，尺度(scale)パラメータ$z$が基底の線形和に乗じられている点が異なる．\footnote{コードは{cite:p}`Orban2016-tm` <https://github.com/gergoorban/sampling_in_gsm>, および{cite:p}`Echeveste2020-sh` <https://bitbucket.org/RSE_1987/ssn_inference_numerical_experiments/src/master/>を参考に作成した．}


### 事前分布
$\mathbf{x} \in \mathbb{R}^{N_x}$, $\mathbf{A} \in \mathbb{R}^{N_x\times N_y}$, $\mathbf{y} \in \mathbb{R}^{N_y}$, $\mathbf{z} \in \mathbb{R}$とする．

$$
\begin{equation}
p\left(\mathbf{x}\mid\mathbf{y}, z\right)=\mathcal{N}\left(z \mathbf{A} \mathbf{y}, \sigma_{\mathbf{x}}^{2} \mathbf{I}\right)
\end{equation}
$$

事前分布を

$$
\begin{align}
p\left(\mathbf{y}\right)&=\mathcal{N}\left(\mathbf{0}, \mathbf{C}\right)\\
p\left(z\right)&=\Gamma (k, \vartheta)
\end{align}
$$

とする．$\Gamma(k, \vartheta)$はガンマ分布であり，$k$は形状(shape)パラメータ，$\vartheta$は尺度(scale)パラメータである．$p\left(\mathbf{y}\right)$は$\mathbf{y}$の事前分布であり，刺激がない場合の自発活動の分布を表していると仮定する．

### 分散共分散行列$\mathbf{C}$の作成
$\mathbf{C}$は$y$の事前分布の分散共分散行列である．{cite:p}`Orban2016-tm`では自然画像を用いて作成しているが，ここでは簡単のため$\mathbf{A}$と同様に{cite:p}`Echeveste2020-sh`に従って作成する．前項で作成した通り，$\mathbf{A}$の各基底には周期性があるため，類似した基底を持つニューロン同士は類似した出力をすると考えられる．Echevesteらは$\theta\in[-\pi/2, \pi/2)$の範囲においてFourier基底を複数作成し，そのグラム行列(Gram matrix)を係数倍したものを$\mathbf{C}$と設定している．ここではガウス過程(Gaussian process)モデルとの類似性から，周期カーネル(periodic kernel) 

$$
\begin{equation}
K(\theta, \theta')=\exp\left[\phi_1 \cos \left(\dfrac{|\theta-\theta'|}{\phi_2}\right)\right]
\end{equation}
$$

を用いる．ここでは$|\theta-\theta'|=m\pi\ (m=0,1,\ldots)$の際に類似度が最大になればよいので，$\phi_2=0.5$とする．これが正定値行列になるように単位行列の係数倍$\epsilon\mathbf{I}$を加算し，スケーリングした上で，`Symmetric(C)`や`Matrix(Hermitian(C)))`により実対象行列としたものを$\mathbf{C}$とする．$\mathbf{C}$を正定値行列にする理由はJuliaの`MvNormal`がCholesky分解を用いて多変量正規分布の乱数を生成するためである． 事前に`cholesky(C)`が実行できるか確認するのもよい．

### 事後分布の計算
事後分布は$z$と$\mathbf{y}$のそれぞれについて次のように求められる．


$$
\begin{align}
p(z \mid \mathbf{x}) &\propto p(z) \mathcal{N}\left(0, z^{2} \mathbf{A C A}^{\top}+\sigma_{x}^{2} \mathbf{I}\right)\\
p(\mathbf{y} \mid z, \mathbf{x})& = \mathcal{N}\left(\mu(z, \mathbf{x}), \Sigma(z)\right)
\end{align}
$$

ただし，

$$
\begin{align}
\Sigma(z)&=\left(\mathbf{C}^{-1}+\frac{z^{2}}{\sigma_{x}^{2}} \mathbf{A}^{\top} \mathbf{A}\right)^{-1}\\
\mu(z, \mathbf{x})&=\frac{z}{\sigma_{x}^{2}} \Sigma(z) \mathbf{A}^{\top} \mathbf{x}
\end{align}
$$

である．

最終的な予測において$z$の事後分布は必要でないため，$p(\mathbf{y} \mid z, \mathbf{x})$から$z$を消去することを考えよう．厳密に行う場合，次式のように周辺化(marginalization)により，$z$を（積分）消去する必要がある．

$$
\begin{equation}
p(\mathbf{y} \mid \mathbf{x}) = \int dz\ p(z\mid \mathbf{x})\cdot p(\mathbf{y} \mid z, \mathbf{x})
\end{equation}
$$

周辺化においては，まず$z$のMAP推定（最大事後確率推定）値 $z_{\mathrm{MAP}}$を求める．

$$
\begin{equation}
z_{\mathrm{MAP}} = \underset{z}{\operatorname{argmax}} p(z\mid \mathbf{x})
\end{equation}
$$

次に$z_{\mathrm{MAP}}$の周辺で$p(z\mid \mathbf{x})$を積分し，積分値が一定の閾値を超える$z$の範囲を求め，この範囲で$z$を積分消去してやればよい．しかし，$z$は単一のスカラー値であり，この手法で推定するのは煩雑であるために近似手法が{cite:p}`Echeveste2017-wu`において提案されている．Echevesteらは第一の近似として，$z$の分布を$z_{\mathrm{MAP}}$でのデルタ関数に置き換える，すなわち，$p(z\mid \mathbf{x})\simeq \delta (z-z_{\mathrm{MAP}})$とすることを提案している．この場合，$z$は定数とみなせ，$p(\mathbf{y} \mid \mathbf{x})\simeq p(\mathbf{y} \mid \mathbf{x}, z=z_{\mathrm{MAP}})$となる．第二の近似として，$z_{\mathrm{MAP}}$を真のコントラスト$z^*$で置き換えることが提案されている．GSMへの入力$\mathbf{x}$は元の画像を$\mathbf{\tilde x}$とすると，$\mathbf{x}=z^* \mathbf{\tilde x}$としてスケーリングされる．この入力の前処理の際に用いる$z^*$を用いてしまおうということである．この場合，$p(\mathbf{y} \mid \mathbf{x})\simeq p(\mathbf{y} \mid \mathbf{x}, z=z^*)$となる．しかし，入力を任意の画像とする場合，$z^*$は未知である．簡便さと精度のバランスを取り，ここでは第一の近似，$z=z_{\mathrm{MAP}}$とする手法を用いることにする．

## 興奮性・抑制性神経回路によるサンプリング
前節で実装したMCMCを**興奮性・抑制性神経回路 (excitatory-inhibitory (E-I) network)** で実装する．HMCとLMCの両方を神経回路で実装する．ハミルトニアンを用いる場合，一般化座標$\mathbf{q}$を興奮性神経細胞の活動$\mathbf{u}$, 一般化運動量$\mathbf{p}$を抑制性神経細胞の活動$\mathbf{v}$に対応させる．$\mathbf{u,\ v}$は同じ次元のベクトルとする．$\mathbf{u}, \mathbf{v}$の時間発展はハミルトニアン$\mathcal{H}$を導入して

$$
\begin{equation}
\tau\frac{d\mathbf{u}}{dt} = \frac{\partial \mathcal{H}}{\partial\mathbf{v}},\quad\tau\frac{d\mathbf{v}}{dt} = - \frac{\partial \mathcal{H}}{\partial\mathbf{u}}
\end{equation}
$$

と書ける．一般的には$\mathcal{H}(\mathbf{u}, \mathbf{v}) = E\left( \mathbf{u} \right) + \frac{1}{2}\mathbf{v}^{\top}\mathbf{v}$であり，$p\left( \mathbf{u},\ \mathbf{v} \right) \propto \exp( - \mathcal{H}(\mathbf{u,v}))$である．力学的エネルギーを保つ運動は，対数同時分布における等値線上の運動と同じである．

\citep{Aitchison2016-xu}では

$$
\begin{equation}
\mathcal{H}(\mathbf{u}, \mathbf{v}) = \log p \left(\mathbf{u}, \mathbf{v} \right) + \textrm{Const.} = \log p \left(\mathbf{v} \middle| \mathbf{u} \right) + \log p\left(\mathbf{u} \right) + \textrm{Const.}
\end{equation}
$$

とし，$p\left( \mathbf{v} \middle| \mathbf{u} \right)\mathcal{= N}\left( \mathbf{v};\mathbf{Bu},\ \mathbf{M}^{- 1} \right),\ \ p\left( \mathbf{u} \right) = \mathcal{N\ (}\mathbf{0},\ \mathbf{C}^{- 1})$としている．この場合，

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{v}} = \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{v}} = \ \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}}\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{u}} = - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} = \  - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}}
\end{align}
$$
となる．このままでは等値線上を運動することになるので，Langevinダイナミクスを付け加える．

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\log{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\log{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\log{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\log{p\left( \mathbf{u},\mathbf{v} \right)}}{\partial\mathbf{v}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\log{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{v}} - \frac{1}{\tau}\frac{\partial\log{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
$$

となる．それぞれの項は

$$
\begin{align}
\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} &= \mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\log{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\log{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{Cu}
\end{align}
$$

であるので，

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) + \frac{1}{\tau_{L}}\mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right) + \frac{1}{\tau}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
$$

となる．$\mathbf{B = I}$ とすると，

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) - \frac{1}{\tau_{L}}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \left\lbrack \left( \frac{1}{\tau} - \frac{1}{\tau_{L}} \right)\mathbf{M} - \frac{1}{\tau_{L}}\mathbf{C} \right\rbrack\mathbf{u} - \left( \frac{1}{\tau} - \frac{1}{\tau_{L}} \right)\mathbf{Mv} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= \frac{1}{\tau}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) + \frac{1}{\tau_{L}}\mathbf{M}\left( \mathbf{u} - \mathbf{v} \right) + \frac{1}{\tau}\mathbf{Cu} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \left\lbrack \left( \frac{1}{\tau} + \frac{1}{\tau_{L}} \right)\mathbf{M} + \frac{1}{\tau_{L}}\mathbf{C} \right\rbrack\mathbf{u} - \left( \frac{1}{\tau} + \frac{1}{\tau_{L}} \right)\mathbf{Mv} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
$$

となり，$\mathbf{u}\mathbf{,\ v}$と定行列およびノイズに依存してサンプリングダイナミクスを記述できる．長々と式変形を書いたが，重要なのは**興奮性・抑制性という2種類の細胞群の相互作用により生み出された振動を用いてサンプリングにおける自己相関を下げることができる**という点である．

簡単のため，前項で用いた入力刺激のうち，最も$z$が大きいサンプルのみを使用する．

Hamiltonianネットワークは自己相関を振動により低下させることで，効率の良いサンプリングを実現している．ToDo: 普通にMCMCやる場合も自己相関は確認したほうがいいという話をどこかに書く．

推定された事後分布を特定の神経細胞のペアについて確認する．

Hamiltonianネットワークの方が安定して事後分布を推定することができている．ToDo: 以下の記述．ここでは重みを設定したが， {cite:p}`Echeveste2020-sh`ではRNNにBPTTで重みを学習させている．動的な入力に対するサンプリング {cite:p}`Berkes2011-xj`．burn-inがなくなり効率良くサンプリングできる．

## Spikingニューラルネットワークにおけるサンプリング
前項で挙げた例は発火率モデルであったが，SNNにおいてサンプリングを実行する機構自体は考案されている．ToDo: 以下の記述．{cite:p}`Buesing2011-dm`{cite:p}`Masset2022-wh`{cite:p}`Zhang2022-bl`

## シナプスサンプリング
ここまでシナプス結合強度は変化せず，神経活動の変動によりサンプリングを行うというモデルについて考えてきた．一方で，シナプス結合強度自体が短時間で変動することによりベイズ推論を実行するというモデルがあり，**シナプスサンプリング(synaptic sampling)** と呼ばれる．ToDo: 以下の記述．{cite:p}`Kappel2015-kq`{cite:p}`Aitchison2021-wo`

## 確率的集団符号化
### 確率的集団符号化 (probabilistic population coding)

Distributional Population Coding or distributed distributional codes (DDCs)

ポアソン分布

$$
\begin{equation}
P(X=k)={\frac  {e^{-\lambda} \lambda^k}{k!}}
\end{equation}
$$

より，

$$
\begin{equation}
p(y \mid \mathbf{x}) \propto \prod_{i} \frac{e^{-f_{i}(y)} f_{i}(y)^{x_{i}}}{x_{i} !} p(y)
\end{equation}
$$