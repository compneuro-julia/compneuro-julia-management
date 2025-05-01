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

1. **共役事前分布を用いた解析的（閉形式）解**  
   - ノイズがガウス，かつ回帰係数に対して共役なガウス事前分布を仮定すると，事後分布もガウスとなり，平均・分散を閉形式で得られる．  
   - 具体的には，  
     \[
       p(\boldsymbol\beta\mid X,y)=\mathcal{N}\bigl(\Sigma_n(X^TX)\beta_0 + \Sigma_n X^Ty,\;\Sigma_n\bigr),\quad
       \Sigma_n=(X^TX+\Sigma_0^{-1})^{-1},
     \]  
     のように書ける（PRML より）  ([Bayesian linear regression - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_linear_regression?utm_source=chatgpt.com))。  

2. **ラプラス近似（Laplace’s method）**  
   - 事後分布を最尤解（MAP）まわりの２次多項展開でガウス近似する手法。高次モーメントは捨象されるが，簡便かつ高速に適用可能。  
   - LaplacesDemon などのソフトウェアでも標準的に実装されている  ([LaplacesDemon - Wikipedia](https://en.wikipedia.org/wiki/LaplacesDemon))。  

3. **変分ベイズ（Variational Inference; VI）**  
   - 事後分布をパラメトリックな簡易分布族 \(q(\theta;\phi)\) で近似し，KLダイバージェンスを最小化する最適化問題として解く。  
   - 平均場近似，α-divergence 最小化，Amortized VB など多様な拡張がある  ([[PDF] Bayesian inference for latent variable models](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-724.pdf?utm_source=chatgpt.com))。  

4. **期待値伝播（Expectation Propagation; EP）**  
   - 近似ファクタを逐次更新し，各因子が除かれた「残差分布」を moment-matching によりガウスで再近似する手法。VI より精度良く，ラプラス近似より堅牢とされる  ([[PDF] Bayesian inference for latent variable models](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-724.pdf?utm_source=chatgpt.com))。  

5. **マルコフ連鎖モンテカルロ（MCMC）**  
   - 事後分布をターゲットとするマルコフ連鎖を構築しサンプルを得る手法。  
   - 代表的アルゴリズムに Gibbs sampling，Metropolis–Hastings，Hamiltonian Monte Carlo（HMC／NUTS）などがある  ([LaplacesDemon - Wikipedia](https://en.wikipedia.org/wiki/LaplacesDemon))。  
