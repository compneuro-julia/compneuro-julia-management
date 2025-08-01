## 摂動を用いた学習則
本節では摂動法 (permutation) による勾配推定について説明する．摂動法に含まれる手法は複数あるが，総じて次のような手法を指す．まず，あるモデル（ネットワーク）を用意し，その目的関数を $\mathcal{L}$ とする．次にモデルのパラメータや活動にランダムな微小変化（摂動）$\mathbf{v}$ を加え，摂動を受ける前後の目的関数の変化量 $\delta \mathcal{L}$ を取得する．この $\delta \mathcal{L}$ や $\mathbf{v}$ およびモデルの活動等を用いてパラメータを更新するのが摂動法である．

こうした摂動法は導関数フリー最適化 (derivative-free optimization; DFO)\footnote{他には，勾配フリー最適化 (greadient-free optimization), 0次最適化 (zeroth-order optimization) やブラックボックス最適化 (blackbox optimization) とも呼ばれる．本書では触れないが，勾配フリー最適化としては焼きなまし法 (Simulated annealing)，進化戦略 (Evolution strategy)，ベイズ最適化 (Bayesian optimization)，遺伝的アルゴリズム (Genetic algorithm, GA)，粒子群最適化 (Particle swarm optimization, PSO) などが該当する．} の一種であり，誤差逆伝播の近似手法のように連鎖率における一部の勾配を近似する手法とは区別される．

### 勾配の近似法
#### 有限差分法

前進差分（forward difference）を使用した有限差分法では


各成分 $i \in \{1, \ldots, n\}$ について、標準基底ベクトル $e_i \in \mathbb{R}^n$ を用いて：

$$
\begin{equation}
\left[ \nabla f(x) \right]_i \approx \frac{f(x + \epsilon e_i) - f(x)}{\epsilon}
\end{equation}
$$

これをすべての次元 $i$ に適用して、勾配ベクトルを構成すれば：

$$
\nabla f(x) \approx \frac{1}{\epsilon}
\begin{bmatrix}
f(x + \epsilon e_1) - f(x) \\
f(x + \epsilon e_2) - f(x) \\
\vdots \\
f(x + \epsilon e_n) - f(x)
\end{bmatrix}
$$

この形式では、$n+1$ 回の関数評価が必要となる．
後の手法に合わせるのであれば，$n+1$ 点法とも呼ぶことができる．

有限差分法 (finite difference method; FDM) との違いは，全ての特徴量に同時に摂動を加える点である．

#### 2点法

$$
\begin{equation}
\nabla f(x) \approx \frac{d}{\sigma}(f(x+\sigma \xi) - f(x))\cdot \xi
\end{equation}
$$

#### 1点法

$$
\begin{equation}
\nabla f(x) \approx \frac{d}{\sigma}f(x+\sigma \xi)\cdot \xi
\end{equation}
$$

Flaxman

2点法 (two-point method) と呼ぶ．
数理最適化の観点からは1点法 (one-point method) も提案がされており，
1点法の利点は，動的に環境が変化する際に1回のみしか関数評価ができない場合でも学習が可能であるという点である．
ただし，2点法よりも推定された勾配の分散が大きく，学習は安定しない．
https://ieeexplore.ieee.org/document/10480627/

前回の関数評価値を利用して，差分を用いる残差1点法も提案されている．

REINFORCEでは1点法が用いられる．
https://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes14.pdf

### ノード摂動法と重み摂動法
2点法を応用して，ニューラルネットワークの重みを訓練する手法を

代表的なニューラルネットワークの摂動法は**ノード摂動法** (Node perturbation; NP) と**重み摂動法** (weight perturbation; WP) である．ノード摂動法は各ノード（ニューロン）の活動に摂動を加える手法であり，重み摂動法は各パラメータ（シナプス結合等）に摂動を加える手法である．両者は統一的に解釈することが可能である．

まず，以下のように順伝播を行う $L$ 層のニューラルネットワークを定義する $(\ell=1,\ldots,L)$. $\mathbf{z}_{\ell}\in \mathbb{R}^{n_\ell}$とすると

$$
\begin{align}
\text{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{u}_\ell=\mathbf{W}_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{u}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{z}_{L+1}
\end{align}
$$

ただし，$\mathbf{W}_\ell \in \mathbb{R}^{n_{\ell+1}\times n_{\ell}}, \mathbf{b}_\ell \in \mathbb{R}^{n_{\ell+1}}$ である．
ここでは単純なMLPを扱うが，RNNでも可能である．損失は $\mathcal{L}(\mathbf{z}_{L+1}; \mathbf{x})$ とする．それぞれの手法において，以下のようにネットワークを摂動する．

$$
\begin{align}
\text{重み摂動法:}\quad &\tilde{\mathbf{z}}_{\ell+1}=f_\ell\left((\mathbf{W}_\ell+\sigma \mathbf{V}_\ell) \tilde{\mathbf{z}}_\ell +\mathbf{b}_\ell +\sigma \mathbf{v}_\ell\right)\\
\text{ノード摂動法:}\quad &\tilde{\mathbf{z}}_{\ell+1}=f_\ell\left(\mathbf{W}_\ell \tilde{\mathbf{z}}_\ell +\mathbf{b}_\ell+\sigma \mathbf{v}_\ell \right)
\end{align}
$$

ただし，$\mathbf{V}_\ell \in \mathbb{R}^{n_{\ell+1}\times n_{\ell}}, \mathbf{v}_\ell \in \mathbb{R}^{n_{\ell+1}}$ であり，各要素は $\mathcal{N}(0, 1)$ より独立にサンプリングされる \footnote{摂動は正規分布以外の分布，例えば $\{-1, 1\}$ (1か-1かを等確率で取る分布) からサンプリングすることも可能である．}．目的関数の変化量を

$$
\begin{equation}
\delta \mathcal{L}=\mathcal{L}(\tilde{\mathbf{z}}_{L+1}; \mathbf{x})-\mathcal{L}(\mathbf{z}_{L+1}; \mathbf{x})
\end{equation}
$$

とする．SGDでパラメータを行う場合，

$$
\begin{align}
\text{重み摂動法:}\quad &\Delta \mathbf{W}_\ell^{\mathrm{WP}}=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{V}_\ell, &\Delta \mathbf{b}_\ell^{\mathrm{WP}}=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell\\
\text{ノード摂動法:}\quad &\Delta \mathbf{W}_\ell^{\mathrm{NP}}=- \eta  \frac{\delta \mathcal{L}}{\sigma} \mathbf{v}_\ell \mathbf{z}_{\ell}^\top, &\Delta \mathbf{b}_\ell^{\mathrm{NP}} =- \eta  \frac{\delta \mathcal{L}}{\sigma} \mathbf{v}_\ell
\end{align}
$$

でパラメータを更新する．

#### 不偏推定量であることの証明
各手法の更新則が勾配の不偏推定量 (unbiased estimator) であることを示す．まず方向微分 (directional derivative) を導入する．関数 $f$ について点 $\mathbf{u}$ における方向 $\mathbf{v}$ の方向微分は

$$
\begin{equation}
\nabla_\mathbf{v}f(\mathbf{u}):= \lim_{h\to 0} \frac{f(\mathbf{u}+h\mathbf{v}) - f(\mathbf{u})}{h}
\end{equation}
$$

として定義される．また $f$ が点 $\mathbf{u}$ において微分可能なら

$$
\begin{equation}
\nabla_\mathbf{v}f(\mathbf{u})=\nabla f(\mathbf{u})\cdot \mathbf{v}\left(=\frac{\partial f(\mathbf{u})}{\partial \mathbf{u}}\cdot \mathbf{v}\right)
\end{equation}
$$

が成り立つ．ここで，$\nabla f(\mathbf{u})\cdot \mathbf{v}$ を Jacobian-vector product (JVP) と呼び，$f(\mathbf{u})\in \mathbb{R}$ の場合，$\nabla f(\mathbf{u})\cdot \mathbf{v}\in \mathbb{R}$ となる．このJVPを有限差分 (finite difference) を用いて近似計算すると\footnote{JVPは順方向自動微分 (Forward-mode automatic differentiation) により計算でき，有限差分法よりも数値的に安定する (順方向自動微分はPythonライブラリのJAX等に実装されている)．Forward Gradientは順方向自動微分を採用して重み摂動法をより安定させた手法である．}，

$$
\begin{equation}
\nabla f(\mathbf{u})\cdot \mathbf{v} \approx \frac{f(\mathbf{u}+\epsilon \mathbf{v}) - f(\mathbf{u})}{\epsilon}
\end{equation}
$$

となる (ただし，$0 < \epsilon \ll 1$)．

まず，重み摂動法について考える．モデルのパラメータを $\boldsymbol{\theta} \in \mathbb{R}^P$ とする．これは $\mathbf{W}_\ell$ および $\mathbf{b}_\ell$ をまとめたベクトルであり，$P$ はパラメータ空間の次元である．$\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_P)\ (\in \mathbb{R}^P)$ とすると，$\sigma\to 0$ の場合，

$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\cdot \mathbf{v} = \frac{\mathcal{L}(\boldsymbol{\theta}+\sigma \mathbf{v}) - \mathcal{L}(\boldsymbol{\theta})}{\sigma}=\frac{\delta \mathcal{L}}{\sigma}
\end{equation}
$$

となるので，

$$
\begin{align}
\mathbb{E}\left[\frac{\delta \mathcal{L}}{\sigma}\mathbf{v}\right] &=
\mathbb{E}\left[\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\cdot \mathbf{v}\right)\mathbf{v}\right]\\
&=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \mathbb{E}[\mathbf{v} \mathbf{v}^\top]=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\quad \left(\because \mathbb{E}[\mathbf{v} \mathbf{v}^\top]=\mathbf{I}_P\right)
\end{align}
$$

が成立する．SGDでパラメータ更新する場合は

$$
\begin{equation}
\mathbb{E}[\Delta \mathbf{W}_\ell]=-\eta \dfrac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell},\quad \mathbb{E}[\Delta \mathbf{b}_\ell]=-\eta \dfrac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}
\end{equation}
$$

であればいいので，$(\boldsymbol{\theta}, \mathbf{v}) \to (\mathbf{W}_\ell, \mathbf{V}_\ell), (\mathbf{b}_\ell, \mathbf{v}_\ell)$ と置き換えて

$$
\begin{equation}
\Delta \mathbf{W}_\ell^{\mathrm{WP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{V}_\ell,\quad \Delta \mathbf{b}_\ell^{\mathrm{WP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell
\end{equation}
$$

となる．ノード摂動法はパラメータのうちバイアス項のみを摂動する重み摂動法であると解釈できるため，$\Delta \mathbf{b}_\ell^{\mathrm{NP}}:=\Delta \mathbf{b}_\ell^{\mathrm{WP}}$ とすることができる．ここで

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{W}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell}\frac{\partial \mathbf{u}_\ell}{\partial \mathbf{b}_\ell}\right) \mathbf{z}_\ell^\top\quad \left(\because \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{b}_\ell}=\mathbf{1}\right)\\
&=\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}\mathbf{z}_\ell^\top
\end{align}
$$

が成り立つので，ノード摂動法の更新則は

$$
\begin{equation}
\Delta \mathbf{W}_\ell^{\mathrm{NP}}:=\Delta \mathbf{b}_\ell^{\mathrm{NP}}\mathbf{z}_\ell^\top=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell\mathbf{z}_\ell^\top,\quad \Delta \mathbf{b}_\ell^{\mathrm{NP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell
\end{equation}
$$

と設定できる．


Chaotic neural dynamics facilitate probabilistic computations through sampling

Effective Learning with Node Perturbation in Multi-Layer Neural Networks (fig1は図の参考になる．)
On the stability and scalability of node perturbation learning
Node perturbation learning without noiseless baseline


重み摂動法 (Weight perturbation; WP)
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021006
Weight Perturbation Learning Performs Similarly or Better than Node Perturbation on Broad Classes of Temporally Extended Tasks

A. Dembo and T. Kailath, Model-Free Distributed Learning, IEEE Trans. Neural Networks 1, 58 (1990).

G. Cauwenberghs, A Fast Stochastic Error-Descent Algorithm for Supervised Learning and Optimization, in Advances in Neural Information Processing Systems (Morgan Kaufmann, Burlington, 1993), Vol. 5, pp. 244–251

https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439155/full

重み摂動法とノード摂動法では，ノード摂動法の方が活動依存的であり，生理学的とされる．

## RNNにおける摂動学習

https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439155/full

## contrastive divergenceとの関係