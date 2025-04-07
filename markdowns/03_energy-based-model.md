# 第3章：エネルギーベースモデル
本章では**エネルギーベースモデル (energy-based models; EBMs)** という枠組みに含まれるモデルを紹介する．エネルギーベースモデルではネットワークの状態をスカラー値に変換するエネルギー関数 (あるいはコスト関数) を定義し，推論時と学習時の双方においてエネルギーを最小化するようにネットワークの状態を更新する．{cite:p}`LeCun2006-dt`

入力 $\mathbf{x}\in \mathbb{R}^d$, エネルギー関数 $E_\theta: \mathbb{R}^d\to \mathbb{R}$を考える．

$$
\begin{align}
p_\theta(\mathbf{x})&=\frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}\\
Z_\theta &= \int \exp(-E_\theta(\mathbf{x})) d\mathbf{x}
\end{align}
$$

$Z_\theta$は分配関数．

## Hopfield モデル

{cite:p}`Hopfield1982-vu`で提案．始めは1と0の状態を取った．

Hopfieldモデルと呼ばれることが多いが，Amariの先駆的研究{cite:p}`Amari1972-fq`を踏まえAmari-Hopfieldモデルと呼ばれることもある．

次のような連続時間線形モデルを考える．シナプス前活動を$\mathbf{x}\in \mathbb{R}^n$, 後活動を$\mathbf{y}\in \mathbb{R}^m$, 重み行列を$\mathbf{W}\in \mathbb{R}^{m\times n}$とする．

$$
\begin{equation}
\frac{d\mathbf{y}}{dt}=-\mathbf{y}+\mathbf{W}\mathbf{x}+\mathbf{b}
\end{equation}
$$

ここで$\dfrac{\partial\mathcal{L}}{\partial\mathbf{y}}:=-\dfrac{d\mathbf{y}}{dt}$となるような$\mathcal{L}\in \mathbb{R}$を仮定すると，

$$
\begin{equation}
\mathcal{L}=\int \left(\mathbf{y}-\mathbf{W}\mathbf{x}-\mathbf{b}\right)\ d\mathbf{y}=\frac{1}{2}\|\mathbf{y}\|^2-\mathbf{y}^\top \mathbf{W}\mathbf{x}-\mathbf{y}^\top \mathbf{b}
\end{equation}
$$

となる． これをさらに$\mathbf{W}$で微分すると，

$$
\begin{equation}
\dfrac{\partial\mathcal{L}}{\partial\mathbf{W}}=-\mathbf{y}\mathbf{x}^\top\Rightarrow
\frac{d\mathbf{W}}{dt}=-\dfrac{\partial\mathcal{L}}{\partial\mathbf{W}}=\mathbf{y}\mathbf{x}^\top=(\text{post})\cdot (\text{pre})^\top
\end{equation}
$$

となり，Hebb則が導出できる．

画像の復元を行う．
エネルギー関数

$$
\begin{equation}
E=-{\frac 12}\sum _{{i,j}}{w_{{ij}}{s_{i}}{s_{j}}}+\sum _{i}{\theta _{i}}{s_{i}}=-{\frac 12}\mathbf{s}^\top\mathbf{W}\mathbf{s}+\mathbf{\theta}^\top\mathbf{s}
\end{equation}
$$

を最小化するように内部状態 $\mathbf{s}$ を更新．

$$
\begin{equation}
\mathbf{s}\leftarrow \text{sign}\left(\mathbf{W}\mathbf{s}-\mathbf{\theta}\right)
\end{equation}
$$

### 稠密連想記憶 (dense associative memory) モデル
**Dense Associative Memory (DAM)** モデル（Modern Hopfield networksとも呼ばれる）．

- Krotov, Dmitry, and John J. Hopfield. 2016. “Dense Associative Memory for Pattern Recognition.” arXiv. arXiv. <http://arxiv.org/abs/1606.01164>.
- Krotov, Dmitry, and John Hopfield. 2018. “Dense Associative Memory Is Robust to Adversarial Inputs.” Neural Computation 30 (12): 3151–67.
- Krotov, Dmitry, and John J. Hopfield. 2019. “Unsupervised Learning by Competing Hidden Units.” Proceedings of the National Academy of Sciences of the United States of America 116 (16): 7723–31.


- Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, et al. 2020. “Hopfield Networks Is All You Need.” arXiv. arXiv. <http://arxiv.org/abs/2008.02217>.

深層ニューラルネットワークへの応用．


- Krotov, Dmitry, and John J. Hopfield. 2020. “Large Associative Memory Problem in Neurobiology and Machine Learning.” <https://openreview.net/pdf?id=X4y_10OX-hX>

“Hopfield Networks Is All You Need.”の論文における非生理学的3ニューロン相互作用の緩和．

## Boltzmann マシン
(Boltzmann machine)

### 制限 Boltzmann マシン
(Restricted Boltzmann machine) 
(cf.) <http://deeplearning.net/tutorial/rbm.html>

離散の観測変数(visible variable) $\mathbf{v}$, 潜在変数(hidden variable) $\mathbf{h}$とする．各ユニットの値は$\{0, 1\}$の2値 (binary)である．

エネルギー関数を

$$
\begin{equation}
E_\theta(\mathbf{v}, \mathbf{h})=-\mathbf{b}^T \mathbf{v} - \mathbf{c}^T \mathbf{h} + \mathbf{v}^T \mathbf{W} \mathbf{h}
\end{equation}
$$

とする．ただし，$\theta=\{\mathbf{W}, \mathbf{b}, \mathbf{c}\}$

シグモイド関数を

$$
\begin{equation}
\sigma(x) = \frac{1}{1+\exp(-x)}
\end{equation}
$$

とする．

### 訓練データで学習
$$
\begin{align}
p_\theta(\mathbf{h}|\mathbf{v})&=\prod_i p_\theta(h_i=1|\mathbf{v})=\prod_i \sigma(c_i + W_i \mathbf{v})\\
p_\theta(\mathbf{v}|\mathbf{h})&=\prod_j p_\theta(v_j=1|\mathbf{h})=\prod_j \sigma(b_j + W_j^T \mathbf{h})
\end{align}
$$

## スパース符号化モデル
### Sparse codingと生成モデル
**Sparse codingモデル** {cite:p}`Olshausen1996-xe` {cite:p}`Olshausen1997-qu`はV1のニューロンの応答特性を説明する**線形生成モデル** (linear generative model)である．まず，画像パッチ $\mathbf{x}$ が基底関数(basis function) $\mathbf{\Phi} = [\phi_j]$ のノイズを含む線形和で表されるとする (係数は $\mathbf{r}=[r_j]$ とする)．

$$
\begin{equation}
\mathbf{x} = \sum_j r_j \phi_j +\boldsymbol{\epsilon}= \mathbf{\Phi} \mathbf{r}+ \boldsymbol{\epsilon}
\end{equation}
$$

ただし，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ である．このモデルを神経ネットワークのモデルと考えると， $\mathbf{\Phi}$ は重み行列，係数 $\mathbf{r}$ は入力よりも高次の神経細胞の活動度を表していると解釈できる．ただし，$r_j$ は負の値も取るので単純に発火率と捉えられないのはこのモデルの欠点である．

Sparse codingでは神経活動 $\mathbf{r}$ が潜在変数の推定量を表現しているという仮定の下，少数の基底で画像 (や目的変数)を表すことを目的とする．要は上式において，ほとんどが0で，一部だけ0以外の値を取るという疎 (=sparse)な係数$\mathbf{r}$を求めたい．

### 確率的モデルの記述
入力される画像パッチ $\mathbf{x}_i\ (i=1, \ldots, N)$ の真の分布を $p_{data}(\mathbf{x})$ とする．また，$\mathbf{x}$ の生成モデルを $p(\mathbf{x}|\mathbf{\Phi})$ とする．さらに潜在変数 $\mathbf{r}$ の事前分布 (prior)を $p(\mathbf{r})$, 画像パッチ $\mathbf{x}$ の尤度 (likelihood)を $p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})$ とする．このとき，

$$
\begin{equation}
p(\mathbf{x}|\mathbf{\Phi})=\int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r}
\end{equation}
$$

が成り立つ．$p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})$は，(1)式においてノイズ項を$\boldsymbol{\epsilon} \sim\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$としたことから，

$$
\begin{equation}
p(\mathbf{x}|\ \mathbf{r}, \mathbf{\Phi})=\mathcal{N}\left(\mathbf{x}|\ \mathbf{\Phi} \mathbf{r}, \sigma^2 \mathbf{I} \right)=\frac{1}{Z_{\sigma}} \exp\left(-\frac{\|\mathbf{x} - \mathbf{\Phi} \mathbf{r}\|^2}{2\sigma^2}\right)
\end{equation}
$$

と表せる．ただし，$Z_{\sigma}$は規格化定数である．

### 事前分布の設定
事前分布$p(\mathbf{r})$としては，0においてピークがあり，裾の重い(heavy tail)を持つsparse distributionあるいは **super-Gaussian distribution** (Laplace 分布やCauchy分布などGaussian分布よりもkurtoticな分布)を用いるのが良い．このような分布では，$\mathbf{r}$の各要素$r_i$はほとんど0に等しく，ある入力に対しては大きな値を取る．$p(\mathbf{r})$は一般化して式(4), (5)のように表記する．

$$
\begin{align}
p(\mathbf{r})&=\prod_j p(r_j)\\
p(r_j)&=\frac{1}{Z_{\beta}}\exp \left[-\beta S(r_j)\right]
\end{align}
$$

ただし，$\beta$は逆温度(inverse temperature), $Z_{\beta}$は規格化定数 (分配関数) である．これらの用語は統計力学における正準分布 (Boltzmann分布)から来ている．$S(x)$と分布の関係をまとめた表が以下となる．

$$
\begin{table}[h]
\centering
\begin{tabular}{ccccc}
\hline
$S(r)$ & $\dfrac{dS(r)}{dr}$ & $p(r)$ & 分布名 & 尖度(kurtosis) \\
\hline
$r^2$ & $2r$ & $\dfrac{1}{\alpha \sqrt{2\pi}}\exp\left(-\dfrac{r^2}{2\alpha^2}\right)$ & Gaussian 分布 & 0 \\
$\vert r\vert$ & $\text{sign}(r)$ & $\dfrac{1}{2\alpha}\exp\left(-\dfrac{\vert r\vert}{\alpha}\right)$ & Laplace 分布 & 3.0 \\
$\ln (\alpha^2+r^2)$ & $\dfrac{2r}{\alpha^2+r^2}$ & $\dfrac{\alpha}{\pi}\dfrac{1}{\alpha^2+r^2}=\dfrac{\alpha}{\pi}\exp[-\ln (\alpha^2+r^2)]$ & Cauchy 分布 & - \\
\hline
\end{tabular}
\end{table}
$$

分布$p(r)$や$S(r)$を描画すると次のようになる．

### 目的関数の設定と最適化
最適な生成モデルを得るために，入力される画像パッチの真の分布 $p_{data}(\mathbf{x})$と$\mathbf{x}$の生成モデル $p(\mathbf{x}|\mathbf{\Phi})$を近づける．このために，2つの分布のKullback-Leibler ダイバージェンス $D_{\text{KL}}\left(p_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$を最小化したい．しかし，真の分布は得られないので，経験分布 

$$
\begin{equation}
\hat{p}_{data}(\mathbf{x}):=\frac{1}{N}\sum_{i=1}^N \delta(\mathbf{x}-\mathbf{x}_i)
\end{equation}
$$

を近似として用いる ($\delta(\cdot)$ はDiracのデルタ関数である)．ゆえに$D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$を最小化する．

$$
\begin{align}
D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)&=\int \hat{p}_{data}(\mathbf{x}) \log \frac{\hat{p}_{data}(\mathbf{x})}{p(\mathbf{x}|\mathbf{\Phi})} d\mathbf{x}\\
&=\mathbb{E}_{\hat{p}_{data}} \left[\ln \frac{\hat{p}_{data}(\mathbf{x})}{p(\mathbf{x}|\mathbf{\Phi})}\right]\\
&=\mathbb{E}_{\hat{p}_{data}} \left[\ln \hat{p}_{data}(\mathbf{x})\right]-\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]
\end{align}
$$

が成り立つ．(7)式の1番目の項は一定なので，$D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$ を最小化するには$\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]$を最大化すればよい．ここで，

$$
\begin{equation}
\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]=\sum_{i=1}^N \hat{p}_{data}(\mathbf{x}_i)\ln p(\mathbf{x}_i|\mathbf{\Phi})=\frac{1}{N}\sum_{i=1}^N \ln p(\mathbf{x}_i|\mathbf{\Phi})
\end{equation}
$$

が成り立つ．また，(2)式より

$$
\begin{equation}
\ln p(\mathbf{x}|\mathbf{\Phi})=\ln \int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r}
\end{equation}
$$

が成り立つので，近似として $\displaystyle \int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r}$ を $p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r}) \left(=p(\mathbf{x}, \mathbf{r}| \mathbf{\Phi})\right)$ で評価する．これらの近似の下，最適な$\mathbf{\Phi}=\mathbf{\Phi}^*$は次のようにして求められる．

$$
\begin{align}
\mathbf{\Phi}^*&=\text{arg} \min_{\mathbf{\Phi}} \min_{\mathbf{r}} D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \| p(\mathbf{x}|\mathbf{\Phi})\right)\\
&=\text{arg} \max_{\mathbf{\Phi}} \max_{\mathbf{r}} \mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]\\
&= \text{arg} \max_{\mathbf{\Phi}}\sum_{i=1}^N \max_{\mathbf{r}_i} \ln p(\mathbf{x}_i|\mathbf{\Phi})\\
&\approx \text{arg} \max_{\mathbf{\Phi}}\sum_{i=1}^N \max_{\mathbf{r}_i} \ln p(\mathbf{x}_i|\mathbf{r}_i, \mathbf{\Phi})p(\mathbf{r}_i)\\
&=\text{arg}\min_{\mathbf{\Phi}} \sum_{i=1}^N \min_{\mathbf{r}_i}\ E(\mathbf{x}_i, \mathbf{r}_i|\mathbf{\Phi})
\end{align}
$$

ただし，$\mathbf{x}_i$に対する神経活動を $\mathbf{r}_i$とした．また，$E(\mathbf{x}, \mathbf{r}|\mathbf{\Phi})$はコスト関数であり，次式のように表される．

$$
\begin{align}
E(\mathbf{x}, \mathbf{r}|\mathbf{\Phi}):=&-\ln p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})\\
=&\underbrace{\left\|\mathbf{x}-\mathbf{\Phi} \mathbf{r}\right\|^2}_{\text{preserve information}} + \lambda \underbrace{\sum_j S\left(r_j\right)}_{\text{sparseness of}\ r_j}
\end{align}
$$

ただし，$\lambda=2\sigma^2\beta$は正則化係数(この式から逆温度$\beta$が正則化の度合いを調整するパラメータであることがわかる．)であり，1行目から2行目へは式(3), (4), (5)を用いた．ここで，第1項が復元損失，第2項が罰則項 (正則化項)となっている．

式(9)で表される最適化手順を最適な$\mathbf{r}$と$\mathbf{\Phi}$を求める過程に分割しよう．まず， $\mathbf{\Phi}$を固定した下で$E(\mathbf{x}_n, \mathbf{r}_i|\mathbf{\Phi})$を最小化する$\mathbf{r}_i=\hat{\mathbf{r}}_i$を求める．

$$
\begin{equation}
\hat{\mathbf{r}}_i=\text{arg}\min_{\mathbf{r}_i}E(\mathbf{x}_i, \mathbf{r}_i|\mathbf{\Phi})\ \left(= \text{arg}\max_{\mathbf{r}_i}p(\mathbf{r}_i|\mathbf{x}_i)\right)
\end{equation}
$$

これは $\mathbf{r}$ について **MAP推定** (maximum a posteriori estimation)を行うことに等しい．次に$\hat{\mathbf{r}}$を用いて

$$
\begin{equation}
\mathbf{\Phi}^*=\text{arg}\min_{\mathbf{\Phi}} \sum_{i=1}^N E(\mathbf{x}_i, \hat{\mathbf{r}}_i|\mathbf{\Phi})\ \left(= \text{arg}\max_{\mathbf{\Phi}} \prod_{i=1}^N p(\mathbf{x}_i|\hat{\mathbf{r}}_i, \mathbf{\Phi})\right)
\end{equation}
$$

とすることにより，$\mathbf{\Phi}$を最適化する．こちらは $\mathbf{\Phi}$ について **最尤推定** (maximum likelihood estimation)を行うことに等しい．

### Locally competitive algorithm (LCA) 
$\mathbf{r}$の勾配法による更新則は，$E$の微分により次のように得られる．

$$
\begin{equation}
\frac{d \mathbf{r}}{dt}= -\frac{\eta_\mathbf{r}}{2}\frac{\partial E}{\partial \mathbf{r}}=\eta_\mathbf{r} \cdot\left[\mathbf{\Phi}^\top (\mathbf{x}-\mathbf{\Phi}\mathbf{r})- \frac{\lambda}{2}S'\left(\mathbf{r}\right)\right]
\end{equation}
$$

ただし，$\eta_{\mathbf{r}}$は学習率である．この式により$\mathbf{r}$が収束するまで最適化するが，単なる勾配法ではなく，{cite:p}`Olshausen1996-xe`では**共役勾配法** (conjugate gradient method)を用いている．しかし，共役勾配法は実装が煩雑で非効率であるため，より効率的かつ生理学的な妥当性の高い学習法として，**LCA**  (locally competitive algorithm)が提案されている {cite:p}`Rozell2008-wp`．LCAは**側抑制** (local competition, lateral inhibition)と**閾値関数** (thresholding function)を用いる更新則である．LCAによる更新を行うRNNは通常のRNNとは異なり，コスト関数(またはエネルギー関数)を最小化する動的システムである．このような機構はHopfield networkで用いられているために，Olshausenは**Hopfield trick**と呼んでいる．

#### 軟判定閾値関数を用いる場合 (ISTA)
$S(x)=|x|$とした場合の閾値関数を用いる手法として**ISTA**(Iterative Shrinkage Thresholding Algorithm)がある．ISTAはL1-norm正則化項に対する近接勾配法で，要はLasso回帰に用いる勾配法である．

解くべき問題は次式で表される．

$$
\begin{equation}
\mathbf{r} = \mathop{\rm arg~min}\limits_{\mathbf{r}}\left\{\|\mathbf{x}-\mathbf{\Phi}\mathbf{r}\|^2_2+\lambda\|\mathbf{r}\|_1\right\}
\end{equation}
$$

詳細は後述するが，次のように更新することで解が得られる．

- $\mathbf{r}(0)$を要素が全て0のベクトルで初期化：$\mathbf{r}(0)=\mathbf{0}$
- $\mathbf{r}_*(t+1)=\mathbf{r}(t)+\eta_\mathbf{r}\cdot \mathbf{\Phi}^\top(\mathbf{x}-\mathbf{\Phi}\mathbf{r}(t))$
- $\mathbf{r}(t+1) = \Theta_\lambda(\mathbf{r}_*(t+1))$
- $\mathbf{r}$が収束するまで2と3を繰り返す

ここで$\Theta_\lambda(\cdot)$は**軟判定閾値関数** (Soft thresholding function)と呼ばれ，次式で表される．

$$
\begin{equation}
\Theta_\lambda(y)= 
\begin{cases} 
y-\lambda & (y>\lambda)\\ 
0 & (-\lambda\leq y\leq\lambda)\\ 
 y+\lambda & (y<-\lambda) 
\end{cases}
\end{equation}
$$

$\Theta_\lambda(\cdot)$を関数として定義すると次のようになる．また，ReLU (ランプ関数)は`max(x, 0)`で実装できる．この点から考えればReLUを軟判定非負閾値関数 (soft nonnegative thresholding function)と捉えることもできる {cite:p}`Papyan2018-yr`．

なお，軟判定閾値関数は次の目的関数$C$を最小化する$x$を求めることで導出できる．

$$
\begin{equation}
C=\frac{1}{2}(y-x)^2+\lambda |x|
\end{equation}
$$

ただし，$x, y, \lambda$はスカラー値とする．$|x|$が微分できないが，これは場合分けを考えることで解決する．$x\geq 0$を考えると，(6)式は

$$
\begin{equation}
C=\frac{1}{2}(y-x)^2+\lambda x = \{x-(y-\lambda)\}^2+\lambda(y-\lambda)
\end{equation}
$$

となる．(7)式の最小値を与える$x$は場合分けをして考えると，$y-\lambda\geq0$のとき二次関数の頂点を考えて$x=y-\lambda$となる． 一方で$y-\lambda<0$のときは$x\geq0$において単調増加な関数となるので，最小となるのは$x=0$のときである．同様の議論を$x\leq0$に対しても行うことで (5)式が得られる．

なお，閾値関数としては軟判定閾値関数だけではなく，硬判定閾値関数や$y=x - \text{tanh}(x)$ (Tanh-shrink)など様々な関数を用いることができる．

### 重み行列の更新則
$\mathbf{r}$が収束したら勾配法により$\mathbf{\Phi}$を更新する．

$$
\begin{equation}
\Delta \phi_i(\boldsymbol{x}) = -\eta \frac{\partial E}{\partial \mathbf{\Phi}}=\eta\cdot\left[\left(\mathbf{x}-\mathbf{\Phi}\mathbf{r}\right)\mathbf{r}^\top\right]
\end{equation}
$$

### Sparse coding networkの実装
ネットワークは入力層を含め2層の単純な構造である．今回は，入力はランダムに切り出した16×16 (＝256)の画像パッチとし，これを入力層の256個のニューロンが受け取るとする．入力層のニューロンは次層の100個のニューロンに投射するとする．100個のニューロンが入力をSparseに符号化するようにその活動および重み行列を最適化する．

## 予測符号化モデル
$u$ を $w$ に変更．

### 観測世界の階層的予測
**階層的予測符号化(hierarchical predictive coding; HPC)** は{cite:p}`Rao1999-zv`により導入された．構築するネットワークは入力層を含め，3層のネットワークとする．LGNへの入力として画像 $\mathbf{x} \in \mathbb{R}^{n_0}$を考える．画像 $\mathbf{x}$ の観測世界における隠れ変数，すなわち**潜在変数** (latent variable)を$\mathbf{r} \in \mathbb{R}^{n_1}$とし，ニューロン群によって発火率で表現されているとする (真の変数と $\mathbf{r}$は異なるので文字を分けるべきだが簡単のためにこう表す)．このとき，

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
$\mathbf{r}$の事前分布$p(\mathbf{r})$はCauchy分布を用いる．$p(\mathbf{r})$の負の対数事前分布を$g(\mathbf{r}):=-\log p(\mathbf{r})$としておく．

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
\frac{d \mathbf{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} \mathbf{U}^{T}\bigg[\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\odot\underbrace{(\mathbf{x}-f(\mathbf{z}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\mathbf{r})\Bigg)\\
\frac{d \mathbf{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(\mathbf{U}^h)^\top\bigg[\frac{\partial f(\mathbf{z}^h)}{\partial \mathbf{z}^h}\odot\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\mathbf{r}^h)\Bigg)
\end{align}
$$

ただし，$k_1$は更新率 (updating rate)である．または，発火率の時定数を$\tau:=1/k_1$として，$k_1$は発火率の時定数$\tau$の逆数であると捉えることもできる．ここで1番目の式において，中間表現 $\mathbf{r}$ のダイナミクスはbottom-up errorとtop-down errorで記述されている．このようにbottom-up errorが $\mathbf{r}$ への入力となることは自然に導出される．なお，top-down errorに関しては高次からの予測 (prediction)の項 $f(\mathbf{x}^h)$とleaky-integratorとしての項 $-\mathbf{r}$に分割することができる．また$\mathbf{U}^\top, (\mathbf{U}^h)^\top$は重み行列の転置となっており，bottom-upとtop-downの投射において対称な重み行列を用いることを意味している．$-g'(\mathbf{r})$は発火率を抑制してスパースにすることを目的とする項だが，無理やり解釈をすると自己再帰的な抑制と言える．