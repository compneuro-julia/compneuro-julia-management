## 生成的知覚とスパース符号化・予測符号化
### 生成的知覚
知覚 (perception) とは，外界からの刺激を感覚受容器により感受し，刺激を意味付けする過程である．

この過程は一般的に原因から結果を推定する順推論であると捉えられる．

一方で，知覚は結果（感覚入力）から外界に存在する潜在的な原因を推定する逆推論の過程であると捉える枠組みがあり，これを生成的知覚 (generative perception) と呼ぶ．生成的知覚は，生成モデル (generative model) と呼ばれるモデルを必要とする．このため，まず生成モデルについて説明をする．

観測データ（感覚入力）を $\mathbf{x}$ とし，その確率分布を $p_{\mathrm{data}}(\mathbf{x})$ とする．$p_{\mathrm{data}}(\cdot)$ が既知であれば，データを生成（サンプリング）できるが，ほとんどの場合で $p_{\mathrm{data}}(\cdot)$ は未知である．ここで，パラメータ $\theta$ を伴う確率モデル $p_\theta (\cdot)$ を導入する．$p_\theta (\cdot)$ が $p_{\mathrm{data}}(\cdot)$ を近似できれば，観測データに近いデータを $p_\theta (\cdot)$ に基づいて生成することが可能である．この $p_\theta (\cdot)$ が生成モデルであり，生成モデルを訓練するとは $p_\theta (\cdot)$ が $p_{\mathrm{data}}(\cdot)$ を近似するようにパラメータ $\theta$ を調整することである．

外界の変数がすべて感覚入力として得られる状態，すなわち全て観測可能 (fully visible) であればよいが，基本的には部分的にのみ観測可能 (partially visible) である．観測できない変数を潜在変数 (latent variable) $\mathbf{z}$ とする．潜在変数に基づいて観測データが生成される過程をモデル化すると，

$$
\begin{equation}
p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z}) p_\theta(\mathbf{z})
\end{equation}
$$

となる．

$$
p(\mathbf{z} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}
$$

前章で扱った主成分分析や独立成分分析も生成モデルと捉えることが可能である．
主成分分析は...
独立成分分析は...

生成モデル→階層的生成モデル→

### 階層的生成モデル
生成モデルの表現力を高めるため，生成モデルを階層化することを考えよう．

本章では階層的生成モデルを導入し，それからスパース符号化，予測符号化について説明する．



### スパース符号化と生成モデル
**スパース符号化モデル** (Sparse coding model) \citep{`Olshausen1996-xe` \citep{`Olshausen1997-qu`はV1のニューロンの応答特性を説明する**線形生成モデル** (linear generative model)である．まず，画像パッチ $\mathbf{x}$ が基底関数(basis function) $\mathbf{\Phi} = [\phi_j]$ のノイズを含む線形和で表されるとする (係数は $\mathbf{r}=[r_j]$ とする)．

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
事前分布$p(\mathbf{r})$としては，0においてピークがあり，裾の重い(heavy tail)を持つsparse distributionあるいは **super-Gaussian distribution** (Laplace分布やCauchy分布などGaussian分布よりもkurtoticな分布) を用いるのが良い．このような分布では，$\mathbf{r}$の各要素$r_i$はほとんど0に等しく，ある入力に対しては大きな値を取る．$p(\mathbf{r})$は一般化して次のように表記する．

$$
\begin{align}
p(\mathbf{r})&=\prod_j p(r_j)\\
p(r_j)&=\frac{1}{Z_{\beta}}\exp \left[-\beta S(r_j)\right]
\end{align}
$$

ただし，$\beta$は逆温度(inverse temperature), $Z_{\beta}$は規格化定数（分配関数）である．これらの用語は統計力学における正準分布 (Boltzmann分布)から来ている．$S(x)$と分布の関係をまとめた表が以下となる．

$$
\begin{array}{c|c|c|c|c}
\hline
S(r) & \dfrac{dS(r)}{dr} & p(r) & \text{分布名} & \text{尖度} \\
\hline
r^2 & 2r & \dfrac{1}{\alpha \sqrt{2\pi}}\exp\left(-\dfrac{r^2}{2\alpha^2}\right) & \text{Gaussian 分布} & 0 \\
\vert r\vert & \text{sign}(r) & \dfrac{1}{2\alpha}\exp\left(-\dfrac{\vert r\vert}{\alpha}\right) & \text{Laplace 分布} & 3.0 \\
\ln (\alpha^2+r^2) & \dfrac{2r}{\alpha^2+r^2} & \dfrac{\alpha}{\pi}\dfrac{1}{\alpha^2+r^2}=\dfrac{\alpha}{\pi}\exp[-\ln (\alpha^2+r^2)] & \text{Cauchy 分布} & - \\
\hline
\end{array}
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

ただし，$\eta_{\mathbf{r}}$は学習率である．この式により$\mathbf{r}$が収束するまで最適化するが，単なる勾配法ではなく，\citep{`Olshausen1996-xe`では**共役勾配法** (conjugate gradient method)を用いている．しかし，共役勾配法は実装が煩雑で非効率であるため，より効率的かつ生理学的な妥当性の高い学習法として，**LCA**  (locally competitive algorithm)が提案されている \citep{`Rozell2008-wp`．LCAは**側抑制** (local competition, lateral inhibition)と**閾値関数** (thresholding function)を用いる更新則である．LCAによる更新を行うRNNは通常のRNNとは異なり，コスト関数(またはエネルギー関数)を最小化する動的システムである．このような機構はHopfield networkで用いられているために，Olshausenは**Hopfield trick**と呼んでいる．

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

$\Theta_\lambda(\cdot)$を関数として定義すると次のようになる．また，ReLU (ランプ関数)は`max(x, 0)`で実装できる．この点から考えればReLUを軟判定非負閾値関数 (soft nonnegative thresholding function)と捉えることもできる \citep{`Papyan2018-yr`．

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

### スパース符号化モデルの実装
ネットワークは入力層を含め2層の単純な構造である．今回は，入力はランダムに切り出した16×16 (＝256)の画像パッチとし，これを入力層の256個のニューロンが受け取るとする．入力層のニューロンは次層の100個のニューロンに投射するとする．100個のニューロンが入力をSparseに符号化するようにその活動および重み行列を最適化する．