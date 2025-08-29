Bayesian network


固定されていないパラメータの同時分布でモデルを定義することとするのが，自然ですね．完全な生成過程の説明になります．以下に例を列挙します．

通常の生成モデル：

$$
p(\mathbf{x}, \theta) = p(\mathbf{x} \mid \theta) p(\theta) 
$$

ベイズ線形回帰モデル：

$$
p(y, \theta \mid \mathbf{x}) = p(y\mid \mathbf{x}, \theta) p(\theta)
$$


本章を執筆する上で全般的に参考した書籍は

### 確率モデルの設計とエネルギーベースモデル
前節で定義した事後分布や，予測分布は確率モデルの具体的な形状を定義しなければ計算することはできない．確率モデルを定義する上で，代表的な多変量正規分布およびそれが属する指数型分布族，さらに指数分布族と関連し，より扱いやすい枠組みであるエネルギーベースモデル (energy based model) を本項では紹介する．

#### 多変量正規分布
まず，多変量正規分布 (ガウス分布) を導入する．1次元の場合，正規分布は次の確率密度関数で表される．

$$
\begin{equation}
\mathcal{N}(x \mid \mu, \sigma^2) 
\coloneq \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
\end{equation}
$$

ここで，$\mu \in \mathbb{R}$ は平均，$\sigma^2 > 0$ は分散を表し，$\sigma$ は標準偏差である．この式を $x \in \mathbb{R}$ から $d$ 次元のベクトル $\mathbf{x} \in \mathbb{R}^d$ に拡張すると，分布は多変量正規分布 (multivariate normal distribution) となる．

$$
\begin{equation}
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) 
\coloneq \frac{1}{\sqrt{(2\pi)^d \, |\boldsymbol{\Sigma}|}}
\exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
\end{equation}
$$

ここで，$\boldsymbol{\mu} \in \mathbb{R}^d$ は各成分の平均を並べた平均ベクトル，$\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ は共分散行列 (covariance matrix) である．

#### 指数型分布族
多変量正規分布が属する確率分布の枠組みを指数型分布族 (exponential family) と呼ぶ．指数型分布族には多数の分布が属しており，他にはポアソン分布，多項分布，ベルヌーイ分布などが属する．指数型分布族の一般形は

$$
p(x\mid \theta) = h(x)\cdot c(\theta) \exp(\eta(\theta)t(x))
$$ 

と定義される．

指数型分布族同士であれば，共役となる．

p.36

#### エネルギーベースモデル
次に指数分布族と関連して扱いやすい枠組みであるエネルギーベースモデル (energy-based model; EBM) を紹介する．エネルギーベースモデルでは，確率変数 $\mathbf{x}$ の確率密度関数を，パラメータ $\theta$ を持つエネルギー関数 $E(\mathbf{x}; \theta):\ \mathbb{R}^{d}\mathbb{\rightarrow R}$ および Gibbs-Boltzmann分布を用いて次のように表す．

$$
\begin{equation}
p(\mathbf{x} \mid \theta)\  = \frac{\exp\left(-E\left(\mathbf{x}; \theta \right)\right)}{Z(\theta)},\quad Z(\theta) = \int \exp(-E\left( \mathbf{x}'; \theta\right))\,\mathrm{d}\mathbf{x}'
\end{equation}
$$

ただし，$Z(\theta)$は規格化定数あるいは分配関数 (partition function) である．エネルギーベースモデルの利点は，確率変数およびパラメータを入力として1次元の値を返すエネルギー関数さえ設計すれば複雑な確率分布を構成する必要はないという点である．このエネルギー関数は，ある状態 $\mathbf{x}$ の「好ましさ」や「自然さ」を定量的に評価するものであり，エネルギーが小さいほどその状態がより実現しやすいと解釈される．そのように設計すれば確率分布として構成をすることができる．もちろん，なぜこのようなことができるのかと言えば，確率分布の総和が1になるという性質をすべて規格化定数 $Z(\theta)$ に押し付けているからである．エネルギーの例としては，第2章で紹介したHopfieldモデルにおけるエネルギーなどがある．

なお，ここでの「エネルギー」は代謝コスト (metabolic cost) と完全に対応するものではなく，計算上取り入れられたものである．神経系における代謝コストはニューロンの発火活動やシナプス伝達に伴ってイオンポンプによる電位回復（たとえばNa/K ポンプ） などによって消費される実際のエネルギー（ATPなど）を指す \citep{jamadar2025metabolic}．例えば神経活動の大きさをエネルギー関数に組み込んだ場合，神経活動が大きいほど代謝コストは大きくなるため，EBMのエネルギーと代謝コストは関連付けることができる．

このエネルギーベースモデルで定義される分布は，指数型分布族と関連がある．指数型分布族をエネルギーベースモデルの式に変換するには，

$$
\to E(\mathbf{x}; \theta)
$$

とすればよい．要するに，指数の外側に $\mathbf{x}$ に依存する項を含まないようにしているのが特徴である．こうすることによる利点があり，それを享受しているのがスコアと呼ばれる概念である．エネルギーベースモデルにおいて，対数尤度の勾配 $\dfrac{\partial \ln p(\mathbf{x} \mid \theta)}{\partial \mathbf{x}}$ をスコアとよぶ．

$$
\frac{\partial \ln p(\mathbf{x} \mid \theta)}{\partial \mathbf{x}} =
- \frac{\partial \ln E(\mathbf{x}; \theta)}{\partial \mathbf{x}} - \frac{\partial \ln Z(\theta)}{\partial \mathbf{x}}=- \frac{\partial \ln E(\mathbf{x}; \theta)}{\partial \mathbf{x}}
$$

このようにスコアには正規化定数 (分配関数) は関与しない．








### 潜在変数モデル

潜在変数モデル (latent variable model) は，観測変数 $\mathbf{x}$，潜在変数 $\mathbf{z}$，およびパラメータ $\theta$ に関する同時分布として定式化される：

$$
p(\mathbf{x}, \mathbf{z}, \theta) 
= p(\mathbf{x} \mid \mathbf{z}, \theta)\, p(\mathbf{z} \mid \theta)\, p(\theta)
$$

推論や学習の目的は，観測データ $\mathbf{x}$ が与えられたときの潜在変数とパラメータの事後分布

$$
p(\mathbf{z}, \theta \mid \mathbf{x}) 
= \frac{p(\mathbf{x}, \mathbf{z}, \theta)}{p(\mathbf{x})}
$$

を求めることである。ここで分母の $p(\mathbf{x})$ は周辺尤度と呼ばれ，

$$
p(\mathbf{x})
= \iint p(\mathbf{x}, \mathbf{z}, \theta)\, \mathrm{d}\mathbf{z}\, \mathrm{d}\theta
$$

で与えられる。この積分は高次元で非線形な場合が多く，解析的に計算することは一般に困難である。そのため，潜在変数モデルの推論には **近似ベイズ推論** が必要となる。代表的な方法として，変分推論やマルコフ連鎖モンテカルロ法（MCMC）が挙げられる。

ここでは近似ベイズ推論の詳細には立ち入らず，より単純な近似として点推定に基づく解法を考える。とくに，ここでは前項で説明したMAP推定を用いることにする。MAP推定では周辺尤度の計算を避け，次の最適化問題を解けばよい：

$$
\begin{align}
\{\mathbf{z}^*, \theta^*\}
&= \arg\max_{\mathbf{z}, \theta} p(\mathbf{z}, \theta \mid \mathbf{x})\\
&= \arg\max_{\mathbf{z}, \theta} \bigl[\ln p(\mathbf{x} \mid \mathbf{z}, \theta) + \ln p(\mathbf{z}\mid \theta) + \ln p(\theta)\bigr]
\end{align}
$$

ここで

$$
\mathcal{L}(\mathbf{z}, \theta)
:= \ln p(\mathbf{x} \mid \mathbf{z}, \theta) 
 + \ln p(\mathbf{z}\mid \theta) 
 + \ln p(\theta)
$$

と定義すれば，勾配上昇法を用いて

$$
\Delta \mathbf{z} \propto \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}}\right)^\top, 
\quad 
\Delta \theta \propto \left(\frac{\partial \mathcal{L}}{\partial \theta}\right)^\top
$$

のように更新を行うことで最適解に近づけることができる（ここでは分子レイアウトを用いた）。

### 階層的潜在変数モデル

潜在変数は一層に限らず，複数の層を階層的に導入することができる。このようなモデルを階層的潜在変数モデル (hierarchical latent variable model) と呼ぶ。潜在変数を $\mathbf{z}_{1:L} := \{\mathbf{z}_i\}_{i=1}^L$，パラメータを $\boldsymbol{\theta}_{0:L} := \{\theta_i\}_{i=0}^L$ とすると，その同時分布は次式で表される：

$$
\begin{aligned}
p(\mathbf{x}, \mathbf{z}_{1:L}, \boldsymbol{\theta}_{0:L})
&= p(\mathbf{x}\mid \mathbf{z}_1,\theta_0)\,
   \prod_{i=1}^{L-1} p(\mathbf{z}_i \mid \mathbf{z}_{i+1}, \theta_i)\,
   p(\mathbf{z}_L \mid \theta_L)\,
   \prod_{i=0}^{L} p(\theta_i) \\
&= \prod_{i=0}^{L} p(\mathbf{z}_i \mid \mathbf{z}_{i+1}, \theta_i)\, p(\theta_i).
\end{aligned}
$$

第2行はすべての層を統一的に書き表すための形式であり，その際に

$$
\mathbf{z}_0 := \mathbf{x}, \quad \mathbf{z}_{L+1} := \varnothing
$$

という境界条件を導入する。ここでの $\varnothing$ は「変数が存在しないこと」を表す形式的な記号であり，集合論的な空集合とは区別される。なお，階層的潜在変数モデルは必ずしも直上の層のみに依存する必要はなく，スキップ結合を導入して「上位すべての層に依存する」と定式化することも可能である。


### 潜在変数モデル
ここまで，データに関しては観測可能な場合のみを考えてきた．
理想的には，外界のすべての変数が観測可能（fully visible）であることが望ましいが，実際には観測できない変数が存在することが多い．そのような隠れた構造を表現するために，潜在変数（latent variable） $\mathbf{z}$ を導入する．潜在変数に対して，$\mathbf{x}$ は観測変数 (observed variable) と呼ぶ．視覚系に対応させると，$\mathbf{x}$ は網膜像のような感覚入力を表し，$\mathbf{z}$ は物体のカテゴリ，三次元形状，照明条件など，より抽象的で高次の視覚的表現に相当すると解釈できる．ただし，この潜在変数は，実際の変数ではなく，推定する（脳内に表現される）ものであることに注意をしてほしい．

ベイズ線形回帰の場合と比較すると同じ文字で役割が異なっているのでややこしいが，生成対象は $y \to \mathbf{x}$ であり，パラメータは $\theta$ のままである．生成対象を説明する変数としては $\mathbf{x} \to \mathbf{z}$ だが，観測されている変数から，観測できていない隠れ変数に代わっているという点が，ベイズ線形回帰と異なっている．

ここから，パラメータ $\theta$ によって条件づけられる分布 $p(\cdot \mid \theta)$ を $p_\theta(\cdot)$ と略記することとする．潜在変数に基づいて観測データが生成されるという構造は，以下のような同時確率密度関数として定式化される：

$$
p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})\,p_\theta(\mathbf{z}) 
$$

ここで $p_\theta(\mathbf{z})$ は潜在変数の事前分布（prior），$p_\theta(\mathbf{x} \mid \mathbf{z})$ は尤度である．

生成モデルの学習や推論を行うためには，観測データ $\mathbf{x}$ に対する潜在変数 $\mathbf{z}$ の事後分布（posterior distribution）を求める必要がある．これはベイズの定理により次のように与えられる：

$$
p_\theta(\mathbf{z} \mid \mathbf{x}) = \frac{p_\theta(\mathbf{x} \mid \mathbf{z}) \cdot p_\theta(\mathbf{z})}{p_\theta(\mathbf{x})}
$$

ただし、分母の $p_\theta(\mathbf{x})$ は $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}$ によって計算される周辺尤度（またはエビデンス）である．

この事後分布 $p(\mathbf{z} \mid \mathbf{x}, \theta)$ は，観測された結果からその原因（$\mathbf{z}$）を推論 (inference) するための分布であり，逆モデル (inverse model), 符号化器 (encoder), あるいは認識モデル（recognition model）と呼ばれる．一方，$p_\theta(\mathbf{x} \mid \mathbf{z})$ は原因から結果を生成する順モデル (forward model), 復号器(decoder)，または生成モデル（generative model）である．

$\theta = \{\psi, \phi\}$

推論的知覚の枠組みにおいては，順モデルは上位から下位への下行性投射（top-down projection）に，逆モデルは下位から上位への上行性投射（bottom-up projection）に対応すると考えられる．

モデル全体 $p(\mathbf{x} \mid \theta)$ を指して生成モデルと呼ぶこともあるが、潜在変数モデルの文脈では、このデータ生成過程を担う部分を特に生成モデルと呼ぶ。

なお，ここで述べた「順モデル」，「逆モデル」という用語は，運動制御における内部モデル（internal model）の文脈で使われるものとは異なる概念であることに注意が必要である．

具体的な確率分布でモデルを設定してみよう．

このような潜在変数モデルの代表例が確率的主成分分析モデルであるが，ここでは扱わない．

### 階層ベイズモデル
生成モデルの表現力を高めるため，生成モデルを階層化することを考えよう．

本章では階層的生成モデルを導入し，それからスパース符号化，予測符号化について説明する．

$$
p_\theta (\mathbf{x}, \mathbf{z}^{1:L})=p_\theta (\mathbf{x} \mid \mathbf{z}^{1:L}) \prod_{\ell=1}^L p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1:L})
$$

隣接する層にしか影響しない場合は，

$$
p_\theta (\mathbf{x}, \mathbf{z}^{1:L})=p_\theta (\mathbf{x} \mid \mathbf{z}^1) \prod_{\ell=1}^L p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1})
$$

となる．

スパース符号化での議論をここでいれる．

$Z_\theta$ は規格化「定数」とついているが，実際には $\theta$ に依存する関数であるため，微分をしても消えない．

MAP推定までの展開を行う．

対数を取ると，

$$
\sum_\ell \ln p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1})
$$


### エネルギーベースモデル
潜在変数生成モデルにおいては、周辺尤度 $\ln p_\theta(\mathbf{x})$ の計算が困難である。この問題を解決するために、変分推論や ELBO（evidence lower bound）が導入されることもある。一方で、こうした明示的な確率密度を前提としない学習枠組みとして、エネルギーベースモデル（EBM）がある。

対数尤度 $\ln p(\mathbf{x})$ の勾配 $\nabla_\mathbf{x} \ln p(\mathbf{x})$ をスコアとよぶ．スコアには分配関数 (正規化定数) は関与しない．

$$
\nabla_\mathbf{x} \ln p_\theta(\mathbf{x}) = -\nabla_\mathbf{x}E_\theta (\mathbf{x}) - \nabla_\mathbf{x} \ln Z(\theta)=-\nabla_\mathbf{x}E_\theta (\mathbf{x}) 
$$




前節での潜在変数モデルの問題は，周辺尤度 $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}$ の計算が困難であるという点である．この問題を回避するために，エネルギーベースモデル（energy-based model; EBM）と呼ばれる確率モデルの枠組みを導入する \citep{lecun2006tutorial}．エネルギーベースモデルではネットワークの状態をスカラー値に変換するエネルギー関数 (あるいはコスト関数) を定義する．モデルのすべての変数をまとめて $\mathbf{s} = \{\mathbf{x}, \mathbf{z}\}$ とする.

このエネルギー関数は，ある状態 $\mathbf{x}'$ の「好ましさ」や「自然さ」を定量的に評価するものであり，エネルギーが小さいほどその状態がより実現しやすいと解釈される．

第2章で紹介したHopfieldモデルにおけるエネルギーが例である．

パラメータ$\theta$, （ポテンシャル）エネルギー関数 $E_{\theta}:\ \mathbb{R}^{n}\mathbb{\rightarrow R}$とすると，$\mathbf{s}$ の分布はGibbs-Boltzmann分布を用いて次のように表せる．

$$
\begin{equation}
p_{\theta}(\mathbf{s})\  = \frac{\exp\left(-E_{\theta}\left(\mathbf{s} \right)\right)}{Z_{\theta}},\quad Z_{\theta} = \int_{\mathbf{s}'\in \mathcal{S}} \exp(-E_{\theta}\left( \mathbf{s}' \right))d\mathbf{s}'
\end{equation}
$$

ただし，$Z_{\theta}$は規格化定数あるいは分配関数である．

なお，ここでの「エネルギー」は代謝コスト (metabolic cost) と完全に対応するものではなく，計算上取り入れられたものである．神経系における代謝コストはニューロンの発火活動やシナプス伝達に伴ってイオンポンプによる電位回復（たとえばNa/K ポンプ） などによって消費される実際のエネルギー（ATPなど）を指す \citep{jamadar2025metabolic}．例えば神経活動の大きさをエネルギー関数に組み込んだ場合，神経活動が大きいほど代謝コストは大きくなるため，EBMのエネルギーと代謝コストは関連付けることができる．

## 階層的生成モデル
生成モデルの表現力を高めるため，生成モデルを階層化することを考えよう．

本章では階層的生成モデルを導入し，それからスパース符号化，予測符号化について説明する．

$$
p_\theta (\mathbf{x}, \mathbf{z}^{1:L})=p_\theta (\mathbf{x} \mid \mathbf{z}^{1:L}) \prod_{\ell=1}^L p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1:L})
$$

隣接する層にしか影響しない場合は，

$$
p_\theta (\mathbf{x}, \mathbf{z}^{1:L})=p_\theta (\mathbf{x} \mid \mathbf{z}^1) \prod_{\ell=1}^L p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1})
$$

となる．

スパース符号化での議論をここでいれる．

$Z_\theta$ は規格化「定数」とついているが，実際には $\theta$ に依存する関数であるため，微分をしても消えない．

MAP推定までの展開を行う．

対数を取ると，

$$
\sum_\ell \ln p_\theta(\mathbf{z}^\ell \mid \mathbf{z}^{\ell+1})
$$

生成モデル→エネルギーベースモデル→MAP推定→スパース→階層的生成モデル?
→予測符号化→不確実性の導入→ベイズ脳…

### 最大事後確率推定

### スパース符号化モデル
スパース符号化モデル (Sparse coding model) \citep{`Olshausen1996-xe`, `Olshausen1997-qu`}はV1のニューロンの応答特性を説明する線形生成モデル (linear generative model)である．まず，画像パッチ $\mathbf{x}$ が基底関数(basis function) $\mathbf{\Phi} = [\phi_j]$ のノイズを含む線形和で表されるとする (係数は $\mathbf{r}=[r_j]$ とする)．

$$
\begin{equation}
\mathbf{x} = \sum_j r_j \phi_j +\boldsymbol{\epsilon}= \mathbf{\Phi} \mathbf{r}+ \boldsymbol{\epsilon}
\end{equation}
$$

ただし，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ である．このモデルを神経ネットワークのモデルと考えると， $\mathbf{\Phi}$ は重み行列，係数 $\mathbf{r}$ は入力よりも高次の神経細胞の活動度を表していると解釈できる．ただし，$r_j$ は負の値も取るので単純に発火率と捉えられないのはこのモデルの欠点である．

Sparse codingでは神経活動 $\mathbf{r}$ が潜在変数の推定量を表現しているという仮定の下，少数の基底で画像 (や目的変数)を表すことを目的とする．要は上式において，ほとんどが0で，一部だけ0以外の値を取るという疎 (=sparse)な係数$\mathbf{r}$を求めたい．

#### 確率的モデルの記述
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

#### 事前分布の設定
事前分布$p(\mathbf{r})$としては，0においてピークがあり，裾の重い(heavy tail)を持つsparse distributionあるいは super-Gaussian distribution (Laplace分布やCauchy分布などGaussian分布よりもkurtoticな分布) を用いるのが良い．このような分布では，$\mathbf{r}$の各要素$r_i$はほとんど0に等しく，ある入力に対しては大きな値を取る．$p(\mathbf{r})$は一般化して次のように表記する．

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

#### 目的関数の設定と最適化
最適な生成モデルを得るために，入力される画像パッチの真の分布 $p_{data}(\mathbf{x})$と$\mathbf{x}$の生成モデル $p(\mathbf{x}|\mathbf{\Phi})$を近づける．このために，2つの分布のKullback-Leibler ダイバージェンス $D_{\text{KL}}\left(p_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$を最小化したい．しかし，真の分布は得られないので，経験分布 

$$
\begin{equation}
\hat{p}_{data}(\mathbf{x}):=\frac{1}{N}\sum_{i=1}^N \delta(\mathbf{x}-\mathbf{x}_i)
\end{equation}
$$

を近似として用いる ($\delta(\cdot)$ はDiracのデルタ関数である)．ゆえに$D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$を最小化する．

$$
\begin{align}
D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)&=\int \hat{p}_{data}(\mathbf{x}) \ln \frac{\hat{p}_{data}(\mathbf{x})}{p(\mathbf{x}|\mathbf{\Phi})} d\mathbf{x}\\
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

これは $\mathbf{r}$ について MAP推定 (maximum a posteriori estimation)を行うことに等しい．次に$\hat{\mathbf{r}}$を用いて

$$
\begin{equation}
\mathbf{\Phi}^*=\text{arg}\min_{\mathbf{\Phi}} \sum_{i=1}^N E(\mathbf{x}_i, \hat{\mathbf{r}}_i|\mathbf{\Phi})\ \left(= \text{arg}\max_{\mathbf{\Phi}} \prod_{i=1}^N p(\mathbf{x}_i|\hat{\mathbf{r}}_i, \mathbf{\Phi})\right)
\end{equation}
$$

とすることにより，$\mathbf{\Phi}$を最適化する．こちらは $\mathbf{\Phi}$ について 最尤推定 (maximum likelihood estimation)を行うことに等しい．

#### 局所競合則
局所競合則 (Locally competitive algorithm; LCA)．

$\mathbf{r}$の勾配法による更新則は，$E$の微分により次のように得られる．

$$
\begin{equation}
\frac{d \mathbf{r}}{dt}= -\frac{\eta_\mathbf{r}}{2}\frac{\partial E}{\partial \mathbf{r}}=\eta_\mathbf{r} \cdot\left[\mathbf{\Phi}^\top (\mathbf{x}-\mathbf{\Phi}\mathbf{r})- \frac{\lambda}{2}S'\left(\mathbf{r}\right)\right]
\end{equation}
$$

ただし，$\eta_{\mathbf{r}}$は学習率である．この式により$\mathbf{r}$が収束するまで最適化するが，単なる勾配法ではなく，\citep{`Olshausen1996-xe`では共役勾配法 (conjugate gradient method)を用いている．しかし，共役勾配法は実装が煩雑で非効率であるため，より効率的かつ生理学的な妥当性の高い学習法として，LCA  (locally competitive algorithm)が提案されている \citep{`Rozell2008-wp`．LCAは側抑制 (local competition, lateral inhibition)と閾値関数 (thresholding function)を用いる更新則である．LCAによる更新を行うRNNは通常のRNNとは異なり，コスト関数(またはエネルギー関数)を最小化する動的システムである．このような機構はHopfield networkで用いられているために，OlshausenはHopfield trickと呼んでいる．

##### 軟判定閾値関数を用いる場合 (ISTA)
$S(x)=|x|$とした場合の閾値関数を用いる手法としてISTA(Iterative Shrinkage Thresholding Algorithm)がある．ISTAはL1-norm正則化項に対する近接勾配法で，要はLasso回帰に用いる勾配法である．

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

ここで$\Theta_\lambda(\cdot)$は軟判定閾値関数 (Soft thresholding function)と呼ばれ，次式で表される．

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

#### 重み行列の更新則
$\mathbf{r}$が収束したら勾配法により$\mathbf{\Phi}$を更新する．

$$
\begin{equation}
\Delta \phi_i(\boldsymbol{x}) = -\eta \frac{\partial E}{\partial \mathbf{\Phi}}=\eta\cdot\left[\left(\mathbf{x}-\mathbf{\Phi}\mathbf{r}\right)\mathbf{r}^\top\right]
\end{equation}
$$

#### スパース符号化モデルの実装
ネットワークは入力層を含め2層の単純な構造である．今回は，入力はランダムに切り出した16×16 (＝256)の画像パッチとし，これを入力層の256個のニューロンが受け取るとする．入力層のニューロンは次層の100個のニューロンに投射するとする．100個のニューロンが入力をSparseに符号化するようにその活動および重み行列を最適化する．

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


### ボルツマンマシン
エネルギーベースモデルの具体例としてボルツマンマシン (Boltzmann machine) を取り上げる．

定義した任意の $E_{\theta}(\mathbf{x}')$ を神経活動 $\mathbf{x}'$ やパラメータ $\theta$ で微分することで，推論と学習ダイナミクスを定義できる．

Hopfieldモデルの各ユニットが取りうる活動を確率的にしたモデルがBoltzmannマシンである．

Boltzmannマシンは，確率的生成モデルの一例として，その状態の確率分布をエネルギー関数に基づいて定義するモデルである．ここで，システムの状態は $\mathbf{s} = (s_1, s_2, \ldots, s_N)$ という2値のユニットの組で表され，各 $s_i$ は0または1の値を取る．Boltzmannマシンでは，各状態のエネルギーは以下の式によって与えられる：

$$
E(\mathbf{s}) = -\sum_{i} b_i s_i - \sum_{i < j} W_{ij} s_i s_j
$$

ここで，$b_i$ は各ユニットに対応するバイアス項，$W_{ij}$ はユニット $i$ と $j$ の間の対称的な結合重みを表す．状態 \(\mathbf{s}\) が出現する確率は，エネルギー関数に基づいてボルツマン分布として定義され，以下のように記述される：

$$
P(\mathbf{s}) = \frac{1}{Z} \exp\left(-E(\mathbf{s})\right)
$$

ここで，正規化定数 $Z$（分配関数）は全状態にわたる和で定義される：

$$
Z = \sum_{\mathbf{s}} \exp\left(-E(\mathbf{s})\right)
$$

このモデルは，全ユニット間に結合が存在するため，内部の依存関係が複雑になり，特に学習の際にパラメータ更新のための勾配計算が指数的な計算量を要するという難点がある．

Boltzmannマシンにおける学習および推論の主要な困難さは，その計算に内在する分配関数 $Z$ の評価に起因する．Boltzmannマシンでは，エネルギー関数

$$
E(\mathbf{s}) = -\sum_{i} b_i s_i - \sum_{i<j} W_{ij} s_i s_j
$$

に従い，状態 \(\mathbf{s}\) の確率分布は

$$
P(\mathbf{s}) = \frac{1}{Z} \exp\left(-E(\mathbf{s})\right)
$$

と定義されるが，ここで正規化定数 $Z$ は

$$
Z = \sum_{\mathbf{s}} \exp\left(-E(\mathbf{s})\right)
$$

と全可能状態 \(\mathbf{s}\) にわたる和として計算されなければならない．各ユニットが2値の確率変数である場合，全状態数は \(2^N\) となるため，ネットワークの規模が大きくなるとこの和は指数関数的に増大し，厳密な計算が事実上不可能となる．

さらに，学習に必要なパラメータ更新のための勾配計算でも，この正規化定数 $Z$ に依存する項が現れる．具体的には，尤度関数の勾配として，例えば重み $W_{ij}$ に関しては

$$
\frac{\partial \ln P(\mathbf{s})}{\partial W_{ij}} = \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}
$$

と表されるが，ここで \(\langle s_i s_j \rangle_{\text{model}}\) はモデル分布における期待値であり，これは

$$
\langle s_i s_j \rangle_{\text{model}} = \sum_{\mathbf{s}} s_i s_j \, P(\mathbf{s})
$$

として計算される必要がある．しかし，前述のように $P(\mathbf{s})$ の計算には $Z$ の求積が不可欠であり，これもまた指数的な計算量を要するため，直接計算することは困難である．

このような計算の困難性は，統計物理における分配関数の計算問題と同様に，組み合わせ爆発（combinatorial explosion）の問題として知られ，計算複雑性理論では #P困難（#P-complete）であると指摘される．これに対処するため，実際の学習ではサンプルに基づく近似手法（モンテカルロ法，ギブスサンプリングなど）や，特定の近似アルゴリズム（コントラスト・ダイバージェンスなど）が利用される．しかしこれら近似手法にも収束の問題や精度の限界が存在するため，一般的なBoltzmannマシンは大規模な問題に対して直接適用するのが難しく，その計算効率の改善は依然として重要な研究課題である．

この問題点を解消するために考案されたのが，制限Boltzmannマシン（Restricted Boltzmann Machine: RBM）である．RBMでは，ネットワークを二層構造に限定し，可視層 $\mathbf{v}$ と隠れ層 $\mathbf{h}$ のみを用いる．ここで，可視ユニット $v_i$ は入力データを表し，隠れユニット $h_j$ はデータの特徴（潜在変数）を表す．RBMのエネルギー関数は次の形で定義される：

$$
E(\mathbf{v}, \mathbf{h}) = -\sum_{i} a_i v_i - \sum_{j} b_j h_j - \sum_{i, j} v_i W_{ij} h_j
$$

このとき，$a_i$ は可視ユニットのバイアス，$b_j$ は隠れユニットのバイアス，そして $W_{ij}$ は可視ユニットと隠れユニット間の結合重みである．RBMでは，同一層内のユニット間の結合（例えば，可視層同士，隠れ層同士）は存在しないため，モデル内の条件付き独立性が成立する．具体的には，隠れ層の各ユニットは可視層が与えられた条件下で独立に分布し，その条件付き確率は次の式で表される：

$$
P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i} v_i W_{ij}\right)
$$

また，可視層の各ユニットに関しても同様に，

$$
P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_{j} h_j W_{ij}\right)
$$

と記述される．ここで，\(\sigma(x) = \frac{1}{1+\exp(-x)}\) はシグモイド関数である．これらの性質により，RBMは効率的なギブスサンプリングが可能となり，コントラスト・ダイバージェンス（Contrastive Divergence, CD）と呼ばれる近似的な学習アルゴリズムが用いられて実用的な学習が可能となる．

このようにして，Boltzmannマシンは複雑な結合を持つモデルとして理論的な基盤を提供する一方，RBMはその結合を制限することにより計算の効率化を実現している．これらのモデルは，特にディープラーニングにおける事前学習や特徴抽出の文脈で重要な役割を果たし，画像認識や信号処理など幅広い応用がなされている．

#### 制限ボルツマンマシン
制限ボルツマンマシン (Restricted Boltzmann machine) 

離散の観測変数(visible variable) $\mathbf{v}$, 潜在変数(hidden variable) $\mathbf{h}$とする．各ユニットの値は$\{0, 1\}$の2値 (binary)である．

エネルギー関数を

$$
\begin{equation}
E_\theta(\mathbf{v}, \mathbf{h})=-\mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h} + \mathbf{v}^\top \mathbf{W} \mathbf{h}
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

#### 訓練データで学習
$$
\begin{align}
p_\theta(\mathbf{h}|\mathbf{v})&=\prod_i p_\theta(h_i=1|\mathbf{v})=\prod_i \sigma(c_i + W_i \mathbf{v})\\
p_\theta(\mathbf{v}|\mathbf{h})&=\prod_j p_\theta(v_j=1|\mathbf{h})=\prod_j \sigma(b_j + W_j^\top \mathbf{h})
\end{align}
$$

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

## ガウス尺度混合モデル
ガウス尺度混合 (Gaussian scale mixture; GSM) モデルは確率的生成モデルの一種である{cite:p}`Wainwright1999-cl`{cite:p}`Orban2016-tm`．GSMモデルでは入力を次式で予測する：

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
前節で実装したMCMCを興奮性・抑制性神経回路 (excitatory-inhibitory (E-I) network) で実装する．HMCとLMCの両方を神経回路で実装する．ハミルトニアンを用いる場合，一般化座標$\mathbf{q}$を興奮性神経細胞の活動$\mathbf{u}$, 一般化運動量$\mathbf{p}$を抑制性神経細胞の活動$\mathbf{v}$に対応させる．$\mathbf{u,\ v}$は同じ次元のベクトルとする．$\mathbf{u}, \mathbf{v}$の時間発展はハミルトニアン$\mathcal{H}$を導入して

$$
\begin{equation}
\tau\frac{d\mathbf{u}}{dt} = \frac{\partial \mathcal{H}}{\partial\mathbf{v}},\quad\tau\frac{d\mathbf{v}}{dt} = - \frac{\partial \mathcal{H}}{\partial\mathbf{u}}
\end{equation}
$$

と書ける．一般的には$\mathcal{H}(\mathbf{u}, \mathbf{v}) = E\left( \mathbf{u} \right) + \frac{1}{2}\mathbf{v}^{\top}\mathbf{v}$であり，$p\left( \mathbf{u},\ \mathbf{v} \right) \propto \exp( - \mathcal{H}(\mathbf{u,v}))$である．力学的エネルギーを保つ運動は，対数同時分布における等値線上の運動と同じである．

\citep{Aitchison2016-xu}では

$$
\begin{equation}
\mathcal{H}(\mathbf{u}, \mathbf{v}) = \ln p \left(\mathbf{u}, \mathbf{v} \right) + \textrm{Const.} = \ln p \left(\mathbf{v} \middle| \mathbf{u} \right) + \ln p\left(\mathbf{u} \right) + \textrm{Const.}
\end{equation}
$$

とし，$p\left( \mathbf{v} \middle| \mathbf{u} \right)\mathcal{= N}\left( \mathbf{v};\mathbf{Bu},\ \mathbf{M}^{- 1} \right),\ \ p\left( \mathbf{u} \right) = \mathcal{N\ (}\mathbf{0},\ \mathbf{C}^{- 1})$としている．この場合，

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{v}} = \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{v}} = \ \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}}\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial \mathcal{H}}{\partial\mathbf{u}} = - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} = \  - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}}
\end{align}
$$
となる．このままでは等値線上を運動することになるので，Langevinダイナミクスを付け加える．

$$
\begin{align}
\frac{d\mathbf{u}}{dt} &= \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u},\ \mathbf{v} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
\frac{d\mathbf{v}}{dt} &= - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{u},\mathbf{v} \right)}}{\partial\mathbf{v}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta\\
&= - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} + \frac{1}{\tau_{L}}\frac{\partial\ln{p\left( \mathbf{v|u} \right)}}{\partial\mathbf{v}} - \frac{1}{\tau}\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} + \sqrt{\frac{2}{\tau_{L}}}\ d\eta
\end{align}
$$

となる．それぞれの項は

$$
\begin{align}
\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{v}} &= \mathbf{B}^{\top}\mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\ln{p\left( \mathbf{v} \middle| \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{M}\left( \mathbf{Bu} - \mathbf{v} \right)\\
\frac{\partial\ln{p\left( \mathbf{u} \right)}}{\partial\mathbf{u}} &= - \mathbf{Cu}
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

となり，$\mathbf{u}\mathbf{,\ v}$と定行列およびノイズに依存してサンプリングダイナミクスを記述できる．長々と式変形を書いたが，重要なのは興奮性・抑制性という2種類の細胞群の相互作用により生み出された振動を用いてサンプリングにおける自己相関を下げることができるという点である．

簡単のため，前項で用いた入力刺激のうち，最も$z$が大きいサンプルのみを使用する．

Hamiltonianネットワークは自己相関を振動により低下させることで，効率の良いサンプリングを実現している．ToDo: 普通にMCMCやる場合も自己相関は確認したほうがいいという話をどこかに書く．

推定された事後分布を特定の神経細胞のペアについて確認する．

Hamiltonianネットワークの方が安定して事後分布を推定することができている．ToDo: 以下の記述．ここでは重みを設定したが， {cite:p}`Echeveste2020-sh`ではRNNにBPTTで重みを学習させている．動的な入力に対するサンプリング {cite:p}`Berkes2011-xj`．burn-inがなくなり効率良くサンプリングできる．

## Spikingニューラルネットワークにおけるサンプリング
前項で挙げた例は発火率モデルであったが，SNNにおいてサンプリングを実行する機構自体は考案されている．ToDo: 以下の記述．{cite:p}`Buesing2011-dm`{cite:p}`Masset2022-wh`{cite:p}`Zhang2022-bl`

## シナプスサンプリング
ここまでシナプス結合強度は変化せず，神経活動の変動によりサンプリングを行うというモデルについて考えてきた．一方で，シナプス結合強度自体が短時間で変動することによりベイズ推論を実行するというモデルがあり，シナプスサンプリング(synaptic sampling) と呼ばれる．ToDo: 以下の記述．{cite:p}`Kappel2015-kq`{cite:p}`Aitchison2021-wo`

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

## 変分推論
近似分布 $q$ を用意する．近似分布族を $\mathcal{Q}$ とすると，$q \in \mathcal{Q}$ において，最適な分布を探すこととなる．