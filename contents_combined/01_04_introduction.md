## 学習と予測の基礎
本書のテーマの1つとして「学習」が挙げられる．
神経科学における「学習」と機械学習における「学習」はやや異なるが，ここで両者における学習を定義しておく．

神経科学の学習は

共通する点として，過去の経験に基づいて，将来の行動や出力を改善するためにシステムを変化させる，という点で共通している．システムの

システムのパラメータが変化する

神経科学：シナプス強度（重み）が変化する。

機械学習：ネットワークの重みやバイアスなどのパラメータが更新される。

異なる点として，

神経科学のモデルに機械学習

### 学習の基礎的概念
#### モデルと学習・予測
**数理モデル**とは、実世界の現象、システム、または過程を数学的対象を用いて表現する枠組みである。モデルは変数とパラメータから構成され、これらの間の関係性は数式、方程式、確率分布、または集合論的記述によって定式化される。数理モデルは現象の分析、予測、制御、最適化などに用いられる。

**機械学習** (machine learning) における**モデル** (model) とは，2つの集合$\mathcal{X}, \mathcal{Y}$ を仮定したとき，入力 $x \in \mathcal{X}$ を出力 $y \in \mathcal{Y}$ に対応づける関数 $f: \mathcal{X} \to \mathcal{Y}$ または条件付き確率分布 $p(y\mid x)$ を指す．モデルは内部に学習可能なパラメータ (parameter) $\theta$を持ち，このパラメータを調整することによって入力と出力の関係を最適化する．

機械学習での**学習** (learning) とは，観測データに基づき，$y = f(x; \theta)$ あるいは $p(y\mid x; \theta)$ をよりよく近似するパラメータ $\theta$ を求める過程である．モデルを学習させることを**訓練** (training) と呼ぶ．このとき，データ集合 $D = \{(x_i, y_i)\}$ を用いて，目的関数（損失関数）を最小化または尤度を最大化することでパラメータ $\theta$ を更新する．学習によって得られた最適なパラメータ $\theta^*$ を用い，未知の入力 $x$ に対して出力 $\hat{y}$ を推定することを**予測** (prediction) と呼ぶ．推定値 $\hat{y}$ の取得には，$p(y \mid x;\theta^*)$ からのサンプリング，$\mathrm{argmax}_y\ p(y \mid x; \theta^*)$ の計算，あるいは期待値 $\mathbb{E}[y \mid x; \theta^*]$ の計算などが用いられる．

$y$ が既知の場合，データ集合 $D=\{(x,y)\}$ は**教師付きデータ** (labeled data) と呼ばれ，この対応関係を学習する過程を**教師あり学習** (supervised learning) という．一方，$y$ が未知で $x$ のみが与えられる場合，$D=\{x\}$ は**ラベルなしデータ**と呼ばれ，その潜在構造や分布を推定する過程を**教師なし学習** (unsupervised learning) という．教師なし学習の典型例にはクラスタリングや次元削減がある．また，ラベルありデータとラベルなしデータの両方を用いる学習を**半教師あり学習** (semi-supervised learning)，ラベルなしデータから自己生成したラベルを用いて教師あり学習の形式で学習する手法を**自己教師あり学習** (self-supervised learning) という．さらに，機械学習の重要な分野のひとつに**強化学習** (reinforcement learning) がある．強化学習では，環境と相互作用しながら行動を選択するエージェントを仮定し，逐次的な意思決定過程において，累積報酬を最大化するための方策を学習する．詳細は第11章で述べる．

#### 回帰と分類
機械学習の課題は、大きく回帰 (regression) と分類 (classification) に分けられる。回帰とは、入力$x$に対して連続的な出力$y \in \mathbb{R}$または$\mathbb{R}^d$を予測する問題を指す。典型的な例としては、住宅価格の予測や気温の予測などが挙げられる。これに対し分類とは、入力$x$に対して離散的なクラスラベル$y \in {1, \dots, K}$を予測する問題を指し、例えば画像から猫・犬・鳥を識別する場合がこれに相当する。回帰と分類はいずれも入力と出力の関係を学習するが、出力空間の性質（連続 vs 離散）により目的関数やモデルの設計が異なる。なお、回帰問題においても、出力を閾値によって離散化することで分類問題に転換することが可能であり、両者の境界は必ずしも絶対的ではない。

#### 識別モデルと生成モデル
入力と出力の関係を学習する方法には、識別モデル (discriminative model) と生成モデル (generative model) という分類がある。識別モデルとは、入力$x$が与えられたときに出力$y$を直接推定する条件付き確率分布$p(y|x)$を学習するモデルを指す。これに対し生成モデルとは、入力と出力の同時確率分布$p(x,y)$、あるいは入力$x$の分布$p(x)$と出力$y$に条件づけた生成分布$p(x|y)$を学習するモデルを指す。生成モデルを用いることで、データのサンプリングや異常検知、データ補完など幅広い応用が可能となる。識別モデルは予測性能に特化する一方、生成モデルはデータ分布そのものの理解と生成に重点を置くという違いがある。

#### オフライン学習とオンライン学習
学習の方法は、利用可能なデータと学習過程の違いに応じて、**オフライン学習** (offline learning) と**オンライン学習** (online learning) に分類できる。オフライン学習では、すべての訓練データ $D=\{(x_i, y_i)\}$ があらかじめ揃っており、この固定されたデータに対して繰り返し学習を行う。オフライン学習は一括学習あるいはバッチ学習 (batch learning) とも呼ばれ、典型的な深層学習や統計的推論はこの枠組みに基づいている。

これに対してオンライン学習では、データは逐次的に到着し、到着するたびにモデルのパラメータを即座に更新する。このためオンライン学習は逐次学習 (sequential learning) とも呼ばれ、環境が変動する場合やリアルタイム処理が求められる場合に有効である。

このオフライン学習とオンライン学習の違いは、学習の**相** (phase) の構造にも現れる。オフライン学習は、モデルから推定値を取得する**推論相** (inference phase) と、パラメータを更新する**訓練相** (training phase) の二つの相を明確に区別して持つ。例えばニューラルネットワークにおいては、順伝播が推論相、逆伝播とパラメータ更新が訓練相に対応する。一方、オンライン学習は基本的に**単相** (single phase) であり、推論と更新が逐次的かつ並行して行われるため、明示的に分離された相を持たない。

生物学的な脳において、オフライン学習とオンライン学習のいずれが行われているかについては、現代の神経科学においても明確な結論は得られていない。ただし、一般的な知見として、睡眠中に記憶再生を通じて運動記憶が固定されるオフライン学習過程（Shadmehr & Brashers-Krug, 1997; Albouy et al., 2008）と、覚醒中の感覚入力に応じて逐次的な誤差修正が行われるオンライン学習過程（Shadmehr et al., 2010）がともに存在すると考えられている。

実際、最近のメタ分析研究（Byczynski et al., 2025）は、運動学習課題における脳活動のパターンを体系的に比較することにより、試行中に進行する**オンライン学習**と試行間に進行する**オフライン学習**がいずれも明確に存在することを示した。この研究では、両者に共通する領域（補足運動野や体性感覚野）と、それぞれに特有な領域が同定され、運動獲得と記憶固定という異なるプロセスにおいて異なる神経基盤が働いていることが示唆された。これにより、脳におけるオンライン・オフライン学習の並立モデルが一層支持された。

https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00457/127405/Neural-signatures-of-online-and-offline-motor

### 線形回帰
**線形回帰モデル**（linear regression）は、与えられた説明変数（explanatory variable）$\mathbf{x}$に基づいて、目的変数（objective variable）$y$を線形に予測することを目的とする。

説明変数の次元が$p$であるとき、線形回帰モデルは次のように表される：

$$
\begin{equation}
y = w_0 + w_1x_1 + \cdots + w_px_p + \varepsilon = w_0 + \sum_{j=1}^p w_j x_j + \varepsilon
\end{equation}
$$

ここで$w_0$は切片（バイアス項）、$w_1, \dots, w_p$は各説明変数に対する重み、$\varepsilon$は誤差項を表す。$p = 1$の場合を**単回帰**（*simple regression*）、$p > 1$の場合を**重回帰**（*multiple regression*）と呼ぶ。

#### 回帰モデルの行列表現

$n$個の観測データからなるデータセット$\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$を考える。ここで$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_p^{(i)}]^\top \in \mathbb{R}^p$は$i$番目の説明変数ベクトル、$y^{(i)} \in \mathbb{R}$は対応する目的変数の値である。なお、添字$(i)$は観測値を表し、添字のない$x_j, w_j$などはモデル内の変数を指すことに注意する。

このとき、モデル全体を行列の形で次のように記述できる：

$$
\begin{equation}
\mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix} \in \mathbb{R}^n,\quad
\mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_p^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_p^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & x_2^{(n)} & \cdots & x_p^{(n)} \end{bmatrix} \in \mathbb{R}^{n \times (p+1)},\quad
\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_p \end{bmatrix} \in \mathbb{R}^{p+1}
\end{equation}
$$

これにより、回帰モデルは次のように簡潔に表される：

$$
\begin{equation}
\mathbf{y} = \mathbf{X} \mathbf{w} + \boldsymbol{\varepsilon}
\end{equation}
$$

ここで$\mathbf{X}$は**計画行列**（design matrix）、$\boldsymbol{\varepsilon}$は誤差ベクトルである。特に、$\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$、すなわち各誤差成分が独立な平均0・分散$\sigma^2$の正規分布に従うと仮定すれば、$\mathbf{y} \sim \mathcal{N}(\mathbf{X} \mathbf{w}, \sigma^2 \mathbf{I})$という確率モデルが得られる。

#### 最小二乗法
**最小二乗法**（ordinary least squares, OLS）では、観測値$\mathbf{y}$と予測値$\mathbf{Xw}$との差（残差）を最小にするようにパラメータ$\mathbf{w}$を推定する。残差ベクトル$\boldsymbol{\delta} = \mathbf{y} - \mathbf{Xw}$に対し、目的関数$\mathcal{L}(\mathbf{w})$は次のように定義される：

$$
\begin{equation}
\mathcal{L}(\mathbf{w}) \coloneqq \|\boldsymbol{\delta}\|^2 = \boldsymbol{\delta}^\top \boldsymbol{\delta}
\end{equation}
$$

この$\mathcal{L}(\mathbf{w})$を最小化する$\mathbf{w}$を求めることで、最適な重み$\hat{\mathbf{w}}$を得る。最適解の推定は主に**正規方程式**（normal equation）あるいは**勾配法**（gradient descent）によって行うことができる．いずれの手法でも，目的関数$\mathcal{L}(\mathbf{w})$の$\mathbf{w}$について微分、すなわち勾配 (gradient)$\nabla \mathcal{L}(\mathbf{w})$が必要となる．$\nabla \mathcal{L}(\mathbf{w})$は以下のように計算できる：

$$
\begin{align}
\nabla \mathcal{L}(\mathbf{w})
&= \frac{\partial}{\partial \mathbf{w}}\left[(\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} - \mathbf{Xw}) \right] \\
&= \frac{\partial}{\partial \mathbf{w}} \left( \mathbf{y}^\top \mathbf{y} - 2 \mathbf{y}^\top \mathbf{Xw} + \mathbf{w}^\top \mathbf{X}^\top \mathbf{Xw} \right) \\
&= -2 \mathbf{X}^\top \mathbf{y} + 2 \mathbf{X}^\top \mathbf{Xw}\\
&= -2\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw})
\end{align}
$$

##### 正規方程式による解析解
目的関数の勾配について$\nabla \mathcal{L}(\mathbf{w})=0$となる解を$\hat{\mathbf{w}}$とすると，次の**正規方程式**（normal equation）が得られる：

$$
\begin{equation}
\mathbf{X}^\top \mathbf{X} \hat{\mathbf{w}} = \mathbf{X}^\top \mathbf{y}
\end{equation}
$$

この方程式を解くことで、パラメータの推定値$\hat{\mathbf{w}}$は次のように求まる：

$$
\begin{equation}
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\end{equation}
$$

なお、$A^+ \coloneqq (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$は$\mathbf{X}$の**Moore–Penrose 擬似逆行列**（pseudoinverse）と呼ばれ、この表現を用いると$\hat{\mathbf{w}} = A^+ \mathbf{y}$と簡潔に記述できる。

##### 勾配法による数値的推定
最小二乗法に基づくパラメータ推定は、数値的には**勾配法**（gradient descent）によっても実現できる。 目的関数の勾配$\nabla \mathcal{L}(\mathbf{w})$を用いると、更新式は次のように与えられる：

$$
\begin{align}
\Delta \mathbf{w} \propto - \nabla \mathcal{L}(\mathbf{w})= 2\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw})\\
\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \frac{1}{n} \mathbf{X}^\top (\mathbf{y} - \mathbf{Xw})
\end{align}
$$

ここで$\alpha$は**学習率**（learning rate）と呼ばれるハイパーパラメータである。

### リッジ回帰
線形回帰においては、説明変数が高次元である場合や、多重共線性（説明変数間の相関）が存在する場合などに、最小二乗法による推定が不安定になることがある。これに対処する手法として、**L2 正則化**を加えた**リッジ回帰**（ridge regression）が用いられる。

リッジ回帰では、目的関数にパラメータの二乗ノルムを加えた正則化項を導入することにより、モデルの複雑さを抑制し、過学習の防止や推定の安定化を図る。具体的には、次のような正則化付き目的関数$\mathcal{L}_\lambda(\mathbf{w})$を最小化する：

$$
\begin{equation}
\mathcal{L}_\lambda(\mathbf{w}) = \|\mathbf{y} - \mathbf{Xw}\|^2 + \lambda \|\mathbf{w}\|^2 = (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} - \mathbf{Xw}) + \lambda \mathbf{w}^\top \mathbf{w},
\end{equation}
$$

ここで$\lambda \geq 0$は**正則化係数**（regularization parameter）であり、モデルのあてはまりと複雑さのトレードオフを制御する。なお通常、$w_0$（切片）には正則化を加えないことが多いため、必要に応じて$\mathbf{w}$の対象を$[w_1, \dots, w_p]^\top$に限定する処理を行う。

#### 正規方程式による解
L2 正則化付きの目的関数を$\mathbf{w}$で微分して0に等しいとおくと、次のような修正された正規方程式が得られる：

$$
\begin{equation}
(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \hat{\mathbf{w}} = \mathbf{X}^\top \mathbf{y},
\end{equation}
$$

ここで$\mathbf{I} \in \mathbb{R}^{(p+1)\times(p+1)}$は単位行列である．ただし、$w_0$を正則化対象から除く場合、$\lambda \mathbf{I}$の最初の対角成分をゼロにすることで対処する。この式を解くと、リッジ回帰におけるパラメータの推定値は次のように求まる：

$$
\begin{equation}
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}.
\end{equation}
$$

この推定式は、$\mathbf{X}^\top \mathbf{X}$が特異（非正則）である場合でも、$\lambda > 0$により逆行列の存在が保証される点で、最小二乗法に比べて数値的に安定であるという利点がある。

#### 勾配法による推定

リッジ回帰に対しても勾配法を適用できる。目的関数$\mathcal{L}_\lambda(\mathbf{w})$の勾配は次のように求まる：

$$
\begin{equation}
\nabla \mathcal{L}_\lambda(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{Xw}) + 2\lambda \mathbf{w},
\end{equation}
$$

これに基づいて、更新式は以下のように与えられる：

$$
\begin{equation}
\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \left( \frac{1}{n} \mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) - \lambda \mathbf{w} \right),
\end{equation}
$$

または$\alpha$を調整することで、$\lambda$を勾配更新の一部として組み込む方法もある。いずれにしても、正則化項によって重みの更新が抑制されることで、過学習を防ぐ効果が得られる。

### ロジスティック回帰
本節では、非線形回帰の一種である**ロジスティック回帰** (logistic regression) について取り扱う。

ロジスティック回帰は、入力$\mathbf{x} \in \mathbb{R}^p$に対して出力$y \in \{0, 1\}$を予測する**確率的な分類モデル**である。出力は事後確率$\Pr(y=1 \mid \mathbf{x})$を表し、その予測にはシグモイド関数（ロジスティック関数）を用いる。

#### モデルの定義

ロジスティック回帰では、まず説明変数の線形結合を求める：

$$
\begin{equation}
z = w_0 + \sum_{j=1}^p w_j x_j = \mathbf{w}^\top \mathbf{x}'
\end{equation}
$$

ここで$\mathbf{x}' \coloneqq [1, x_1, x_2, \dots, x_p]^\top \in \mathbb{R}^{p+1}$はバイアス項を含んだ拡張入力ベクトル、$\mathbf{w} \in \mathbb{R}^{p+1}$はパラメータベクトルである。

この線形出力$z$に対して、シグモイド関数$\sigma(z)$を適用することで、出力の確率的解釈が得られる：

$$
\begin{equation}
\Pr(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + \exp(-z)}
\end{equation}
$$

したがって、クラスラベル$y \in \{0, 1\}$の**確率モデル**は次のように表される：

$$
\begin{equation}
p(y \mid \mathbf{x}; \mathbf{w}) = \sigma(z)^y (1 - \sigma(z))^{1 - y}
\end{equation}
$$

#### パラメータの推定：最尤推定

ロジスティック回帰のパラメータは**最尤推定**により求める。データ集合$\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$に対して、対数尤度関数は以下のように定義される：

$$
\begin{equation}
\ell(\mathbf{w}) = \sum_{i=1}^n \left[ y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log (1 - \sigma(z^{(i)})) \right]
\end{equation}
$$

ここで$z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)}$である。

この尤度を最大化することで$\mathbf{w}$を学習する。一般には閉形式解を持たないため、**勾配降下法**などの最適化手法を用いて数値的に解く。

勾配は以下のように計算される：

$$
\begin{equation}
\nabla \ell(\mathbf{w}) = \sum_{i=1}^n (y^{(i)} - \sigma(z^{(i)})) \mathbf{x}^{(i)}
\end{equation}
$$

Cox, D. R. (1958). "The regression analysis of binary sequences." Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215–242.