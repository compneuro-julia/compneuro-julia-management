# 第3章：エネルギーベースモデル
本章では、**エネルギーベースモデル**（energy-based models; EBMs）と呼ばれる確率モデルの枠組みを取り上げる。EBMsは、入力データに対して**スカラー値のエネルギー（あるいはコスト）を割り当てる関数**を定義し、そのエネルギーを最小化するようにシステムの状態を決定・学習するという特徴を持つ。これは、ニューラルネットワークなどの高次元な状態空間における確率的な推論や学習に広く応用されている枠組みである{cite:p}`LeCun2006-dt`。

エネルギーベースモデルでは、入力ベクトル $\mathbf{x} \in \mathbb{R}^d$ に対して、パラメータ $\theta$ に依存する**エネルギー関数** (energy function) $E_\theta : \mathbb{R}^d \to \mathbb{R}$ を定義する。このエネルギー関数は、ある状態 $\mathbf{x}$ の「好ましさ」や「自然さ」を定量的に評価するものであり、エネルギーが小さいほどその状態がより実現しやすいと解釈される。

このようなエネルギー関数を用いて、状態 $\mathbf{x}$ の**確率密度関数** $p_\theta(\mathbf{x})$ を以下のように定義する：

$$
p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}
$$

ここで $Z_\theta$ は**分配関数**（partition function）と呼ばれ、確率分布が正規化されるようにするための定数である。すなわち、

$$
Z_\theta = \int \exp(-E_\theta(\mathbf{x})) \, d\mathbf{x}
$$

と定義される。$Z_\theta$ は状態空間全体にわたる積分であり、一般には計算が困難である点がEBMの学習と推論を難しくする主な要因の一つである。

このように、エネルギーベースモデルは、確率分布を明示的にパラメトライズする代わりに、各状態に対するスカラーのスコア（＝エネルギー）を割り当て、そのスコアを通じて確率的な解釈を与えるという柔軟な表現力を持つ。そのため、EBMsは画像生成、異常検知、表現学習など、多様な応用分野で注目されている。

また、エネルギー関数 $E_\theta(\mathbf{x})$ の定義により、EBMs は生成モデルとして扱うこともできるが、識別的モデルとして利用することも可能である。たとえば、識別タスクにおいては、入力 $\mathbf{x}$ とラベル $y$ のペア $(\mathbf{x}, y)$ に対して $E_\theta(\mathbf{x}, y)$ を定義し、正しいラベルに対するエネルギーを最小にするようなパラメータ $\theta$ を学習することができる。このように、EBM は生成と識別の両方の枠組みにまたがる柔軟なモデルである。

推論時と学習時の双方においてエネルギーを最小化するようにネットワークの状態を更新する

エネルギーベースモデルは、
神経系の状態遷移と安定性を記述する枠組みとして自然であり、
記憶や知覚といった脳の高次機能の数理的モデル化を可能にし、
確率的処理や最適化の観点からも神経活動の特徴をうまく表現できるため、
計算論的神経科学における理論的支柱の1つとして重要な役割を果たしています。

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

## Hopfieldモデル2
**Hopfieldモデル**は、記憶やパターン補完といった機能を持つ単純なリカレント型のニューラルネットワークであり、エネルギーベースモデル（EBM）の古典的な例でもある。1982年にJohn Hopfield によって提案されたこのモデルは、**連想記憶**（associative memory）の理論的な基盤として広く知られている。

Hopfieldモデルは、$n$ 個のノード（ニューロン）から構成され、それぞれのノードは2値の状態 $x_i \in \{-1, +1\}$ をとる。各ノードは他のすべてのノードと対称な重みで接続されており、自己結合（$w_{ii}$）は存在しないと仮定される。すなわち、接続重み行列 $\mathbf{W} = [w_{ij}]$ は対称であり、$w_{ij} = w_{ji}$、かつ $w_{ii} = 0$ が成り立つ。

このネットワークの**状態ベクトル**を $\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top$ とすると、Hopfieldモデルにおける**エネルギー関数**は以下のように定義される：

$$
E(\mathbf{x}) = -\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} x_i x_j + \sum_{i=1}^n \theta_i x_i
$$

ここで $\theta_i$ は各ニューロンのバイアス項に相当する。エネルギー関数はネットワークの状態 $\mathbf{x}$ に対する「好ましさ」を表しており、エネルギーが低い状態ほど、安定で記憶として定着しやすい状態と解釈される。

### 状態の更新則

Hopfieldモデルにおける状態の更新は、次のような**非同期更新**で行われることが多い。任意のニューロン $i$ に対して、他のノードの状態に基づいて次のように状態を更新する：

$$
x_i \leftarrow \mathrm{sgn}\left(\sum_{j=1}^n w_{ij} x_j - \theta_i\right)
$$

ここで $\mathrm{sgn}(\cdot)$ は符号関数であり、引数が正なら $+1$、負なら $-1$ を返す（0の場合の処理は文脈に応じて定義される）。この更新を繰り返すことで、ネットワークはエネルギーを徐々に減少させながら、局所的な極小点へと収束していく。

特筆すべき性質として、更新によってエネルギー関数 $E(\mathbf{x})$ は単調に減少するため、エネルギー関数をポテンシャルとしてもつ力学系としての解析が可能である。すなわち、HopfieldモデルはLyapunov関数としてのエネルギーを持ち、定常点（安定な記憶状態）に向かって収束する。

### 記憶の埋め込み

Hopfieldネットワークは、あらかじめいくつかのパターン $\{\boldsymbol{\xi}^\mu\}_{\mu=1}^P$ を「記憶」として埋め込むことができる。その際、Hebbの学習則に基づいて重み $w_{ij}$ は以下のように設定される：

$$
w_{ij} = \frac{1}{n} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu \quad (i \ne j)
$$

ここで $\xi_i^\mu \in \{-1, +1\}$ は、記憶パターン $\mu$ におけるニューロン $i$ の状態である。こうして構築されたネットワークは、記憶パターンの近傍からの入力に対しても正しい記憶を再生する「パターン補完」能力を持つ。

このように、Hopfieldモデルはエネルギーベースの観点から明確に定式化されており、記憶・補完・安定性といった知能の基本機能をシンプルに実現する数理的枠組みである。後のBoltzmannマシンやディープラーニングに至る発展的研究においても、その構造的・理論的基盤として重要な役割を果たしている。

## Hopfieldネットワークの記憶容量

Hopfieldネットワークは、$n$ 個の2値ニューロン（$x_i \in \{-1, +1\}$）を持ち、重み行列 $\mathbf{W} = [w_{ij}]$ により完全連結された再帰的ネットワークである。このモデルは、与えられたパターン $\boldsymbol{\xi}^\mu = (\xi_1^\mu, \ldots, \xi_n^\mu)^\top \in \{-1, +1\}^n$ をエネルギー極小点として埋め込む、いわゆる**連想記憶**の機構として機能する。

### 記憶の埋め込み：Hebb則

$P$ 個のパターン $\{ \boldsymbol{\xi}^1, \dots, \boldsymbol{\xi}^P \}$ をネットワークに記憶させるとき、典型的にはHebb則に基づいて重みは次のように設定される：

$$
w_{ij} = \frac{1}{n} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu \quad (\text{ただし } w_{ii} = 0)
$$

この重み行列により、各パターンはネットワークの**安定点**（エネルギー極小点）となるように構成される。

### 記憶容量の理論限界

Hopfieldモデルの記憶容量とは、**パターンを誤りなく安定に記憶できる最大のパターン数 $P_{\max}$**を指す。著名な統計力学的解析（Amit, Gutfreund & Sompolinsky, 1985）によれば、ランダムに独立なパターンを格納する場合、次のような容量限界が存在する：

$$
P_{\max} \approx 0.138 \, n
$$

すなわち、ニューロン数 $n$ に比例して最大約13.8%の個数の独立パターンが格納可能である。

より正確には、記憶されたパターンが**安定な固定点（誤り訂正可能なアトラクタ）**である確率が十分高いような $P$ を求めると、次の臨界値 $\alpha_c$ が得られる：

$$
\alpha = \frac{P}{n} < \alpha_c \approx 0.138
$$

この $\alpha$ は「負荷率（loading rate）」と呼ばれ、$\alpha > \alpha_c$ となると、記憶パターンが互いに干渉し、誤認識やスピンガラス的状態が生じやすくなる。

### 誤り訂正能力とパターンの性質

Hopfieldネットワークは、記憶されたパターンの**近傍**からでも正しく復元する能力（誤り訂正能力）を持つ。ただしこれは格納パターン同士が**十分に直交的（相互に非相関）**であることが前提であり、類似したパターンを多数記憶させた場合には干渉が大きくなり、容量が実質的に低下する。

### Dense Associative Memory との比較

上述のように、古典的Hopfieldモデルでは $P_{\max} = O(n)$ にとどまるが、**Dense Associative Memory**（DAM）や**Modern Hopfield Networks**では、エネルギー関数の非線形化やベクトル表現の導入により、**記憶容量が $O(e^n)$ に拡張可能**であることが示されている。これは、記憶を単純な線形超平面ではなく、複雑な非線形エネルギー地形として表現することにより、多数のパターンを安定に埋め込めるようになるためである。

Hopfieldネットワークの記憶容量は理論的に最大でも $0.138n$ に制限される。この限界は、モデルのシンプルさに起因するが、非線形拡張や高次表現を導入することで指数的な容量向上が可能となる。記憶容量の議論は、連想記憶モデルの性能と限界を理解するうえで極めて重要な理論的指標である。


### コラム：稠密連想記憶モデルと現代的Hopfieldモデル
古典的Hopfieldネットワークは、安定な状態（固定点）としてパターンを記憶する連想記憶モデルであるが、記憶容量がニューロン数 $n$ に比例するという制約があった。これを克服するために、エネルギー関数に高次非線形性を導入した**稠密連想記憶** (Dense Associative Memor; DAM）が提案され、指数的に多くの記憶を格納できるようになった。

さらにこの発想を発展させたのが、**現代的Hopfieldモデル** (Modern Hopfield Network; MHNs）である。MHNsは連続ベクトルを扱い、入力（クエリ）ベクトルに対して**ソフトマックス付き内積**に基づく重み付き平均により、最も類似した記憶を再生する。この動作はTransformerにおける**注意機構（attention）**と数学的に等価であり、attentionの背後にはHopfield型のエネルギー最小化原理が潜んでいるとも解釈できる。

この構造はまた、ニューラルネットワークにおける**Key-Value Memory**の設計とも一致する。すなわち、記憶ベクトル（value）に対応するキーを入力と比較し、類似度に応じて情報を再構成する仕組みである．

は、Modern Hopfield Networksの動作と本質的に同一である。MHNsはこのような**連想によるメモリ検索**の理論的基盤として、記憶・注意・検索の統合的理解を可能にする。

このように、DAMやMHNsは、古典的連想記憶を超えて、現代の深層学習における注意・記憶メカニズムとの橋渡しをする重要な枠組みとなっている。

https://www.cell.com/neuron/fulltext/S0896-6273(25)00172-2
https://ieeexplore.ieee.org/document/5008975

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

Boltzmannマシンは、確率的生成モデルの一例として、その状態の確率分布をエネルギー関数に基づいて定義するモデルである。ここで、システムの状態は $\mathbf{s} = (s_1, s_2, \ldots, s_N)$ という2値のユニットの組で表され、各 $s_i$ は0または1の値を取る。Boltzmannマシンでは、各状態のエネルギーは以下の式によって与えられる：

$$
E(\mathbf{s}) = -\sum_{i} b_i s_i - \sum_{i < j} W_{ij} s_i s_j
$$

ここで、$b_i$ は各ユニットに対応するバイアス項、$W_{ij}$ はユニット $i$ と $j$ の間の対称的な結合重みを表す。状態 \(\mathbf{s}\) が出現する確率は、エネルギー関数に基づいてボルツマン分布として定義され、以下のように記述される：

$$
P(\mathbf{s}) = \frac{1}{Z} \exp\left(-E(\mathbf{s})\right)
$$

ここで、正規化定数 $Z$（分配関数）は全状態にわたる和で定義される：

$$
Z = \sum_{\mathbf{s}} \exp\left(-E(\mathbf{s})\right)
$$

このモデルは、全ユニット間に結合が存在するため、内部の依存関係が複雑になり、特に学習の際にパラメータ更新のための勾配計算が指数的な計算量を要するという難点がある。

Boltzmannマシンにおける学習および推論の主要な困難さは、その計算に内在する分配関数 $Z$ の評価に起因する。Boltzmannマシンでは、エネルギー関数

$$
E(\mathbf{s}) = -\sum_{i} b_i s_i - \sum_{i<j} W_{ij} s_i s_j
$$

に従い、状態 \(\mathbf{s}\) の確率分布は

$$
P(\mathbf{s}) = \frac{1}{Z} \exp\left(-E(\mathbf{s})\right)
$$

と定義されるが、ここで正規化定数 $Z$ は

$$
Z = \sum_{\mathbf{s}} \exp\left(-E(\mathbf{s})\right)
$$

と全可能状態 \(\mathbf{s}\) にわたる和として計算されなければならない。各ユニットが2値の確率変数である場合、全状態数は \(2^N\) となるため、ネットワークの規模が大きくなるとこの和は指数関数的に増大し、厳密な計算が事実上不可能となる。

さらに、学習に必要なパラメータ更新のための勾配計算でも、この正規化定数 $Z$ に依存する項が現れる。具体的には、尤度関数の勾配として、例えば重み $W_{ij}$ に関しては

$$
\frac{\partial \log P(\mathbf{s})}{\partial W_{ij}} = \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}
$$

と表されるが、ここで \(\langle s_i s_j \rangle_{\text{model}}\) はモデル分布における期待値であり、これは

$$
\langle s_i s_j \rangle_{\text{model}} = \sum_{\mathbf{s}} s_i s_j \, P(\mathbf{s})
$$

として計算される必要がある。しかし、前述のように $P(\mathbf{s})$ の計算には $Z$ の求積が不可欠であり、これもまた指数的な計算量を要するため、直接計算することは困難である。

このような計算の困難性は、統計物理における分配関数の計算問題と同様に、組み合わせ爆発（combinatorial explosion）の問題として知られ、計算複雑性理論では #P困難（#P-complete）であると指摘される。これに対処するため、実際の学習ではサンプルに基づく近似手法（モンテカルロ法、ギブスサンプリングなど）や、特定の近似アルゴリズム（コントラスト・ダイバージェンスなど）が利用される。しかしこれら近似手法にも収束の問題や精度の限界が存在するため、一般的なBoltzmannマシンは大規模な問題に対して直接適用するのが難しく、その計算効率の改善は依然として重要な研究課題である。

この問題点を解消するために考案されたのが、制限Boltzmannマシン（Restricted Boltzmann Machine: RBM）である。RBMでは、ネットワークを二層構造に限定し、可視層 $\mathbf{v}$ と隠れ層 $\mathbf{h}$ のみを用いる。ここで、可視ユニット $v_i$ は入力データを表し、隠れユニット $h_j$ はデータの特徴（潜在変数）を表す。RBMのエネルギー関数は次の形で定義される：

$$
E(\mathbf{v}, \mathbf{h}) = -\sum_{i} a_i v_i - \sum_{j} b_j h_j - \sum_{i, j} v_i W_{ij} h_j
$$

このとき、$a_i$ は可視ユニットのバイアス、$b_j$ は隠れユニットのバイアス、そして $W_{ij}$ は可視ユニットと隠れユニット間の結合重みである。RBMでは、同一層内のユニット間の結合（例えば、可視層同士、隠れ層同士）は存在しないため、モデル内の条件付き独立性が成立する。具体的には、隠れ層の各ユニットは可視層が与えられた条件下で独立に分布し、その条件付き確率は次の式で表される：

$$
P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i} v_i W_{ij}\right)
$$

また、可視層の各ユニットに関しても同様に、

$$
P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_{j} h_j W_{ij}\right)
$$

と記述される。ここで、\(\sigma(x) = \frac{1}{1+\exp(-x)}\) はシグモイド関数である。これらの性質により、RBMは効率的なギブスサンプリングが可能となり、コントラスト・ダイバージェンス（Contrastive Divergence, CD）と呼ばれる近似的な学習アルゴリズムが用いられて実用的な学習が可能となる。

このようにして、Boltzmannマシンは複雑な結合を持つモデルとして理論的な基盤を提供する一方、RBMはその結合を制限することにより計算の効率化を実現している。これらのモデルは、特にディープラーニングにおける事前学習や特徴抽出の文脈で重要な役割を果たし、画像認識や信号処理など幅広い応用がなされている。

### 制限 Boltzmann マシン
(Restricted Boltzmann machine) 
(cf.) <http://deeplearning.net/tutorial/rbm.html>

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

### 訓練データで学習
$$
\begin{align}
p_\theta(\mathbf{h}|\mathbf{v})&=\prod_i p_\theta(h_i=1|\mathbf{v})=\prod_i \sigma(c_i + W_i \mathbf{v})\\
p_\theta(\mathbf{v}|\mathbf{h})&=\prod_j p_\theta(v_j=1|\mathbf{h})=\prod_j \sigma(b_j + W_j^\top \mathbf{h})
\end{align}
$$

## スパース符号化モデル
### スパース符号化と生成モデル
**スパース符号化モデル** (Sparse coding model) {cite:p}`Olshausen1996-xe` {cite:p}`Olshausen1997-qu`はV1のニューロンの応答特性を説明する**線形生成モデル** (linear generative model)である．まず，画像パッチ $\mathbf{x}$ が基底関数(basis function) $\mathbf{\Phi} = [\phi_j]$ のノイズを含む線形和で表されるとする (係数は $\mathbf{r}=[r_j]$ とする)．

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

### スパース符号化モデルの実装
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
\frac{d \mathbf{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} \mathbf{U}^{\top}\bigg[\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\odot\underbrace{(\mathbf{x}-f(\mathbf{z}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\mathbf{r})\Bigg)\\
\frac{d \mathbf{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(\mathbf{U}^h)^\top\bigg[\frac{\partial f(\mathbf{z}^h)}{\partial \mathbf{z}^h}\odot\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\mathbf{r}^h)\Bigg)
\end{align}
$$

ただし，$k_1$は更新率 (updating rate)である．または，発火率の時定数を$\tau:=1/k_1$として，$k_1$は発火率の時定数$\tau$の逆数であると捉えることもできる．ここで1番目の式において，中間表現 $\mathbf{r}$ のダイナミクスはbottom-up errorとtop-down errorで記述されている．このようにbottom-up errorが $\mathbf{r}$ への入力となることは自然に導出される．なお，top-down errorに関しては高次からの予測 (prediction)の項 $f(\mathbf{x}^h)$とleaky-integratorとしての項 $-\mathbf{r}$に分割することができる．また$\mathbf{U}^\top, (\mathbf{U}^h)^\top$は重み行列の転置となっており，bottom-upとtop-downの投射において対称な重み行列を用いることを意味している．$-g'(\mathbf{r})$は発火率を抑制してスパースにすることを目的とする項だが，無理やり解釈をすると自己再帰的な抑制と言える．