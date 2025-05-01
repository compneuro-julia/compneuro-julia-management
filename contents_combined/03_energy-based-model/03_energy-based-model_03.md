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