# 第2章：発火率モデルと局所学習則
## 神経細胞の生理


こうしたイオンチャネルの働き等を考慮した神経細胞の生物物理モデルに関しては，後の第6章で説明を行う．本章から第5章までは単純化した発火率モデルを使用する．

発火率モデル → 線形回帰 → リッジ回帰 → パーセプトロン → ロジスティック回帰 → Hebb則

## ニューロンの発火率モデルとパーセプトロン


生物学的な神経回路に着想を得た計算モデルのひとつに、**ニューロンの発火率モデル**がある。このモデルでは、個々のニューロンが入力刺激に対してどの程度活性化されるか（＝発火するか）を連続値で表現し、情報の処理や伝達を数理的に記述する。

### ニューロンの発火率モデル

生物学的ニューロンは、樹状突起を通じて他のニューロンから電気的信号（シナプス電位）を受け取り、その総和が一定の閾値を超えた場合に軸索を通じてスパイク（活動電位）を発生させる。このスパイクの頻度（単位時間あたりの発火回数）は、入力の強度に依存して変化することが知られており、この関係を**発火率モデル** (*firing rate model*) と呼ぶ。

数学的には、入力信号の線形結合を $z = \sum_{j=1}^p w_j x_j + w_0$ とし、発火率（ニューロンの出力）を何らかの**活性化関数** $f(z)$ を通じて表現する：

\[
y = f(z) = f\left(\sum_{j=1}^p w_j x_j + w_0\right)
\]

このように、ニューロンは入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して加重和を計算し、それに非線形な関数を適用することで、出力 $y$ を生成する。代表的な活性化関数としては、**シグモイド関数**や**ReLU（Rectified Linear Unit）関数**がある。特にシグモイド関数は、出力が $[0,1]$ の範囲に収まるため、発火率（スパイク頻度）の確率的な解釈と親和性が高い。

この発火率モデルは、後述するロジスティック回帰やニューラルネットワークの基本構成要素として広く用いられている。

### パーセプトロン

このようなニューロンの抽象化を、分類問題に適用するために提案されたのが**パーセプトロン** (*perceptron*) である。パーセプトロンは、入力と重みの線形結合に対して符号関数（ステップ関数）を適用することにより、2クラス分類を実現する最も基本的な形式の**人工ニューロンモデル**である。

#### モデルの構造

入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して、重みベクトル $\mathbf{w} \in \mathbb{R}^{p+1}$（バイアス項 $w_0$ を含む）を用いて線形結合 $z = \mathbf{w}^\top \mathbf{x}'$ を計算する。ただし、$\mathbf{x}' := [1, x_1, x_2, \dots, x_p]^\top$ としてバイアス項を組み込んだ拡張ベクトルを用いる。

次に、活性化関数として**符号関数**を適用する：

\[
\hat{y} = \text{sign}(z) = \begin{cases}
+1 & (z \geq 0) \\
-1 & (z < 0)
\end{cases}
\]

これにより、出力 $\hat{y} \in \{-1, +1\}$ が得られ、2クラス分類を実現する。

#### 学習アルゴリズム：パーセプトロン則

パーセプトロンは**教師あり学習**に基づいて重み $\mathbf{w}$ を更新する。各ステップにおいて、予測と正解が一致していれば何も行わず、誤分類されたときにのみ以下のように重みを更新する：

\[
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
\]

ここで、$\eta > 0$ は学習率、$y^{(i)}$ は正解ラベルである。この更新則により、パーセプトロンは誤分類を修正する方向に重みを調整する。

もし訓練データが**線形分離可能**であるならば、このアルゴリズムは有限回の更新で必ず収束する（**パーセプトロン収束定理**）。ただし、データが線形分離不可能な場合は、収束せずに振動を続けることがある。

### ニューロンモデルとの関係

パーセプトロンは、ニューロンの発火率モデルを単純化した形式として位置づけられる。ニューロンの出力が連続値であるのに対し、パーセプトロンでは出力が離散値（2値）である点が主な違いである。また、ニューロンモデルではなめらかな活性化関数が使われるのに対し、パーセプトロンでは不連続な符号関数が用いられる。

このように、**パーセプトロンはニューロンの発火を単純な2値信号として抽象化した分類器**であり、人工知能の初期における重要なモデルのひとつである。さらに、現代の多層ニューラルネットワーク（ディープラーニング）においても、ニューロンの発火率モデルの構造は基本単位として継承されている。

### 1層パーセプトロン

**パーセプトロン**は、1958年に Rosenblatt によって提案された、最も基本的な形式の線形分類器である。ロジスティック回帰と同様に線形結合を用いるが、確率ではなく**符号関数によって離散的な出力**を行う点が異なる。

#### モデルの定義

入力 $\mathbf{x} \in \mathbb{R}^p$ に対し、次のように線形結合を計算する：

$$
z = \mathbf{w}^\top \mathbf{x}'
$$

そして、活性化関数として符号関数 $\text{sign}(z)$ を適用することで、2クラス分類を行う：

$$
\hat{y} = \begin{cases}
+1 & \text{if } z \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$$

このように、パーセプトロンは出力を $\{-1, +1\}$ のいずれかに決定的に分類する。

#### パーセプトロン学習則

パーセプトロンは教師あり学習アルゴリズムに基づいて重みを更新する。予測が正しい場合は更新を行わず、予測が誤っていた場合にのみ次のようにパラメータを更新する：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
$$

ここで $\eta > 0$ は学習率である。更新は誤分類されたデータ点に対してのみ行われ、分類境界が調整される。データが線形分離可能であれば、パーセプトロン学習則は有限回の更新で収束することが知られている（**パーセプトロン収束定理**）。

分類問題
, perceptron
<https://www.cs.utexas.edu/~gdurrett/courses/fa2022/perc-lr-connections.pdf>

<https://en.wikipedia.org/wiki/Perceptron>

<https://arxiv.org/abs/2012.03642>


perceptronは0/1 or -1/1のどちらか

UNDERSTANDING STRAIGHT-THROUGH ESTIMATOR IN TRAINING ACTIVATION QUANTIZED NEURAL NETS

Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Hinton (2012) in his lecture 15b

G. Hinton. Neural networks for machine learning, 2012.
<https://www.cs.toronto.edu/~hinton/coursera_lectures.html>

delta rule


Here σ denotes the (point-wise) activation function, $W \in R^{m\times n}$
is the weight-matrix and $b \in R^n$
is
the bias-vector. The vector $x \in R^m$ and the vector $y \in R^n$ denote the input, respectively the output

$$
\begin{equation}
y=\sigma(W^\top x + b)
\end{equation}
$$

$$
\begin{align}
& \text { Initialize } W^0, b^0 \text {; } \\
& \text { for } k=1,2, \ldots \text { do } \\
& \qquad \begin{array}{|l}
\text { for } i=1, \ldots, s \text { do } \\
e_i=y_i-\sigma\left(\left(W^k\right)^{\top} x_i+b^k\right) \\
W^{k+1}=W^k+e_i x_i^{\top} \\
b^{k+1}=b^k+e_i
\end{array} \\
& \text { end }
\end{align}
$$

## Hebb則と主成分分析
### Hebb則
神経回路はどのようにして自己組織化するのだろうか．1940年代にカナダの心理学者Donald O. Hebbにより著書"The Organization of Behavior"{cite:p}`Hebb1949-iv` で提案された学習則は「細胞Aが反復的または持続的に細胞Bの発火に関与すると，細胞Aが細胞Bを発火させる効率が向上するような成長過程または代謝変化が一方または両方の細胞に起こる」というものであった．すなわち，発火に時間的相関のある細胞間のシナプス結合を強化するという学習則である．これを**Hebbの学習則** (Hebbian learning rule) あるいは**Hebb則** (Hebb's rule) という．Hebb則は（Hebb自身ではなく）Shatzにより"cells that fire together wire together"（共に活動する細胞は共に結合する）と韻を踏みながら短く言い換えられている {cite:p}`Shatz1992-he`．

数式でHebb則を表してみよう．$n$個のシナプス前細胞と$m$個の後細胞の発火率をそれぞれ$\mathbf{x}\in \mathbb{R}^n, \mathbf{y}\in \mathbb{R}^m$ とする．前細胞と後細胞間のシナプス結合強度を表す行列を$\mathbf{W}\in \mathbb{R}^{m\times n}$とし，$\mathbf{y}=\mathbf{W}\mathbf{x}$が成り立つとする．このようなモデルを線形ニューロンモデル (Linear neuron model) という．このとき，Hebb則は

$$
\begin{equation}
\tau\frac{d\mathbf{W}}{dt}=\phi(\mathbf{y})\varphi(\mathbf{x})^\top
\end{equation}
$$

として表される．ただし，$\tau$は時定数であり，$\eta:=1/\tau$ は**学習率** (learning rate) と呼ばれる学習の速さを決定するパラメータとなる．$\varphi(\cdot)$および$\phi(\cdot)$は，それぞれシナプス前細胞および後細胞の活動量に応じて重みの変化量を決定する関数である．ただし，$\varphi(\cdot), \phi(\cdot)$は基本的に恒等関数に設定される場合が多い．この場合，Hebb則は$
\tau\dfrac{d\mathbf{W}}{dt}=\mathbf{y}\mathbf{x}^\top=(\text{post})\cdot (\text{pre})^\top
$と簡潔に表現される．

このHebb則は数学的に導出されたものではないが，特定の目的関数を神経活動及び重みを変化させて最適化するようなネットワークを構築すれば自然に出現する．このようなネットワークを**エネルギーベースモデル** (energy-based models) といい，次章で扱う．エネルギーベースモデルでは，先にエネルギー関数 (あるいはコスト関数) $\mathcal{E}$ を定義し，その目的関数を最小化するような神経活動 $\mathbf{z}$ および重み行列 $\mathbf{W}$ のダイナミクスをそれぞれ,

$$
\begin{equation}
\frac{d \mathbf{z}}{dt}\propto-\frac{\partial \mathcal{E}}{\partial \mathbf{z}},\ \frac{d \mathbf{W}}{dt}\propto-\frac{\partial \mathcal{E}}{\partial \mathbf{W}}
\end{equation}
$$

として導出する．この手順の逆を行う，すなわち先に神経細胞の活動ダイナミクスを定義し，神経活動で積分することで神経回路のエネルギー関数$\mathcal{E}$を導出し，さらに $\mathcal{E}$ を重み行列で微分することでHebb則が導出できる {cite:p}`Isomura2020-sn`．Hebb則の導出を連続時間線形ニューロンモデル $\dfrac{d\mathbf{y}}{dt}=\mathbf{W}\mathbf{x}$ を例にして考えよう（簡単のため $\tau=1$ とした）．ここで$\dfrac{\partial\mathcal{E}}{\partial\mathbf{y}}:=-\dfrac{d\mathbf{y}}{dt}$となるようなエネルギー関数 $\mathcal{E}(\mathbf{x}, \mathbf{y}, \mathbf{W})$を仮定すると，

$$
\begin{equation}
\mathcal{E}(\mathbf{x}, \mathbf{y}, \mathbf{W})=-\int \mathbf{W}\mathbf{x}\ d\mathbf{y}=-\mathbf{y}^\top \mathbf{W}\mathbf{x} \in \mathbb{R}
\end{equation}
$$

となる．これをさらに$\mathbf{W}$で微分すると，

$$
\begin{equation}
\dfrac{\partial\mathcal{E}}{\partial\mathbf{W}}=-\mathbf{y}\mathbf{x}^\top\Rightarrow
\frac{d\mathbf{W}}{dt}=-\dfrac{\partial\mathcal{E}}{\partial\mathbf{W}}=\mathbf{y}\mathbf{x}^\top
\end{equation}
$$

となり，Hebb則が導出できる．

### Hebb則の安定化とLTP/LTD
#### BCM則
Hebb則には問題点があり，シナプス結合強度が際限なく増大するか，0に近づくこととなってしまう．これを数式で確認しておこう．前細胞と後細胞がそれぞれ1つの場合を考える．2細胞間の結合強度を$w\ (>0)$ とし，$y=wx$が成り立つとすると，Hebb則は$\dfrac{dw}{dt}=\eta yx=\eta x^2w$となる．この場合，$\eta x^2>1$ なら $\lim_{t\to\infty} w= \infty$, $\eta x^2<1$ なら $\lim_{t\to\infty} w= 0$ となる．当然，生理的にシナプス結合強度が無限大となることはあり得ないが，不安定なほど大きくなってしまう可能性があることに違いはない．このため，Hebb則を安定化させるための修正が必要とされた．

Cooper, Liberman, Ojaらにより頭文字をとって**CLO則** (CLO rule) が提案された {cite:p}`Cooper1979-wz`．その後，Bienenstock, Cooper, Munroらにより提案された学習則は同様に頭文字をとって**BCM則** (BCM rule) と呼ばれている{cite:p}`Bienenstock1982-km` {cite:p}`Cooper2012-ec`．

$\mathbf{x}\in \mathbb{R}^d, \mathbf{w}\in \mathbb{R}^d, y\in \mathbb{R}$とし，単一の出力$y = \mathbf{w}^\top \mathbf{x}=\mathbf{x}^\top \mathbf{w}$を持つ線形ニューロンを仮定する．重みの更新則は次のようにする．

$$
\begin{equation}
\frac{d\mathbf{w}}{dt} = \eta_w \mathbf{x} \phi(y, \theta_m)
\end{equation}
$$

ここで関数$\phi$は$\phi(y, \theta_m)=y(y-\theta_m)$などとする．また$\theta_m:=\mathbb{E}[y^2]$は閾値を決定するパラメータ，**修正閾値(modification threshold)** であり，

$$
\begin{equation}
\frac{d\theta_m}{dt} = \eta_{\theta} \left(y^2-\theta_m\right)
\end{equation}
$$

として更新される．

#### Hebb則の生理的機序
LTPの実験的発見 {cite:p}`Bliss1973-vj` {cite:p}`Dudek1992-nz`

### Oja則
Hebb則を安定化させる別のアプローチとして，結合強度を正規化するという手法が考えられる．BCM則と同様に$\mathbf{x}\in \mathbb{R}^d, \mathbf{w}\in \mathbb{R}^d, y\in \mathbb{R}$とし，単一の出力$y = \mathbf{w}^\top \mathbf{x}=\mathbf{x}^\top \mathbf{w}$を持つ線形ニューロンを仮定する．$\eta$を学習率とすると，$\mathbf{w}\leftarrow\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とすれば正規化できる．ここで，$f(\eta):=\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とし，$\eta=0$においてTaylor展開を行うと，

$$
\begin{align}
f(\eta)&\approx f(0) + \eta \left.\frac{df(\eta^*)}{d\eta^*}\right|_{\eta^*=0} + \mathcal{O}(\eta^2)\\
&=\frac{\mathbf{w}}{\|\mathbf{w}\|} + \eta \left(\frac{\mathbf{x}y}{\|\mathbf{w}\|}-\frac{y^2\mathbf{w}}{\|\mathbf{w}\|^3}\right)+ \mathcal{O}(\eta^2)
\end{align}
$$

ここで$\|\mathbf{w}\|=1$として，1次近似すれば$f(\eta)\approx \mathbf{w} + \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)$となる．重みの変化が連続的であるとすると，

$$
\begin{equation}
\frac{d\mathbf{w}}{dt} = \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)
\end{equation}
$$

として重みの更新則が得られる．これを**Oja則 (Oja's rule)** と呼ぶ {cite:p}`Oja1982-yd`．こうして得られた学習則において$\|\mathbf{w}\|\to 1$となることを確認しよう．

$$
\begin{equation}
\frac{d\|\mathbf{w}\|^2}{dt}=2\mathbf{w}^\top\frac{d\mathbf{w}}{dt}= 2\eta y^2\left(1-\|\mathbf{w}\|^2\right)
\end{equation}
$$

より，$\dfrac{d\|\mathbf{w}\|^2}{dt}=0$のとき，$\|\mathbf{w}\|= 1$となる．

#### 恒常的可塑性
Oja則は更新時の即時的な正規化から導出されたものであるが，恒常的可塑性 (synaptic scaling)により安定化しているという説がある{cite:p}`Turrigiano2008-lm`{cite:p}`Yee2017-fb`．しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない{cite:p}`Zenke2017-el`

ToDo:恒常的可塑性の詳細

Johansen, Joshua P., Lorenzo Diaz-Mataix, Hiroki Hamanaka, Takaaki Ozawa, Edgar Ycu, Jenny Koivumaa, Ashwani Kumar, et al. 2014. “Hebbian and Neuromodulatory Mechanisms Interact to Trigger Associative Memory Formation.” Proceedings of the National Academy of Sciences 111 (51): E5584–92.

#### Hebb則と主成分分析
Oja則を用いることで**主成分分析(Principal component analysis; PCA)** という処理をニューラルネットワークにおいて実現できる．主成分分析とは-

ToDo:主成分分析の説明

### Oja則によるPCAの実行
ここでOja則が主成分分析を実行できることを示す．重みの変化量の期待値を取る．

$$
\begin{align}
\frac{d\mathbf{w}}{dt} &= \eta \left(\mathbf{x}y - y^2 \mathbf{w}\right)=\eta \left(\mathbf{x}\mathbf{x}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{x}\mathbf{x}^\top \mathbf{w}\right] \mathbf{w}\right)\\
\mathbb{E}\left[\frac{d\mathbf{w}}{dt}\right] &= \eta \left(\mathbf{C} \mathbf{w} - \left[\mathbf{w}^\top \mathbf{C} \mathbf{w}\right] \mathbf{w}\right)
\end{align}
$$

$\mathbf{C}:=\mathbb{E}[\mathbf{x}\mathbf{x}^\top]\in \mathbb{R}^{d\times d}$とする．$\mathbf{x}$の平均が0の場合，$\mathbf{C}$は分散共分散行列である．$\mathbb{E}\left[\dfrac{d\mathbf{w}}{dt}\right]=0$となる$\mathbf{w}$が収束する固定点(fixed point)では次の式が成り立つ．

$$
\begin{equation}
\mathbf{C}\mathbf{w} = \lambda \mathbf{w}
\end{equation}
$$

これは固有値問題であり，$\lambda:=\mathbf{w}^\top \mathbf{C} \mathbf{w}$は固有値，$\mathbf{w}$は固有ベクトル(eigen vector)になる．

ここでサンプルサイズを$n$とし，$\mathbf{X} \in \mathbb{R}^{d\times n}, \mathbf{y}=\mathbf{X}^\top\mathbf{w} \in \mathbb{R}^n$とする．標本平均で近似して$\mathbf{C}\simeq \mathbf{X}\mathbf{X}^\top$とする．この場合，

$$
\begin{align}
\mathbb{E}\left[\frac{d\mathbf{w}}{dt}\right] &\simeq \eta \left(\mathbf{X}\mathbf{X}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{X}\mathbf{X}^\top \mathbf{w}\right] \mathbf{w}\right)\\
&=\eta \left(\mathbf{X}\mathbf{y} - \left[\mathbf{y}^\top\mathbf{y}\right] \mathbf{w}\right)
\end{align}
$$

となる．

後のためにOja則においてネットワークが$q$個の複数出力を持つ場合を考えよう．重み行列を$\mathbf{W} \in \mathbb{R}^{q\times d}$, 出力を$\mathbf{y}=\mathbf{W}\mathbf{x} \in \mathbb{R}^{q}, \mathbf{Y}=\mathbf{W}\mathbf{X} \in \mathbb{R}^{q\times n}$とする．この場合の更新則は

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left(\mathbf{y}\mathbf{x}^\top - \mathrm{Diag}\left[\mathbf{y}\mathbf{y}^\top\right] \mathbf{W}\right)
\end{equation}
$$

となる．ただし，$\mathrm{Diag}(\cdot)$は行列の対角成分からなる対角行列を生み出す作用素である．

### Sanger則
Oja則に複数の出力を持たせた場合であっても，出力が直交しないため，PCAの第1主成分しか求めることができない．**Sanger則** (Sanger's rule)，あるいは**一般化Hebb則** (generalized Hebbian algorithm; GHA) は，Oja則に**Gram–Schmidtの正規直交化法** (Gram–Schmidt orthonormalization) を組み合わせた学習則であり，次式で表される．

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left[\mathbf{y}\mathbf{x}^\top - \mathrm{LT}\left(\mathbf{y}\mathbf{y}^\top\right) \mathbf{W}\right]
\end{equation}
$$

$\mathrm{LT}(\cdot)$は行列の対角成分より上側の要素を0にした下三角行列(lower triangular matrix)を作り出す作用素である．Sanger則を用いればPCAの第2主成分以降も求めることができる．

### 非線形Hebb学習
出力$\mathbf{y}$に非線形関数$g(\cdot)$を適用し，$\mathbf{y}\to g(\mathbf{y})$として置き換えることで非線形Hebb学習となる{cite:p}`Oja1997-hr`{cite:p}`Brito2016-mx`. 関数`HebbianPCA`の`func`引数に非線形関数を渡すことで実現できる．

ToDo: 詳細

#### 非負主成分分析によるグリッドパターンの創発
内側嗅内皮質(MEC)にある**グリッド細胞 (grid cells)** は六角形格子状の発火パターンにより自己位置等を符号化するのに貢献している．この発火パターンを生み出すモデルは多数あるが，**場所細胞(place cells)** の発火パターンを**非負主成分分析(nonnegative principal component analysis)** で次元削減するとグリッド細胞のパターンが生まれるというモデルがある {cite:p}`Dordek2016-ff`．非線形Hebb学習を用いてこのモデルを実装しよう．なお，同様のことは**非負値行列因子分解 (NMF: nonnegative matrix factorization)** でも可能である．

##### 場所細胞の発火パターン
まず，訓練データとなる場所細胞の発火パターンを人工的に作成する．場所細胞の発火パターンは**Difference of Gaussians (DoG)** で近似する．DoGは大きさの異なる2つのガウス関数の差分を取った関数であり，画像に適応すればband-passフィルタとして機能する．また，DoGは網膜神経節細胞等の受容野のON中心OFF周辺型受容野のモデルとしても用いられる．受容野中央では活動が大きく，その周辺では活動が抑制される，という特性を持つ．2次元のガウス関数とDoG関数を実装する．

Place cellの受容野をDoGに設定したが，これが無いと格子状の受容野は出現しない．path integrationをRNNで実行する場合も同様．一方で，DoGは場所細胞の受容野としては不適切である．

No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit 
<https://openreview.net/forum?id=mxi1xKzNFrb>

ToDo: 他のgrid cellsのモデルについて

## 独立成分分析
独立成分分析（Independent Component Analysis; ICA）は，観測された多次元信号が，統計的に独立な複数の潜在変数（独立成分）の線形混合であると仮定し，元の独立成分を復元することを目的とする手法である．ICAは特に，脳波や自然画像などに見られる信号分離問題に有効である．Blind source separation.

ICAでは，観測ベクトル $\mathbf{x} \in \mathbb{R}^n$ が独立な潜在変数ベクトル $\mathbf{s} \in \mathbb{R}^n$ の線形混合であると仮定する．すなわち，

\[
\mathbf{x} = \mathbf{A} \mathbf{s}
\]

と表される．ここで，$\mathbf{A}$ は未知の正則行列であり，これを分離行列 $\mathbf{W}$ によって推定することを目指す．独立成分 $\mathbf{s}$ の推定は，

\[
\mathbf{y} = \mathbf{W} \mathbf{x}
\]

と表されるように行い，得られた $\mathbf{y}$ の各成分が統計的に独立となるように $\mathbf{W}$ を求める．

ICAを実現するための代表的な原理の一つに，InfoMax（情報最大化）原理がある．これは，出力変数の情報量（エントロピー）を最大化するように変換を学習する枠組みである．InfoMaxにおいては，神経回路の情報伝達能力を最大化するという考えに基づき，非線形関数を通じた変換の出力エントロピーを最大化する．

具体的には，非線形活性化関数 $g(\cdot)$ を用いた出力

\[
\mathbf{y} = g(\mathbf{W} \mathbf{x})
\]

に対し，出力のエントロピー $H(\mathbf{y})$ を最大化するように $\mathbf{W}$ を調整する．ただし，$g(\cdot)$ は例えばシグモイド関数のような非線形性を持つ関数とする．

InfoMax原理に基づくICAの学習則は，出力の対数尤度を最大化する勾配上昇法として導出される．例えば，出力の対数尤度を $L(\mathbf{W})$ としたとき，

\[
\nabla_{\mathbf{W}} L(\mathbf{W}) \propto \left( \mathbf{I} + (\mathbf{1} - 2\mathbf{y}) \mathbf{x}^\top \right) \mathbf{W}^{-\top}
\]

といった形の学習則が得られる（ここで，$\mathbf{1}$ は全ての成分が1のベクトル）．このようにして，$\mathbf{y}$ の統計的独立性が最大化されるような $\mathbf{W}$ を求めることが可能となる．

InfoMax ICAは，確率密度関数の仮定を明示せずに信号の非ガウス性を利用する点で有効であり，実際の信号分離問題において高い性能を示すことが多い．また，非ガウス性の測度としてはクルトーシスやネガエントロピーなども用いられ，これによりFastICAなどの手法も導出されている．

以上より，独立成分分析は，観測データを生成する潜在変数の独立性という前提に基づき，情報理論的な原理に従ってその分離を行う手法であり，InfoMaxはその実現方法の一つとして広く用いられている．

## 低速特徴分析
**Slow Feature Analysis (SFA)** とは, 複数の時系列データの中から低速に変化する成分 (slow feature) を抽出する教師なし学習のアルゴリズムである \citep{Wiskott2002-vb,Wiskott2011-uz}．潜在変数 $y$ の時間変化の2乗である $\left(\frac{dy}{dt}\right)^2$を最小にするように教師なし学習を行う．初期視覚野の受容野 \citep{Berkes2005-i} や格子細胞・場所細胞などのモデルに応用がされている \citep{Franzius2007-sf}．

生理学的妥当性についてはいくつかの検討がされている．\citep{Sprekeler2007-qm} ではSTDP則によりSFAが実現できることを報告している．古典的な線形Recurrent neural networkでの実装も提案されている \citep{Lipshutz2020-uj}．

より具体的には，観測された高次元の入力信号 $\mathbf{x}(t) \in \mathbb{R}^n$ から，できるだけゆっくりと変化するスカラー出力 $y(t) = g(\mathbf{x}(t))$ を学習によって導出することが目的である．このとき，関数 $g(\cdot)$ は通常，入力に対して線形または非線形な写像である．

SFAの基本的な最適化問題は以下のように定式化される：

\[
\min_{g} \left\langle \left( \frac{d}{dt} g(\mathbf{x}(t)) \right)^2 \right\rangle_t
\]

ただし，$\langle \cdot \rangle_t$ は時間平均を意味する．このままでは自明な定数解（全く変化しない出力）が得られるため，以下のような制約条件を課す：

1. **零平均**：$\langle y(t) \rangle_t = 0$
2. **単位分散**：$\langle y(t)^2 \rangle_t = 1$
3. **異なる特徴間の直交性**（複数のslow featureを抽出する場合）：$\langle y_i(t) y_j(t) \rangle_t = 0\quad (i \ne j)$

これらの制約により，情報量がありながらも変化の遅い特徴を抽出することが可能となる．実際のアルゴリズムでは，まず入力信号に対して一定の非線形写像（例えば多項式基底関数など）を適用した後，主成分分析（PCA）によって前処理を行い，その後時間的変化の最小化問題を一般化固有値問題として解くことでslow featuresを得る．

まずデータセットの生成を行う．\citep{Wiskott2002-vb}で用いられているトイデータを用いる．

Slow Feature Analysis (SFA) は，時系列データに含まれる情報のうち，時間的に最もゆっくりと変化する成分（slow features）を抽出するための教師なし学習アルゴリズムである．このアルゴリズムでは，観測された高次元の信号 $\mathbf{x}(t) \in \mathbb{R}^n$ に対して，線形または非線形な写像 $y(t) = g(\mathbf{x}(t))$ を学習し，その出力が時間的に滑らかになるように設計される．特に線形SFAの場合，写像 $g(\mathbf{x})$ は線形関数 $\mathbf{w}^\top \mathbf{x}(t)$ として表され，その時間微分の2乗平均 $\left\langle \left( \frac{d}{dt} \mathbf{w}^\top \mathbf{x}(t) \right)^2 \right\rangle_t$ を最小化することが目的となる．

この最適化問題を解くためには，まず入力データ $\mathbf{x}(t)$ を前処理し，時間平均を引くことでゼロ平均化する．次に，共分散行列 $\mathbf{C}_x = \langle \tilde{\mathbf{x}}(t) \tilde{\mathbf{x}}(t)^\top \rangle_t$ を求め，これに対して固有値分解 $\mathbf{C}_x = \mathbf{E} \mathbf{D} \mathbf{E}^\top$ を適用することで主成分空間を構成し，白色化変換 $\mathbf{z}(t) = \mathbf{D}^{-1/2} \mathbf{E}^\top \tilde{\mathbf{x}}(t)$ を得る．この変換により，$\mathbf{z}(t)$ は単位分散かつ直交性を持つ特徴ベクトルとなる．

白色化されたデータに対して時間微分を近似的に計算し，$\dot{\mathbf{z}}(t) = \mathbf{z}(t+1) - \mathbf{z}(t)$ と定義することで，その共分散行列 $\mathbf{C}_{\dot{z}} = \langle \dot{\mathbf{z}}(t) \dot{\mathbf{z}}(t)^\top \rangle_t$ を構築することができる．SFAにおける主たる目的は，この微分共分散行列に関する最小固有値問題を解くことである．すなわち，$\mathbf{C}_{\dot{z}}$ に対する固有値分解または特異値分解（SVD）を行い，最小固有値に対応する固有ベクトル $\mathbf{u}_1$ を求めることで，最もゆっくりと変化する成分 $y(t) = \mathbf{u}_1^\top \mathbf{z}(t)$ を得ることができる．複数のslow featuresを得たい場合は，対応する小さい固有値順に固有ベクトルを選択することで可能となる．

最終的に，元のデータ空間におけるslow featuresを得るためには，逆変換を施して $\mathbf{W} = \mathbf{E} \mathbf{D}^{-1/2} \mathbf{U}$ とし，$\mathbf{U}$ は選択された固有ベクトルからなる行列である．この射影行列 $\mathbf{W}$ を用いることで，元の信号 $\tilde{\mathbf{x}}(t)$ からslow feature $y(t) = \mathbf{W}^\top \tilde{\mathbf{x}}(t)$ を得ることができる．このようにして，SFAはSVDを通じて効率的に解くことが可能であり，低速に変化する潜在表現を抽出するための強力な手法となっている．

## 自己組織化マップ

### 競合学習

### 自己組織化マップと視覚野の構造
視覚野にはコラム構造が存在する．こうした構造は神経活動依存的な発生  (activity dependent development) により獲得される．本節では視覚野のコラム構造を生み出す数理モデルの中で，**自己組織化マップ (self-organizing map)** {cite:p}`Kohonen1982-mn`, {cite:p}`Kohonen2013-yt`を取り上げる．

自己組織化マップを視覚野の構造に適応したのは{cite:p}`Obermayer1990-gq` {cite:p}`N_V_Swindale1998-ri`などの研究である．視覚野マップの数理モデルとして自己組織化マップは受容野を考慮しないなどの簡略化がなされているが，単純な手法にして視覚野の構造に関する良い予測を与える．他の数理モデルとしては自己組織化マップと発想が類似している **Elastic net**  {cite:p}`Durbin1987-bp` {cite:p}`Durbin1990-xx` {cite:p}`Carreira-Perpinan2005-gy`　(ここでのElastic netは正則化手法としてのElastic net regularizationとは異なる)や受容野を明示的に設定した {cite:p}`Tanaka2004-vz`， {cite:p}`Ringach2007-oe`などのモデルがある．総説としては{cite:p}`Das2005-mq`，{cite:p}`Goodhill2007-va` ，数理モデル同士の関係については{cite:p}`2002-nm`が詳しい．

自己組織化マップでは「抹消から中枢への伝達過程で損失される情報量」，および「近い性質を持ったニューロン同士が結合するような配線長」の両者を最小化するような学習が行われる．包括性 (coverage) と連続性 (continuity) のトレードオフとも呼ばれる {cite:p}`Carreira-Perpinan2005-gy`　 (Elastic netは両者を明示的に計算し，線形結合で表されるエネルギー関数を最小化する．Elastic netは本書では取り扱わないが，MATLAB実装が公開されている
<https://faculty.ucmerced.edu/mcarreira-perpinan/research/EN.html>) ． 連続性と関連する事項として，近い性質を持つ細胞が脳内で近傍に存在するような発生/発達過程を**トポグラフィックマッピング (topographic mapping)** と呼ぶ．トポグラフィックマッピングの数理モデルの初期の研究としては{cite:p}`Von_der_Malsburg1973-bz` {cite:p}`Willshaw1976-zo` {cite:p}`Takeuchi1979-mi`などがある．

発生の数理モデルに関する総説 {cite:p}`Van_Ooyen2011-fz`, {cite:p}`Goodhill2018-ho`

### 単純なデータセット
SOMにおける$n$番目の入力を $\mathbf{v}(t)=\mathbf{v}_n\in \mathbb{R}^{D} (n=1, \ldots, N)$，$m$番目のニューロン$ (m=1, \ldots, M) $の重みベクトル (または活動ベクトル, 参照ベクトル) を$\mathbf{w}_m(t)\in \mathbb{R}^{D}$とする {cite:p}`Kohonen2013-yt`．また，各ニューロンの物理的な位置を$\mathbf{x}_m$とする．このとき，$\mathbf{v}(t)$に対して$\mathbf{w}_m(t)$を次のように更新する．

まず，$\mathbf{v}(t)$と$\mathbf{w}_m(t)$の間の距離が最も小さい (類似度が最も大きい) ニューロンを見つける．距離や類似度としてはユークリッド距離やコサイン類似度などが考えられる．

$$
\begin{align}
&[\text{ユークリッド距離}]: c = \underset{m}{\operatorname{argmin}}\left[\|\mathbf{v}(t)-\mathbf{w}_m(t)\|^2\right]\\
&[\text{コサイン類似度}]: c  = \underset{m}{\operatorname{argmax}}\left[\frac{\mathbf{w}_m(t)^\top\mathbf{v}(t)}{\|\mathbf{w}_m(t)\|\|\mathbf{v}(t)\|}\right]
\end{align}
$$

この，$c$番目のニューロンを**勝者ユニット(best matching unit; BMU)** と呼ぶ．コサイン類似度において，$\mathbf{w}_m(t)^\top\mathbf{v}(t)$は線形ニューロンモデルの出力となる．このため，コサイン距離を採用する方が生理学的に妥当でありSOMの初期の研究ではコサイン類似度が用いられている {cite:p}`Kohonen1982-mn`．しかし，コサイン類似度を用いる場合は$\mathbf{w}_m$および$\mathbf{v}$を正規化する必要がある．ユークリッド距離を用いると正規化なしでも学習できるため，SOMを応用する上ではユークリッド距離が採用される事が多い．ユークリッド距離を用いる場合，$\mathbf{w}_m$は重みベクトルではなくなるため，活動ベクトルや参照ベクトルと呼ばれる．ここでは結果の安定性を優先してユークリッド距離を用いることとする．

こうして得られた$c$を用いて$\mathbf{w}_m$を次のように更新する．

$$
\begin{equation}
\mathbf{w}_m(t+1)=\mathbf{w}_m(t)+h_{cm}(t)[\mathbf{v}(t)-\mathbf{w}_m(t)]
\end{equation}
$$

ここで$h_{cm}(t)$は近傍関数 (neighborhood function) と呼ばれ，$c$番目と$m$番目のニューロンの距離が近いほど大きな値を取る．ガウス関数を用いるのが一般的である．

$$
\begin{equation}
h_{cm}(t)=\alpha(t)\exp\left(-\frac{\|\mathbf{x}_c-\mathbf{x}_m\|^2}{2\sigma^2(t)}\right)
\end{equation}
$$

ここで$\mathbf{x}$はニューロンの位置を表すベクトルである．また，$\alpha(t), \sigma(t)$は単調に減少するように設定する．\footnote{Generative topographic map (GTM)を用いれば$\alpha(t), \sigma(t)$の縮小は必要ない．また，SOMとGTMの間を取ったモデルとしてS-mapがある．}