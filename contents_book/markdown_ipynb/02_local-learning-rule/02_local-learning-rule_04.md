## パーセプトロンの学習
ここからは，Hebb則とその派生形による局所学習則を紹介する．前々節で紹介したパーセプトロン

このようなニューロンの抽象化を，分類問題に適用するために提案されたのが**パーセプトロン** (*perceptron*) である．パーセプトロンは，入力と重みの線形結合に対して符号関数（ステップ関数）を適用することにより，2クラス分類を実現する最も基本的な形式の**人工ニューロンモデル**である．

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

#### モデルの構造

入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して，重みベクトル $\mathbf{w} \in \mathbb{R}^{p+1}$（バイアス項 $w_0$ を含む）を用いて線形結合 $z = \mathbf{w}^\top \mathbf{x}'$ を計算する．ただし，$\mathbf{x}' := [1, x_1, x_2, \dots, x_p]^\top$ としてバイアス項を組み込んだ拡張ベクトルを用いる．

次に，活性化関数として**符号関数**を適用する：

$$
\hat{y} = \text{sign}(z) = \begin{cases}
+1 & (z \geq 0) \\
-1 & (z < 0)
\end{cases}
$$

これにより，出力 $\hat{y} \in \{-1, +1\}$ が得られ，2クラス分類を実現する．

#### 学習アルゴリズム：パーセプトロン則

パーセプトロンは**教師あり学習**に基づいて重み $\mathbf{w}$ を更新する．各ステップにおいて，予測と正解が一致していれば何も行わず，誤分類されたときにのみ以下のように重みを更新する：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
$$

ここで，$\eta > 0$ は学習率，$y^{(i)}$ は正解ラベルである．この更新則により，パーセプトロンは誤分類を修正する方向に重みを調整する．

もし訓練データが**線形分離可能**であるならば，このアルゴリズムは有限回の更新で必ず収束する（**パーセプトロン収束定理**）．ただし，データが線形分離不可能な場合は，収束せずに振動を続けることがある．

これは単純ではあるが，この微分不可能な関数による学習則は，現代的に**Straight-Through Estimator** (STE) と呼ばれる概念と同一である．STEの考えはスパイキングニューラルネットワークの学習や，ニューラルネットワークの量子化へと発展する．ここでは深く触れず，第7章で改めて紹介を行う．