- [第2章：発火率モデルと局所学習則](#第2章発火率モデルと局所学習則)
  - [神経細胞の生理](#神経細胞の生理)
  - [発火率モデルとパーセプトロン](#発火率モデルとパーセプトロン)
    - [発火率モデル](#発火率モデル)
    - [線形回帰※](#線形回帰)
      - [最小二乗法によるパラメータの推定](#最小二乗法によるパラメータの推定)
        - [正規方程式を用いた推定](#正規方程式を用いた推定)
        - [勾配法を用いた推定](#勾配法を用いた推定)
    - [ロジスティック回帰とパーセプトロン※](#ロジスティック回帰とパーセプトロン)
  - [Hebb則と主成分分析](#hebb則と主成分分析)
    - [Hebb則](#hebb則)
      - [Hebb則の導出](#hebb則の導出)
    - [Hebb則の安定化とLTP/LTD](#hebb則の安定化とltpltd)
      - [BCM則](#bcm則)
      - [Hebb則の生理的機序](#hebb則の生理的機序)
    - [Oja則](#oja則)
      - [恒常的可塑性](#恒常的可塑性)
      - [Hebb則と主成分分析](#hebb則と主成分分析-1)
    - [Oja則によるPCAの実行](#oja則によるpcaの実行)
    - [Sanger則](#sanger則)
    - [非線形Hebb学習](#非線形hebb学習)
      - [非負主成分分析によるグリッドパターンの創発](#非負主成分分析によるグリッドパターンの創発)
        - [場所細胞の発火パターン](#場所細胞の発火パターン)
  - [独立成分分析](#独立成分分析)
  - [低速特徴分析](#低速特徴分析)
  - [自己組織化マップ](#自己組織化マップ)
    - [競合学習](#競合学習)
    - [自己組織化マップと視覚野の構造](#自己組織化マップと視覚野の構造)
    - [単純なデータセット](#単純なデータセット)

---

# 第2章：発火率モデルと局所学習則
## 神経細胞の生理
## 発火率モデルとパーセプトロン
### 発火率モデル

### 線形回帰※
線形回帰モデル (linear regression) では説明変数 (explanatory variable) $\mathbf{x}$ を線形変換し，目的変数 (objective variable) $y$を予測することを目的とする．説明変数$p$個の線形モデル 

$$
\begin{equation}
y=w_0+w_1x_1+\cdots+w_px_p+\varepsilon=w_0+\sum_{j=1}^p w_jx_j+\varepsilon
\end{equation}
$$

で説明することを考える．説明変数が単一 $(p=1)$ の場合を単回帰，複数 $(p>1)$ の場合を重回帰と呼ぶことがある．

次に，データセット $\mathcal{D}=\left\{\mathbf{x}^{(i)}, y^{(i)}\right\}_{i=1}^n$ を考える．ただし，$\mathbf{x}^{(i)}=\left[x_1^{(i)}, x_2^{(i)}, \ldots, x_p^{(i)}\right]^\top\in \mathbb{R}^p,\ y^{(i)}\in \mathbb{R}$とする．ここで添え字 $(i)$ が付いている場合は観測値を，無い場合はモデル内変数を表すことに注意しよう．
ここで，
$$
\mathbf{y}= \left[ \begin{array}{c} y^{(1)}\\ y^{(2)}\\ \vdots \\ y^{(n)} \end{array} \right] \in \mathbb{R}^n,\quad 
\mathbf{X}=\left[ \begin{array}{ccccc} 1 & x_{1}^{(1)}& x_{2}^{(1)} &\cdots & x_{p}^{(1)} \\ 1& x_{1}^{(2)}& x_{2}^{(2)}&\cdots & x_{p}^{(2)}\\ \vdots & \vdots& \vdots& \ddots & \vdots \\1 &x_{1}^{(n)} & x_{2}^{(n)} &\cdots & x_{p}^{(n)} \end{array} \right] \in \mathbb{R}^{n\times (p+1)}, \quad \mathbf{w}= \left[ \begin{array}{c} w_0\\ w_1\\ \vdots \\ w_p \end{array} \right] \in \mathbb{R}^{p+1}
$$

この場合，回帰モデルは $\mathbf{y}=\mathbf{X}\mathbf{w}+\mathbf{\varepsilon}$と書ける．ただし，$\mathbf{X}$は計画行列 (design matrix)，$\boldsymbol{\varepsilon}$は誤差項である．特に，$\mathbf{\varepsilon}$が平均0, 分散$\sigma^2$の独立な正規分布に従う場合，$\mathbf{y}\sim \mathcal{N}(\mathbf{X}\mathbf{w}, \sigma^2\mathbf{I})$と表せる．

#### 最小二乗法によるパラメータの推定
最小二乗法 (ordinary least squares)により線形回帰のパラメータを推定する．$y$の予測値は$\mathbf{X} \mathbf{w}$なので，誤差 $\mathbf{\delta} \in \mathbb{R}^n$は
$\mathbf{\delta} = \mathbf{y}-\mathbf{X} \mathbf{w}$と表せる．ゆえに目的関数$L(\mathbf{w})$は 

$$
\begin{equation}
L(w)=\sum_{i=1}^n \delta_i^2 = \|\mathbf{\delta}\|^2=\mathbf{\delta}^\top \mathbf{\delta}
\end{equation}
$$

となり， $L(\mathbf{w})$を最小化する$\mathbf{w}$, つまり $\hat {\mathbf {w }}={\underset {\mathbf {w}}{\operatorname {arg min} }}\,L({\mathbf{w}})$
を求める．

##### 正規方程式を用いた推定
条件に基づいて目的関数$L(\mathbf{w})$を微分すると次のような方程式が得られる．

$$
\begin{equation}
\mathbf{X}^\top\mathbf{X}\mathbf{\hat w}=\mathbf{X}^\top\mathbf{y}
\end{equation}
$$

これを**正規方程式** (normal equation)と呼ぶ．この正規方程式より、係数の推定値は$\mathbf{\hat w}={(\mathbf{X}^\top\mathbf{X})}^{-1}X^\top\mathbf{y}$という式で得られる．なお，正規方程式自体は$\mathbf{y}=\mathbf{X}\mathbf{w}$の左から$\mathbf{X}^\top$をかける，と覚えると良い．

##### 勾配法を用いた推定
最小二乗法による回帰直線を勾配法で求めてみよう．$w$の更新式は$w \leftarrow w + \alpha\cdot \dfrac{1}{n} \delta \mathbf{X}$と書ける．ただし，$\alpha$は学習率である．

### ロジスティック回帰とパーセプトロン※
本節では非線形回帰の一種であるロジスティック回帰 (logistic regression) および 1層パーセプトロン (perceptron) を取り扱う．

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
神経回路はどのようにして自己組織化するのだろうか．1940年代にカナダの心理学者Donald O. Hebbにより著書"The Organization of Behavior"{cite:p}`Hebb1949-iv` で提案された学習則は「細胞Aが反復的または持続的に細胞Bの発火に関与すると，細胞Aが細胞Bを発火させる効率が向上するような成長過程または代謝変化が一方または両方の細胞に起こる」というものであった．すなわち，発火に時間的相関のある細胞間のシナプス結合を強化するという学習則である．これを**Hebbの学習則 (Hebbian learning rule)** あるいは**Hebb則(Hebb's rule)** という．Hebb則は (Hebb自身ではなく) Shatzにより"cells that fire together wire together" (共に活動する細胞は共に結合する)と韻を踏みながら短く言い換えられている {cite:p}`Shatz1992-he`．

#### Hebb則の導出
数式でHebb則を表してみよう．$n$個のシナプス前細胞と$m$個の後細胞の発火率をそれぞれ$\mathbf{x}\in \mathbb{R}^n, \mathbf{y}\in \mathbb{R}^m$ とする．前細胞と後細胞間のシナプス結合強度を表す行列を$\mathbf{W}\in \mathbb{R}^{m\times n}$とし，$\mathbf{y}=\mathbf{W}\mathbf{x}$が成り立つとする．このようなモデルを線形ニューロンモデル (Linear neuron model) という．このとき，Hebb則は

$$
\begin{equation}
\tau\frac{d\mathbf{W}}{dt}=\phi(\mathbf{y})\varphi(\mathbf{x})^\top
\end{equation}
$$

として表される．ただし，$\tau$は時定数であり，$\eta:=1/\tau$ は**学習率 (learning rate)** と呼ばれる学習の速さを決定するパラメータとなる．$\varphi(\cdot)$および$\phi(\cdot)$は，それぞれシナプス前細胞および後細胞の活動量に応じて重みの変化量を決定する関数である．ただし，$\varphi(\cdot), \phi(\cdot)$は基本的に恒等関数に設定される場合が多い．この場合，Hebb則は$
\tau\dfrac{d\mathbf{W}}{dt}=\mathbf{y}\mathbf{x}^\top=(\text{post})\cdot (\text{pre})^\top
$と簡潔に表現される．

このHebb則は数学的に導出されたものではないが，特定の目的関数を神経活動及び重みを変化させて最適化するようなネットワークを構築すれば自然に出現する．このようなネットワークを**エネルギーベースモデル (energy-based models)** といい，次章で扱う．エネルギーベースモデルでは，先にエネルギー関数 (あるいはコスト関数) $\mathcal{E}$ を定義し，その目的関数を最小化するような神経活動 $\mathbf{z}$ および重み行列 $\mathbf{W}$ のダイナミクスをそれぞれ,

$$
\begin{equation}
\frac{d \mathbf{z}}{dt}\propto-\frac{\partial \mathcal{E}}{\partial \mathbf{z}},\ \frac{d \mathbf{W}}{dt}\propto-\frac{\partial \mathcal{E}}{\partial \mathbf{W}}
\end{equation}
$$

として導出する．この手順の逆を行う，すなわち先に神経細胞の活動ダイナミクスを定義し，神経活動で積分することで神経回路のエネルギー関数$\mathcal{E}$を導出し，さらに $\mathcal{E}$ を重み行列で微分することでHebb則が導出できる {cite:p}`Isomura2020-sn`．Hebb則の導出を連続時間線形ニューロンモデル $\dfrac{d\mathbf{y}}{dt}=\mathbf{W}\mathbf{x}$ を例にして考えよう．ここで$\dfrac{\partial\mathcal{E}}{\partial\mathbf{y}}:=-\dfrac{d\mathbf{y}}{dt}$となるようなエネルギー関数 $\mathcal{E}(\mathbf{x}, \mathbf{y}, \mathbf{W})$を仮定すると，

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

となり，Hebb則が導出できる (簡単のため時定数は1とした)．

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
Oja則に複数の出力を持たせた場合であっても，出力が直交しないため，PCAの第1主成分しか求めることができない．**Sanger則 (Sanger's rule)**，あるいは**一般化Hebb則 (generalized Hebbian algorithm; GHA)** は，Oja則に**Gram–Schmidtの正規直交化法(Gram–Schmidt orthonormalization)** を組み合わせた学習則であり，次式で表される．

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left(\mathbf{y}\mathbf{x}^\top - \mathrm{LT}\left[\mathbf{y}\mathbf{y}^\top\right] \mathbf{W}\right)
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

## 低速特徴分析
**Slow Feature Analysis (SFA)** とは, 複数の時系列データの中から低速に変化する成分 (slow feature) を抽出する教師なし学習のアルゴリズムである \citep{Wiskott2002-vb,Wiskott2011-uz}．潜在変数 $y$ の時間変化の2乗である $\left(\frac{dy}{dt}\right)^2$を最小にするように教師なし学習を行う．初期視覚野の受容野 \citep{Berkes2005-i} や格子細胞・場所細胞などのモデルに応用がされている \citep{Franzius2007-sf}．

生理学的妥当性についてはいくつかの検討がされている．\citep{Sprekeler2007-qm} ではSTDP則によりSFAが実現できることを報告している．古典的な線形Recurrent neural networkでの実装も提案されている \citep{Lipshutz2020-uj}．

まずデータセットの生成を行う．\citep{Wiskott2002-vb}で用いられているトイデータを用いる．

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