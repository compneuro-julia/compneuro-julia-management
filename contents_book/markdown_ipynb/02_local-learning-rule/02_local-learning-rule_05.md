## 主成分分析
前々節で紹介したOja則を用いることで**主成分分析** (Principal component analysis; PCA) という処理をニューラルネットワークにおいて実現できる．

#### 主成分分析
主成分分析 (PCA) は，高次元のデータに内在する低次元の構造を抽出するための線形次元削減法である．この手法は，分散が最大となる方向にデータを射影することにより，元の情報をなるべく保ちながら次元を削減する．

まず，$n$ 個のサンプル $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$ が $d$ 次元の実ベクトル空間 $\mathbb{R}^d$ に属するとし，これらを列ベクトルとしてまとめたデータ行列を $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ とする．PCA では以下の手順を踏む．

PCA の目的は，情報損失（再構成誤差）を最小限に抑えながら，できるだけ少ない次元でデータを表現することである．この観点から，PCA は次の最適化問題の解とみなすこともできる：

$$
\max_{\mathbf{W}_m \in \mathbb{R}^{d \times m}} \operatorname{Tr}(\mathbf{W}_m^\top \mathbf{C} \mathbf{W}_m), \quad \text{s.t. } \mathbf{W}_m^\top \mathbf{W}_m = \mathbf{I}_m,
$$

ここで $\operatorname{Tr}(\cdot)$ はトレース演算，$\mathbf{I}_m$ は $m$ 次の単位行列である．この最適化問題の解は，共分散行列 $\mathbf{C}$ の上位 $m$ 個の固有ベクトルからなる直交行列 $\mathbf{W}_m$ である．

PCA はデータの冗長性を取り除くと同時に，ノイズの低減や可視化の手法としても広く応用される．また，線形変換であるため，計算効率も高いという特徴がある．

svdを用いて実装をする．


1. **平均の除去**  
   各特徴量について平均を 0 にするため，データを中心化する：
   $$
   \bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i, \quad \tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}.
   $$
   中心化されたデータ行列を $\tilde{\mathbf{X}}$ とおく．

2. **共分散行列の構築**  
   中心化後のデータから共分散行列 $\mathbf{C}$ を求める：
   $$
   \mathbf{C} = \frac{1}{n} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \in \mathbb{R}^{d \times d}.
   $$

3. **固有値分解**  
   共分散行列に対して固有値分解を行い，固有ベクトル $\{\mathbf{w}_1, \dots, \mathbf{w}_\mathrm{d}\}$ と対応する固有値 $\{\lambda_1, \dots, \lambda_\mathrm{d}\}$ を求める．固有値は分散量に対応し，$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ の順に並べる．固有ベクトルは以下を満たす：
   $$
   \mathbf{C} \mathbf{w}_k = \lambda_k \mathbf{w}_k, \quad k=1,\dots,d.
   $$

4. **次元削減と主成分の構成**  
   上位 $m < d$ 個の固有ベクトル $\mathbf{W}_m = [\mathbf{w}_1, \dots, \mathbf{w}_m]$ を用いて，元のデータを $m$ 次元に射影する：
   $$
   \mathbf{z}_i = \mathbf{W}_m^\top \tilde{\mathbf{x}}_i \in \mathbb{R}^m.
   $$
   これにより得られる $\mathbf{z}_i$ は主成分と呼ばれる．

#### Oja則によるPCAの実行
主成分分析はOja則を応用することで神経回路上に実装できる．重みの変化量の期待値を取る．

$$
\begin{align}
\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t} &= \eta \left(\mathbf{x}y - y^2 \mathbf{w}\right)=\eta \left(\mathbf{x}\mathbf{x}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{x}\mathbf{x}^\top \mathbf{w}\right] \mathbf{w}\right)\\
\mathbb{E}\left[\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t}\right] &= \eta \left(\mathbf{C} \mathbf{w} - \left[\mathbf{w}^\top \mathbf{C} \mathbf{w}\right] \mathbf{w}\right)
\end{align}
$$

$\mathbf{C}:=\mathbb{E}[\mathbf{x}\mathbf{x}^\top]\in \mathbb{R}^{\mathrm{d}\times d}$とする．$\mathbf{x}$の平均が0の場合，$\mathbf{C}$は分散共分散行列である．$\mathbb{E}\left[\dfrac{\mathrm{d}\mathbf{w}}{\mathrm{d}t}\right]=0$となる$\mathbf{w}$が収束する固定点(fixed point)では次の式が成り立つ．

$$
\begin{equation}
\mathbf{C}\mathbf{w} = \lambda \mathbf{w}
\end{equation}
$$

これは固有値問題であり，$\lambda:=\mathbf{w}^\top \mathbf{C} \mathbf{w}$は固有値，$\mathbf{w}$は固有ベクトル(eigen vector)になる．

ここでサンプルサイズを$n$とし，$\mathbf{X} \in \mathbb{R}^{\mathrm{d}\times n}, \mathbf{y}=\mathbf{X}^\top\mathbf{w} \in \mathbb{R}^n$とする．標本平均で近似して$\mathbf{C}\simeq \mathbf{X}\mathbf{X}^\top$とする．この場合，

$$
\begin{align}
\mathbb{E}\left[\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t}\right] &\simeq \eta \left(\mathbf{X}\mathbf{X}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{X}\mathbf{X}^\top \mathbf{w}\right] \mathbf{w}\right)\\
&=\eta \left(\mathbf{X}\mathbf{y} - \left[\mathbf{y}^\top\mathbf{y}\right] \mathbf{w}\right)
\end{align}
$$

となる．

後のためにOja則においてネットワークが$q$個の複数出力を持つ場合を考えよう．重み行列を$\mathbf{W} \in \mathbb{R}^{q\times d}$, 出力を$\mathbf{y}=\mathbf{W}\mathbf{x} \in \mathbb{R}^{q}, \mathbf{Y}=\mathbf{W}\mathbf{X} \in \mathbb{R}^{q\times n}$とする．この場合の更新則は

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{W}}{\mathrm{d}t} = \eta \left(\mathbf{y}\mathbf{x}^\top - \mathrm{Diag}\left[\mathbf{y}\mathbf{y}^\top\right] \mathbf{W}\right)
\end{equation}
$$

となる．ただし，$\mathrm{Diag}(\cdot)$は行列の対角成分からなる対角行列を生み出す作用素である．

#### Sanger則
Oja則に複数の出力を持たせた場合であっても，出力が直交しないため，PCAの第1主成分しか求めることができない．**Sanger則** (Sanger's rule)，あるいは**一般化Hebb則** (generalized Hebbian algorithm; GHA)\footnote{あくまでSangerが「一般化」と呼んでいるだけで，Hebb則の一般化された形式ではない．} は，Oja則に**Gram–Schmi\mathrm{d}tの正規直交化法** (Gram–Schmi\mathrm{d}t orthonormalization) を組み合わせた学習則であり，次式で表される．

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{W}}{\mathrm{d}t} = \eta \left[\mathbf{y}\mathbf{x}^\top - \mathrm{LT}\left(\mathbf{y}\mathbf{y}^\top\right) \mathbf{W}\right]
\end{equation}
$$

$\mathrm{LT}(\cdot)$は行列の対角成分より上側の要素を0にした下三角行列(lower triangular matrix)を作り出す作用素である．Sanger則を用いればPCAの第2主成分以降も求めることができる．

#### 非負主成分分析によるグリッドパターンの創発
内側嗅内皮質(MEC)にある**グリッド細胞** (grid cells) は六角形格子状の発火パターンにより自己位置等を符号化するのに貢献している．この発火パターンを生み出すモデルは多数あるが，**場所細胞** (place cells) の発火パターンを**非負主成分分析** (nonnegative principal component analysis) で次元削減するとグリッド細胞のパターンが生まれるというモデルがある \citep{Dordek2016-ff}．非線形Hebb学習を用いてこのモデルを実装しよう．なお，同様のことは**非負値行列因子分解** (nonnegative matrix factorization; NMF) でも可能である．

関数`HebbianPCA`の`func`引数に非線形関数を渡すことで非線形Hebb学習は実現できる．

##### 場所細胞の発火パターン
まず，訓練データとなる場所細胞の発火パターンを人工的に作成する．場所細胞の発火パターンはガウス差分フィルタ (difference of Gaussians; DoG) で近似する．DoGは大きさの異なる2つのガウス関数の差分を取った関数であり，画像に適応すればband-passフィルタとして機能する．また，DoGは網膜神経節細胞等の受容野のON中心OFF周辺型受容野のモデルとしても用いられる．受容野中央では活動が大きく，その周辺では活動が抑制される，という特性を持つ．2次元のガウス関数とDoG関数を実装する．

Place cellの受容野をDoGに設定したが，これが無いと格子状の受容野は出現しない．path integrationをRNNで実行する場合も同様．一方で，DoGは場所細胞の受容野としては不適切である．

No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit 
<https://openreview.net/forum?id=mxi1xKzNFrb>

ToDo: 他のgrid cellsのモデルについて