## Hopfield モデル

連想記憶のとそのモデルの説明 (Asociatron)
相関の記憶モデル
Hopfieldモデルの構造，エネルギーと学習則，同期・非同期更新
エネルギー関数はなぜこのように設定されているか？
記憶容量
DAM

\citep{Hopfield1982-vu}で提案．始めは1と0の状態を取った．

Hopfieldモデルと呼ばれることが多いが，Amariの先駆的研究\citep{`Amari1972-fq`を踏まえAmari-Hopfieldモデルと呼ばれることもある．

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

ホップフィールドモデルにおけるエネルギー関数は，まずネットワークの各ニューロンの出力をスピン系にならい \(S_i\in\{+1,-1\}\) と置き，ニューロン間の結合強度を対称行列 \(W=(w_{ij})\)（ただし \(w_{ii}=0\)）と仮定するところから出発します。このとき，ネットワーク全体の「エネルギー」\(E\) を次式で定義します：  
\[
E(\mathbf{S}) \;=\; -\frac{1}{2}\sum_{i\neq j} w_{ij}\,S_i\,S_j \;-\;\sum_{i} b_i\,S_i,
\]
ここで \(\mathbf{S}=(S_1,\dots,S_N)\) は全ニューロンの状態ベクトル，\(b_i\) は各ニューロンにかかるバイアス項（外部磁場に相当）です。このエネルギー関数は，非同期更新則  
\[
S_i \;\longleftarrow\;\mathrm{sign}\Bigl(\sum_j w_{ij}S_j + b_i\Bigr)
\]
のたびに必ず減少（あるいは変化なし）となるように設計されており，その結果ネットワーク状態はエネルギーの局所最小点へと収束します。  

この定義はまさに統計物理学におけるイジング模型（Ising model）のハミルトニアン  
\[
H = -\sum_{i<j} J_{ij}\,s_i\,s_j \;-\;\sum_i h_i\,s_i
\]
と同型であり，ホップフィールド自身もこの類推をもとにモデルを提案しました。したがって，ホップフィールドモデルのエネルギー関数はイジング模型由来のハミルトニアンをそのまま神経回路の結合重みとニューロン状態に置き換えた形で定義されていると言えます。

イジング模型のハミルトニアン  
\[
H(\{s_i\}) \;=\; -\sum_{⟨i,j⟩}J_{ij}\,s_i\,s_j \;-\;\sum_i h_i\,s_i
\]
は、もともと厳密に「導出」されたものではなく、フェロ磁性を記述するための簡易化モデルとして仮定されたものです。  

1920年にレッツ（W. Lenz）が一次元格子上の二値スピン間相互作用を考えるモデルを提案し、1924年にイジング（E. Ising）がそれを解析したのが始まりです。相互作用 \(J_{ij}\) を隣接サイト同士に限定し、スピンを \(s_i=\pm1\) に固定するという仮定は、現実の強磁性体に対する多自由度スピン系（たとえばハイゼンベルク模型）から厳密に導かれたのではなく、計算可能かつ相転移の本質を捉える簡略化として置かれたものです。  

後に、ハイゼンベルク模型  
\[
\mathcal{H}=-\sum_{⟨i,j⟩}\bigl(J_x\,S_i^xS_j^x + J_y\,S_i^yS_j^y + J_z\,S_i^zS_j^z\bigr)
\]
の異方性極限 \(J_x,J_y\to0,\,J_z>0\) を考えることで、イジング模型が量子スピン系の一種として得られることが分かりました。しかしこのような「導出」は後付けの解釈であって、オリジナルのハミルトニアン自体は物理系から一義的に導かれたものではなく、あくまで相転移現象を扱うための経験的・統計力学的モデルとして設定されたものです。

ホップフィールドモデルのエネルギー関数  
\[
E(\mathbf{S}) = -\frac{1}{2}\sum_{i\neq j}w_{ij}S_iS_j - \sum_i b_iS_i
\]
は，単にイジング模型から借用した形をそのまま持ち込んだだけ、というよりも，「ネットワークの状態更新則が必ず単調に減少していく指標」を意図して構成されています。具体的には，各ニューロンを非同期・確定的に  
\[
S_i \;\longleftarrow\;\mathrm{sign}\Bigl(\sum_jw_{ij}S_j + b_i\Bigr)
\]
という更新で動かしたとき，その一回の更新でエネルギー \(E\) が必ず変化量  
\[
\Delta E = E(\dots,S_i^{\rm new},\dots)-E(\dots,S_i^{\rm old},\dots)
= -\bigl(\sum_jw_{ij}S_j + b_i\bigr)\bigl(S_i^{\rm new}-S_i^{\rm old}\bigr)
\]
のように負（またはゼロ）になるように設計されています。すなわち「エネルギー関数」はネットワークの動きを安定化し，漸近的に準安定点（局所最小点）へと収束させるためのリャプノフ関数（Lyapunov function）そのものなのです。

この構成が可能となる要件は，重み行列 \(W=(w_{ij})\) が対称（\(w_{ij}=w_{ji}\)）でかつ対角要素 \(w_{ii}=0\) を満たすことにあります。対称性があることで，あるニューロン \(i\) を更新した際に「ほかのニューロンから受け取る影響」が一意に定まり，それがエネルギーの減少方向に一致します。逆に非対称な結合を許すと，更新時にエネルギーが増減を繰り返し，ネットワークは発散したり周期振動を起こしたりしてしまいます。

さらに，学習規則にヘッブ則（Hebb’s rule）を用いると，記憶したいパターンがエネルギーの局所最小点として配置され，入力パターンから最も近い記憶パターンへマッピングされる「連想メモリ」として機能します。したがって，このエネルギー関数は単なる物理的なハミルトニアンを模倣しただけではなく，「動的システムとして収束性を保証しつつ，望む安定状態（記憶パターン）をエネルギー最小点として刻み込む」ための数学的装置であると解釈できます。

ネットワーク状態の収束性を担保するために，ホップフィールドモデルではエネルギー関数  
\[
E(\mathbf S)
=-\tfrac12\sum_{i\neq j}w_{ij}S_iS_j-\sum_i b_iS_i
\]
が「リャプノフ関数（Lyapunov function）」として働くことを示します。リャプノフ関数とは，状態変化のたびに常に（または非増加に）値が減少し，最低値をもつことによって系が安定な定常点へ収束することを保証するスカラー関数です。以下にその性質を詳細に示します。

まず，各時刻 \(t\) のニューロン状態を \(\mathbf S^{(t)}=(S_1^{(t)},\dots,S_N^{(t)})\) とし，あるひとつのユニット \(i\) を選んで非同期に更新する更新則を  
\[
S_i^{(t+1)}=\mathrm{sign}\!\Bigl(h_i^{(t)}\Bigr),
\quad
h_i^{(t)}=\sum_{j}w_{ij}S_j^{(t)}+b_i,
\]
それ以外のユニットはそのまま \(S_j^{(t+1)}=S_j^{(t)}\) とします。このときのエネルギー変化量を計算すると，

\[
\begin{aligned}
\Delta E
&=E\bigl(S_1^{(t+1)},\dots,S_i^{(t+1)},\dots\bigr)
 -E\bigl(S_1^{(t)},\dots,S_i^{(t)},\dots\bigr)\\
&=\Bigl[-S_i^{(t+1)}\bigl(\sum_{j\neq i}w_{ij}S_j^{(t)}+b_i\bigr)\Bigr]
 -\Bigl[-S_i^{(t)}\bigl(\sum_{j\neq i}w_{ij}S_j^{(t)}+b_i\bigr)\Bigr]\\
&= -\bigl(S_i^{(t+1)}-S_i^{(t)}\bigr)\,h_i^{(t)}.
\end{aligned}
\]
しかし，更新則より \(S_i^{(t+1)}\) は必ず \(h_i^{(t)}\) と同符号を持つため，積 \((S_i^{(t+1)}-S_i^{(t)})\,h_i^{(t)}\) は非負になり，したがって  
\[
\Delta E \;=\;-\,\bigl(S_i^{(t+1)}-S_i^{(t)}\bigr)\,h_i^{(t)}\;\le\;0
\]
が成立します。すなわち，どのユニットを更新してもエネルギーは必ず減少（あるいは変化なし）となるのです。

この性質が成り立つための鍵は重み行列の「対称性」 \(w_{ij}=w_{ji}\) と「自己結合の不在」 \(w_{ii}=0\) にあります。もしこれらが破られると，上記の差分計算で打ち消しあうべき項が残存し，更新によりエネルギーが増加する場合も生じ、収束性が失われます。

さらに，エネルギー関数は結合強度やバイアスが有限である限り下限をもちます。具体的には各項が有限の有界和になるため，エネルギーはある最小値 \(E_{\min}\) のまわりで振動しながら必ず停留点（すなわち局所最小点）に定着します。これにより，状態遷移は必ずある固定点へと漸近し，発散や発振を起こしません。

以上の理由から，ホップフィールドモデルのエネルギー関数は離散時間ダイナミクスに対するリャプノフ関数の要件（①各更新で非増加，②下限をもつ）を満たし，ネットワークの安定収束性を理論的に保証する数学的装置として解釈できます。

ご指摘のとおりです。ホップフィールドモデルにおけるエネルギー関数の相互作用項  
\[
-\frac{1}{2}\sum_{i\neq j}w_{ij}S_iS_j
\]
は「結合の強さ \(w_{ij}\) とニューロン活動の相関 \(S_iS_j\) の積」を評価しています。具体的には、あるペア \(i,j\) について

1. **結合 \(w_{ij}>0\)（興奮性）かつ \(S_iS_j=+1\)（両方が同時に発火／非発火）** ならば  
   \[
   w_{ij}S_iS_j > 0
   \;\Longrightarrow\;
   -\tfrac12\,w_{ij}S_iS_j < 0
   \]
   よってエネルギーが減少し、その状態が「安定」になります。これは「興奮性シナプスは，両ユニットが同時に活動するときにネットワークを安定化する」ことを意味します。

2. **結合 \(w_{ij}<0\)（抑制性）かつ \(S_iS_j=-1\)（片方が発火し，他方が抑制）** ならば  
   \[
   w_{ij}S_iS_j > 0
   \;\Longrightarrow\;
   -\tfrac12\,w_{ij}S_iS_j < 0
   \]
   となり，抑制結合は「異なる活動パターン」を安定化します。

逆に，結合の符号と活動の相関が不一致（例えば興奮性結合なのに一方だけ発火）だと，その項はエネルギーを増大させ，その状態はネットワークが避けようとする「非安定」なパターンになります。  
このように、\(\,w_{ij}S_iS_j>0\) となる局所条件を満たすほどエネルギーが低くなる仕組みは、「重みの符号とニューロン活動の一致度」を測り、ネットワークが学習した相関構造を安定なアトラクターとして再現する〈局所的な整合性ルール〉であると解釈できます。


**エネルギー地形解析** (Energy landscape analysis)

[1] T. Ezaki, T. Watanabe, M. Ohzeki, and N. Masuda, "Energy landscape analysis of neuroimaging data", Phil. Trans. R. Soc. A 375, 20160287 (2017).

[2] T. Watanabe, N. Masuda, F. Megumi, R. Kanai, G. Rees, "Energy Landscape and dynamics of brain activity during human bistable perception", Nat. Commun. 5, 4765 (2014).

[3] T. Watanabe, S. Hirose, H. Wada, Y. Imai, T. Machida, I Shirouzu, Y. Miyashita, N. Masuda, "Energy Landscapes of resting-state brain networks", Front. Neuroinform. 8, 12 (2014).