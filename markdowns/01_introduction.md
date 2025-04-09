# 第1章：はじめに
## 本書の目的と構成
### 神経科学における計算論
本書では神経科学における数理モデルを主として取り扱う．初めに神経科学におけるモデルの役割について触れておこう．まず，神経科学の目標は端的に言えば「脳神経系を理解する」ことにある．神経科学に限らず，種々の学問分野においては実験と理論の2本柱で，対象とする現象や物質の理解が進められる．ここで実験は調査等も踏まえ実データを取る行為とする．理論の役割は複数あり，実験結果の抽象化，仮説の提供，現象の予測等である \citep{Blohm2020-vc}．

「脳神経系を理解する」ということに関して，その定義は研究者により様々である．ここでは脳の計算処理に関する理論的理解を進めるための1つの方法として Marrの3レベル (Marr's Three Levels) を紹介する \citep{Marr1982-wk}．Marrの3レベルは視覚系における計算処理の理解を主としていたが，他でも適用可能である．3レベルとは(1)計算理論 (computational theory), (2) 表現・アルゴリズム (representation and algorithm), (3)実装 (implementation) であり，それぞれの段階での議論や理解を行う．(1)では脳の目的関数とそれを用いた最適化問題の設定を行う．(2)では(1)を実現するための表現およびアルゴリズムを解明する．(3)では(1,2)を神経回路・ハードウェア上で実装する方法を解明することを目標とし，平易には「脳」を作って理解すると言い換えることもできる\footnote{ここでの「作る」は計算機等でシミュレーションするという意味であり，脳オルガノイド (brain organoid) を作成するなどの意味ではない}．本書ではこの(3)を重視し，読者が自らの手で理論を検証し，数値計算による結果を再現できることを目標とした．また，本書は数式をプログラミングのコードに変換する具体例集としての役割も持っている．

モデルの中でも，本書では機械学習に関連する内容が多数登場する．これは神経科学と機械学習は互いに影響を及ぼし合ってきたためである \citep{Hassabis2017-zm}. 
神経科学から機械学習への応用は例えば，ニューラルネットワーク，記憶モデル，注意モデルなどがある．逆に機械学習から神経科学への応用は強化学習，運動制御，ベイズ脳仮説などが挙げられる．

### 本書の構成

第1章では，Julia言語の使用法と用いる数学について簡単に説明する．

まとまりを重視するため，本書では発火率モデルおよびニューラルネットワークについて説明した後に，スパイキングモデルおよびスパイキングニューラルネットワークの説明を行う．

第2章から第5章まではニューラルネットワークとその学習

第2章では，まず神経細胞の簡単な生理学について説明する．発火率モデルを説明したのち，局所学習則によって訓練されるネットワークの説明を行う．第3章では，同じく局所学習則ではあるが，ネットワーク全体のエネルギーを下げることを目的としたエネルギーベースモデルと呼ばれる枠組みのネットワークについて説明をする．第4章では，誤差逆伝播法に基づいたニューラルネットワークを説明し，貢献度分配問題の生理学的な解決策について説明をする．第5章では，さらに再起型ニューラルネットワークを説明し，経時的貢献度分配問題について説明を行う．

第2部 スパイキングニューラルネットワークとその学習

第6・7章ではスパイキングニューラルネットワークとその学習について取り扱う．第6章ではネットワークレベルの話から再び神経細胞とシナプスに回帰する．

細胞からネットワークへの流れを全体として保つことも考えたが，実装上のまとまりを優先してこのような流れとした．

第6章ではスパイキングニューラルネットワークの章では，初めにスパイキングニューロンのモデルについて説明を行う．次に，シナプスのダイナミクスについて説明を補いながらモデルを構築する．ランダムネットワークを構築した後に，ネットワークの学習則を説明する．

第8章から12章は上記以外の内容について各論的に説明を行う．

- リザバーコンピューティングの章では，リザバーコンピューティングと呼ばれる枠組みのネットワークについて，発火率・スパイキングモデルの双方をまとめて紹介する．カオスの縁についても触れる．
- ベイズ推論の章では，神経回路網により，如何にして確率計算を行うかを説明する．
- 運動学習では，最適制御問題の解決策について説明する．
- 強化学習では，強化学習の基本的事項の説明と，大脳基底核との関連性について説明する．
- 最後の章は補足的な話題であり，ネットワーク・形態学・グリアについて説明を行う．

https://www.sciencedirect.com/science/article/pii/S0364021387800253

## Julia言語の使用法
### Julia言語の特徴
Julia言語は

本書を執筆するにあたり，なぜJulia言語を選択したかというのにはいくつか理由がある．

JuliaはJIT（Just-In-Time）コンパイルを用いており

JITコンパイラ

実行速度が高速であること．
ライセンスフリーであり，無料で使用できること．
線型代数演算が簡便に書けること．
Unicodeを使用できるため，疑似コードに近いコードを書けること．

他の言語の候補として，MATLAB, Pythonが挙げられた．MATLABは神経科学分野で根強く使用される言語であり，線型代数計算の記述が簡便である．なお，線型代数演算の記法に関してはJuliaはMATLABを参考に構築されたため，ほぼ同様に記述することができる．また，MATLABを使用するには有償ライセンスが必要である．ただし，互換性を持ったフリーソフトウェアであるOctaveが存在することは明記しておく．

Pythonは機械学習等の豊富なライブラリと書きやすさから広く利用されている言語である．ただし，numpyを用いないと高速な処理を書けない場合が多く，ナイーブな実装では実行速度が低下してしまう問題がある．線型代数計算も簡便に書くことができず，数式をコードに変換する際の手間が増えるという問題がある．

多重ディスパッチ（multiple dispatch）があることはJulia言語の大きな特徴である．

### Julia言語のインストール方法

Julia (\url{https://julialang.org/}) に

juliaup (\url{https://github.com/JuliaLang/juliaup}) でバージョン管理

Google ColabにおいてPythonやRに並んでJuliaを選択して使用することが可能となっている．

### Julia言語の基本構文

https://docs.julialang.org/en/v1/manual/noteworthy-differences/

## 基礎的数学とJuliaでの記法
数式の表記法も兼ねて，本書で使用する数学的内容を整理する．

### 表記法
本書では次のような記号表記を用いる．
- 実数全体を$\mathbb{R}$, 複素数全体は$\mathbb{C}$と表記する．
- スカラーは小文字・斜体で $x$ のように表記する．
- ベクトルは小文字・立体・太字で $\mathbf{x}$ のように表記し，列ベクトル (縦ベクトル) として扱う．
- 行列は大文字・立体・太字で $\mathbf{X}$ のように表記する．
- $n\times 1$の実ベクトルの集合を $\mathbb{R}^n$, $n\times m$ の実行列の集合を $\mathbb{R}^{n\times m}$と表記する．
- 行列 $\mathbf{X}$ の置換は $\mathbf{X}^\top$と表記する．ベクトルの要素を表す場合は $\mathbf{x} = (x_1, x_2,\cdots, x_n)^\top$のように表記する．
- 単位行列を $\mathbf{I}$ と表記する．
- ゼロベクトルは $\mathbf{0}$ , 要素が全て1のベクトルは $\mathbf{1}$ と表記する．  
- $e$を自然対数の底とし，指数関数を $e^x=\exp(x)$と表記する．また，自然対数を $\ln(x)$と表記する．
- 定義を$\coloneqq$を用いて行う．例えば，$f(x)\coloneqq2x$は$f(x)$という関数を$2x$として定義するという意味である．定義する対象が右側である場合は，$\eqqcolon$を用いる．
- 平均 $\mu$, 標準偏差 $\sigma$ の正規分布を $\mathcal{N}(\mu, \sigma^2)$ と表記する．

### 線形代数

**線形代数 (Linear Algebra)** は、ベクトルや行列といった線形構造を持つ対象の性質を解析する数学の分野であり、現代のあらゆる数学・工学・情報科学の基礎をなしている。線形代数の中心的な対象は、**ベクトル空間**、**線形写像**、およびそれらの表現である**行列**である。

まず、**ベクトル空間 (vector space)** とは、スカラー体（通常は実数 $\mathbb{R}$ または複素数 $\mathbb{C}$）に対して定義された加法とスカラー倍という2つの演算に関して閉じている集合である。たとえば $\mathbb{R}^n$ は、$n$ 個の実数からなるベクトル全体の集合であり、典型的なベクトル空間の例である。任意のベクトル $\mathbf{v}, \mathbf{w} \in \mathbb{R}^n$ とスカラー $\alpha \in \mathbb{R}$ に対して、

\[
\alpha\mathbf{v} + \mathbf{w} \in \mathbb{R}^n
\]

が成り立つ。

**線形写像 (linear transformation)** とは、ベクトル空間からベクトル空間への写像 $T: V \to W$ であり、加法とスカラー倍に対して線形性を持つもの、すなわち

\[
T(\alpha \mathbf{v} + \beta \mathbf{w}) = \alpha T(\mathbf{v}) + \beta T(\mathbf{w})
\]

が任意の $\mathbf{v}, \mathbf{w} \in V$ とスカラー $\alpha, \beta$ に対して成り立つ写像である。

このような線形写像は、基底を定めることで**行列 (matrix)** によって表現できる。たとえば、$n$ 次元から $m$ 次元への線形写像は、$m \times n$ の行列 $A$ を用いて

\[
\mathbf{y} = A\mathbf{x}
\]

という形で記述される。ここで $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{y} \in \mathbb{R}^m$ はそれぞれ入力および出力ベクトルである。

行列の基本演算には以下が含まれる：

- **行列の積**：$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$ に対し、$AB \in \mathbb{R}^{m \times p}$ を定義。
- **転置**：$A^\top$ は行列 $A$ の行と列を交換したもの。
- **逆行列**：$A \in \mathbb{R}^{n \times n}$ が正則（可逆）であれば、$A^{-1}$ が存在し $AA^{-1} = A^{-1}A = I$ を満たす。
- **行列式**（determinant）：$\det A$ は正方行列 $A$ に対するスカラー量で、行列の体積のスケーリング率や可逆性の指標となる。

特に、線形代数の重要な応用の一つは**線形方程式系**の解法である。$A\mathbf{x} = \mathbf{b}$ の形をした方程式において、$A$ の逆行列が存在するならば、その解は

\[
\mathbf{x} = A^{-1}\mathbf{b}
\]

と求められる。

また、行列の固有値問題も重要である。ある正方行列 $A$ に対し、スカラー $\lambda$ およびベクトル $\mathbf{v} \ne \mathbf{0}$ が

\[
A\mathbf{v} = \lambda \mathbf{v}
\]

を満たすとき、$\lambda$ は $A$ の**固有値 (eigenvalue)**、$\mathbf{v}$ は**固有ベクトル (eigenvector)** と呼ばれる。固有値分解や対角化は、線形変換の構造解析や行列の関数（例：指数関数）を考える際に中心的な役割を果たす。

matrix cookbookに詳しいが，

### 微分方程式
微分方程式はある関数とそれを微分した導関数の関係式であり，関数の特定の変数に対する変化を記述することができる．まず，1階線形微分方程式を例として見てみよう．

$$
\begin{equation}
\frac{dx(t)}{dt}=a_c x(t)+b_c u(t)
\end{equation}
$$

状態変数 $x(t)$は，時間$t$に対する関数である．

添え字の$c$は連続 (continuous) を意味するが，これは後で離散化する際に区別するためである．この方程式においては$b_c=0$の場合を**同次方程式**, $b_c\neq 0$の場合を**非同次方程式**という．

#### 微分方程式の解
微分方程式を解くとは$x(t)$のような関数の具体的な式を求めることである．上式の解は

$$
\begin{equation}
x(t)=e^{a_c t}x(0)+\int_0^t e^{a_c (t-\tau)}b_c u(\tau) d\tau
\end{equation}
$$

として与えられる．微分方程式を解く手法は様々で，それぞれの方程式について適切な手法を選択する．本書ではLaplace変換を多用するが，細かい説明は付録にて行う．

#### 連立線形微分方程式
$n$個の微分方程式

連立線形微分方程式という．これをベクトル，行列を用いて

時不変 (time-invariant) の定数行列を$\mathbf{A}_c \in \mathbb{R}^{n\times n}, \mathbf{B}_c \in \mathbb{R}^{n\times m}$, 状態ベクトルを$\mathbf{x}(t)\in\mathbb{R}^n$, 入力ベクトルを$\mathbf{u}(t)\in\mathbb{R}^m$とする．

$$
\begin{equation}
\frac{d\mathbf{x}(t)}{dt} = \mathbf{A}_c\mathbf{x}(t) + \mathbf{B}_c\mathbf{u}(t)
\end{equation}
$$

解は

$$
\begin{equation}
\mathbf{x}(t)=e^{t\mathbf{A}_c}\mathbf{x}(0)+\int_0^t e^{(t-\tau)\mathbf{A}_c}\mathbf{B}_c\mathbf{u}(\tau) d\tau
\end{equation}
$$

#### Laplace変換
**Laplace変換**は、与えられた時間領域の関数 $f(t)$ を複素数変数 $s$ の関数 $F(s)$ に写像する積分変換である。特に、線形微分方程式の解析や制御工学において非常に有効な手法であり、Fourier変換と密接な関係をもつ。

Laplace変換は、実時間領域 $t \ge 0$ 上で定義された関数 $f(t)$ に対して、以下のように定義される：

\[
F(s) := \mathscr{L}(f(t)) = \int_0^{\infty} f(t)\, e^{-st} dt
\]

ここで $s \in \mathbb{C}$ は複素数変数であり、通常は $s = \sigma + i\omega$ の形をとる。変換核 $e^{-st}$ を掛けて積分することにより、関数 $f(t)$ の無限大での振る舞いを抑制し、積分を収束させる効果を持つ。特に、$f(t)$ が指数関数的増加を含む場合でも、$e^{-st}$ による減衰によってその成分を抑えることが可能となる。

Laplace変換の大きな利点の一つは、**微分演算を代数演算に変換できる**という性質にある。すなわち、$f(t)$ の微分 $\frac{d}{dt}f(t)$ に対するLaplace変換は、次のように与えられる：

\[
\mathscr{L}\left(\frac{df}{dt}\right) = sF(s) - f(0)
\]

この性質により、常微分方程式はLaplace変換の下で代数方程式に変換され、解の導出が容易となる。初期値を含んだ微分方程式を直接的に解くことができるため、初期値問題への応用にも適している。

さらに、Laplace変換には線形性：

\[
\mathscr{L}(af(t) + bg(t)) = a\mathscr{L}(f(t)) + b\mathscr{L}(g(t))
\]

および畳み込みに関する定理：

\[
\mathscr{L}(f * g)(t) = F(s)G(s)
\]

など、多くの有用な性質がある。これにより、時間領域での複雑な演算が周波数領域で簡単な演算として扱えるようになる．

#### 1階線形行列微分方程式の解
時不変 (time-invariant) の定数行列を$\mathbf{A} \in \mathbb{R}^{n\times n}, \mathbf{B} \in \mathbb{R}^{n\times m}$, 状態ベクトルを$\mathbf{x}(t)\in\mathbb{R}^n$, 入力ベクトルを$\mathbf{u}(t)\in\mathbb{R}^m$とする．

$$
\begin{equation}
\frac{d\mathbf{x}(t)}{dt} = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)
\end{equation}
$$

この線形行列微分方程式をLaplace変換 $\mathcal{L}$を用いて解こう．$\boldsymbol{X}(s) := \mathcal{L}(\mathbf{x}(t)), \boldsymbol{U}(s) := \mathcal{L}(\mathbf{u}(t))$とすると，

$$
\begin{align}
s\boldsymbol{X}(s) - \mathbf{x}(0) &= \mathbf{A}\boldsymbol{X}(s)+ \mathbf{B}\boldsymbol{U}(s)\\
(s\mathbf{I} - \mathbf{A}) \boldsymbol{X}(s) &= \mathbf{x}(0) + \mathbf{B}\boldsymbol{U}(s)\\
\boldsymbol{X}(s) &= (s\mathbf{I} - \mathbf{A})^{-1}(\mathbf{x}(0) + \mathbf{B}\boldsymbol{U}(s))\\
\end{align}
$$

行列指数関数 (matrix exponential)は

$$
\begin{equation}
e^\mathbf{A} = \exp(\mathbf{A}) := \sum_{k=0}^\infty \frac{1}{k!}\mathbf{A}^k = \mathbf{I}+\mathbf{A}+\frac{\mathbf{A}^2}{2!}+\cdots
\end{equation}
$$

として定義される．

天下り的だが，

$$
\begin{align}
\mathcal{L}(e^{at})&=\frac{1}{s-a}\\
\mathcal{L}(e^{t\mathbf{A}})&=(s\mathbf{I} - \mathbf{A})^{-1}\\
\end{align}
$$

となる．よって

$$
\begin{align}
\boldsymbol{X}(s) &= (s\mathbf{I} - \mathbf{A})^{-1}(\mathbf{x}(0) + \mathbf{B}\boldsymbol{U}(s))\\
&= (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{x}(0) + (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B}\boldsymbol{U}(s)\\
\mathbf{x}(t)&=e^{t\mathbf{A}}\mathbf{x}(0)+\int_0^t e^{(t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) d\tau
\end{align}
$$

となる．最後の式は両辺を逆Laplace変換した．ここで，$\mathcal{L}^{-1}(F(s)G(s))=\int_0^tf(\tau)g(t-\tau)d\tau$であることを用いた．区間$[t, t+\Delta t]$において入力$\mathbf{u}(t)$が一定であると仮定すると，

$$
\begin{align}
\mathbf{x}(t+\Delta t)&=e^{(t+\Delta t)\mathbf{A}}\mathbf{x}(0)+\int_0^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) d\tau\\
&=e^{\Delta t\mathbf{A}}e^{t\mathbf{A}}\mathbf{x}(0)+e^{\Delta t\mathbf{A}}\int_0^{t} e^{(t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) d\tau + \int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) d\tau\\
&\approx \underbrace{e^{\Delta t\mathbf{A}}}_{=: \mathbf{A}_d}\mathbf{x}(t)+\underbrace{\left[\int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}} d\tau\right] \mathbf{B}}_{=: \mathbf{B}_d}\mathbf{u}(t)\\
&=\mathbf{A}_d\mathbf{x}(t)+\mathbf{B}_d\mathbf{u}(t)\\
\end{align}
$$

となる．添え字の$d$は離散化(discretization)を意味する．$\mathbf{A}_c$が正則行列の場合，

$$
\begin{align}
\mathbf{B}_d &= \left[\int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}} d\tau\right] \mathbf{B}\\
&=\mathbf{A}^{-1}\left[e^{\Delta t \mathbf{A}}-\mathbf{I}\right]\mathbf{B}
\end{align}
$$

が成り立つ．

### 確率論
確率論の基本的な対象は**確率分布 (probability distribution)** である。確率分布は、ある確率変数がどのような値をどの程度の確率でとるかを定量的に記述するものである。確率変数 $x$ は、離散的あるいは連続的な値をとる場合があり、それぞれに応じて確率分布の定義も異なる。

離散的な場合、確率分布は**確率質量関数** (probability mass function; PMF) により定義され、任意の値 $x$ に対して $p(x)$ はその値が観測される確率を与える。このとき、全ての確率の総和は 1 に等しくなければならない：

\[
\sum_x p(x) = 1
\]

この代表例として**ポアソン分布 (Poisson distribution)** がある。ポアソン分布は、ある固定時間・空間内における稀な離散事象の発生回数をモデル化するものであり、以下のように定義される：

\[
p(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0,1,2,\dots
\]

ここで $\lambda > 0$ は単位時間（または空間）あたりの平均発生回数を表す。この分布は事象が独立かつ一定の発生率で起きると仮定する場面で用いられる。

一方、連続的な場合には**確率密度関数** (probability density function; PDF) を用いて定義される。確率密度関数 $p(x)$ は特定の値における確率そのものではなく、ある範囲に入る確率を積分によって与える関数である。たとえば、区間 $[a,b]$ における確率は次のように表される：

\[
\mathbb{P}(a \leq x \leq b) = \int_a^b p(x)\,dx
\]

確率密度関数もまた、定義域全体にわたる積分が 1 でなければならない：

\[
\int p(x)\,dx = 1
\]

この典型例として**正規分布 (normal distribution)** が挙げられる。正規分布は、多くの自然現象や測定誤差の分布を記述するのに適しており、平均 $\mu$、分散 $\sigma^2$ をパラメータとして次のように定義される：

\[
p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\]

この分布は平均値 $\mu$ を中心に左右対称であり、確率変数の分布が「中央に集まり、端にいくほど稀になる」という性質を持つ。特に $\mu = 0$, $\sigma^2 = 1$ の場合は標準正規分布と呼ばれる．また，正規分布の概念は一変数の場合に限らず、多次元の確率変数にも拡張される。これが**多変量正規分布 (multivariate normal distribution)** であり、ベクトル値の確率変数がとる値の分布を記述する。

$d$ 次元の確率変数 $\mathbf{x} \in \mathbb{R}^d$ が平均ベクトル $\boldsymbol{\mu} \in \mathbb{R}^d$、共分散行列 $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ をもつ多変量正規分布に従うとき、その確率密度関数は以下のように定義される：

\[
p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma})^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
\]

ここで、$\mathbf{x}$ は $d$ 次元の確率変数ベクトル、$\boldsymbol{\mu}$ は平均ベクトルであり、各成分が $\mathbf{x}$ の平均値，$\boldsymbol{\Sigma}$ は対象対称かつ正定値な共分散行列であり、成分 $\Sigma_{ij}$ は $\mathrm{Cov}(x_i, x_j)$ を表す．

この分布は、各変数が正規分布に従い、かつそれらの間の線形な関係（共分散）もモデル化できる点で、非常に広範に用いられる。特に共分散行列が対角行列のとき、すなわち変数間が独立な場合には、各変数は独立な一変量正規分布に従う。


確率論においては、不確実性を定量的に扱うための基本的な概念がいくつか存在する。以下では、期待値、情報量、エントロピー、Kullback-Leibler情報量、そして相互情報量について簡単に説明を行う。

まず、**期待値 (Expectation)** は、確率変数 $x$ に関する関数 $f(x)$ の平均値を、$x$ の確率分布 $p(x)$ に基づいて計算する操作である。連続値の場合、期待値は次のように定義される。

\[
\mathbb{E}_{x\sim p(x)}\left[f(x)\right] := \int f(x)p(x)\,dx
\]

ここで $x \sim p(x)$ は、$x$ が分布 $p(x)$ に従うことを表す。文脈が明確な場合には、簡略に $\mathbb{E}_{p(x)}[f(x)]$ や $\mathbb{E}[f(x)]$ と表記する。

次に、**情報量 (Information)** は、ある特定の事象 $x$ の出現がどれほどの「驚き」や「情報」をもたらすかを定量化するものである。情報理論の創始者であるShannon (1948) によって導入された。出現確率が低い事象ほど、多くの情報を含むと考えられる。情報量は次のように定義される。

\[
\mathbb{I}(x) := \ln\left(\frac{1}{p(x)}\right) = -\ln p(x)
\]

**エントロピー (Entropy)** は、確率変数の持つ平均的な不確実性、すなわち平均情報量を表す。離散的な場合には和を、連続的な場合には積分を用いて定義されるが、ここでは連続的な場合を考える。エントロピーは以下のように定義される。

\[
\mathbb{H}(x) := \mathbb{E}[-\ln p(x)] = -\int p(x) \ln p(x)\,dx
\]

また、条件付きエントロピー $\mathbb{H}(x|y)$ は、$y$ が与えられたときの $x$ の不確実性を測る指標であり、次のように定義される。

\[
\mathbb{H}(x \vert y) := \mathbb{E}_{x,y}[-\ln p(x \vert y)]
\]

この期待値は、$p(x,y)$ に基づいて計算される。

次に、**Kullback-Leibler情報量 (KL divergence)** は、ある確率分布 $p(x)$ と別の分布 $q(x)$ の間の「距離」あるいは「ずれ」を測る尺度である。対称性は持たないため、厳密には距離ではないが、情報理論や機械学習において極めて重要な概念である。KLダイバージェンスは以下のように定義される。

\[
\begin{aligned}
D_{\text{KL}}(p(x)\Vert q(x)) &:= \int p(x) \ln \frac{p(x)}{q(x)} dx \\
&= \int p(x) \ln p(x)\,dx - \int p(x) \ln q(x)\,dx \\
&= \mathbb{E}_{x\sim p(x)}[\ln p(x)] - \mathbb{E}_{x\sim p(x)}[\ln q(x)] \\
&= -\mathbb{H}(x) - \mathbb{E}_{x\sim p(x)}[\ln q(x)]
\end{aligned}
\]

最後に、**相互情報量 (Mutual Information)** は、二つの確率変数 $x$ と $y$ の間にどれほどの情報的関連性があるか、すなわち $y$ を知ることによって $x$ の不確実性がどれほど減少するかを定量化する。相互情報量は、エントロピーの差として次のように定義される。

\[
\mathbb{I}(x;y) := \mathbb{H}(x) - \mathbb{H}(x\vert y)
\]

これはまた、対称的な形でも書ける。

\[
\mathbb{I}(x;y) = \mathbb{H}(x) + \mathbb{H}(y) - \mathbb{H}(x,y)
\]

あるいは、確率分布の比を使って次のようにも表現される。

\[
\mathbb{I}(x;y) = \int p(x,y) \ln \frac{p(x,y)}{p(x)p(y)} dxdy
\]

この表現は、相互情報量が、$p(x,y)$ と $p(x)p(y)$ のKLダイバージェンスであることを示しており、すなわち独立であれば情報共有はゼロであることを意味する。

## 学習に関する基礎的概念
本書のテーマの1つとして「学習」が挙げられる．
神経科学における「学習」と機械学習における「学習」はやや異なるが，ここで両者における学習を定義しておく．

神経科学の学習は

共通する点として，過去の経験に基づいて，将来の行動や出力を改善するためにシステムを変化させる，という点で共通している．システムの

システムのパラメータが変化する

神経科学：シナプス強度（重み）が変化する。

機械学習：ネットワークの重みやバイアスなどのパラメータが更新される。

異なる点として，

神経科学のモデルに機械学習

### モデルと学習・予測
**機械学習** (machine learning) における**モデル** (model) とは，2つの集合 $\mathcal{X}, \mathcal{Y}$ を仮定した際に，入力 $x\in \mathcal{X}$ を出力 $y\in \mathcal{Y}$ に変換する関数 (写像) $f: x \to y$ あるいは条件付き確率分布 $p(y|x)$ を意味する．モデルは内部に媒介変数あるいはパラメータ (parameter) $\theta$ を持ち，$\mathcal{Y}$ を設定した後に $y=f(x; \theta)$ あるいは $p(y|x; \theta)$ を満たすように $\theta$ を更新する．この過程を**学習** (learning) あるいは**訓練** (training) と呼ぶ．学習後のパラメータ $\theta^*$を用い，$x$が与えられた際の $y$ の推定値 $\hat{y}$ を $\hat{y}=f(x; \theta^*)$ あるいは $p(y|x; \theta^*)$ から取得することを**予測** (prediction) と呼ぶ．推定値の取得の方法としてはサンプリング $\hat{y}\sim p(y|x; \theta^*)$ や $\hat{y}=\textrm{argmax}\ p(y|x; \theta^{*})$などが考えられる．学習の際に用いられるデータを訓練データ (training data) と呼び，学習後のモデルの予測精度の評価に用いるデータを評価データ (test data) と呼ぶ．

$y$が既知の場合は$D=\{(x,y)\}$は教師付きデータ ($y$がラベルの場合はラベル付きデータ) と呼ばれ，$x$ と $y$ の対応関係を学習する過程を教師あり学習 (supervised learning) と呼ぶ．$y$が未知の場合，$D=\{x\}$はラベルなしデータと呼ばれ，これのみでモデルを学習する過程を教師なし学習 (unsupervised learning) と呼ぶ．この2つの学習の派生として，ラベルあり・なしデータを併用する半教師あり学習 (semi-supervised learning), 教師なし学習の一種であり，入力データの部分集合から他の部分集合を予測する自己教師あり学習 (self-supervised learning) などが存在する．この他の学習手法として強化学習 (reinforcement learning) があり，第11章で詳しく説明を行う．強化学習では環境の中で行動するエージェントを仮定し，状態に応じて多くの報酬を得るための行動を学習することが目的である．

### 回帰と分類

\citep{2015-yz}.

### 識別モデル・生成モデル

オンライン・オフライン学習

### 線形回帰
**線形回帰モデル**（linear regression）は、与えられた説明変数（explanatory variable）$\mathbf{x}$ に基づいて、目的変数（objective variable）$y$ を線形に予測することを目的とする。

説明変数の次元が $p$ であるとき、線形回帰モデルは次のように表される：

$$
\begin{equation}
y = w_0 + w_1x_1 + \cdots + w_px_p + \varepsilon = w_0 + \sum_{j=1}^p w_j x_j + \varepsilon
\end{equation}
$$

ここで $w_0$ は切片（バイアス項）、$w_1, \dots, w_p$ は各説明変数に対する重み、$\varepsilon$ は誤差項を表す。$p = 1$ の場合を**単回帰**（*simple regression*）、$p > 1$ の場合を**重回帰**（*multiple regression*）と呼ぶ。

#### 回帰モデルの行列表現

$ n $ 個の観測データからなるデータセット $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ を考える。ここで $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_p^{(i)}]^\top \in \mathbb{R}^p$ は $i$ 番目の説明変数ベクトル、$y^{(i)} \in \mathbb{R}$ は対応する目的変数の値である。なお、添字 $(i)$ は観測値を表し、添字のない $x_j, w_j$ などはモデル内の変数を指すことに注意する。

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

ここで $\mathbf{X}$ は**計画行列**（design matrix）、$\boldsymbol{\varepsilon}$ は誤差ベクトルである。特に、$\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$、すなわち各誤差成分が独立な平均0・分散 $\sigma^2$ の正規分布に従うと仮定すれば、$\mathbf{y} \sim \mathcal{N}(\mathbf{X} \mathbf{w}, \sigma^2 \mathbf{I})$ という確率モデルが得られる。

#### 最小二乗法
**最小二乗法**（ordinary least squares, OLS）では、観測値 $\mathbf{y}$ と予測値 $\mathbf{Xw}$ との差（残差）を最小にするようにパラメータ $\mathbf{w}$ を推定する。残差ベクトル $\boldsymbol{\delta} = \mathbf{y} - \mathbf{Xw}$ に対し、目的関数 $\mathcal{L}(\mathbf{w})$ は次のように定義される：

$$
\begin{equation}
\mathcal{L}(\mathbf{w}) := \|\boldsymbol{\delta}\|^2 = \boldsymbol{\delta}^\top \boldsymbol{\delta}
\end{equation}
$$

この $\mathcal{L}(\mathbf{w})$ を最小化する $\mathbf{w}$ を求めることで、最適な重み $\hat{\mathbf{w}}$ を得る。最適解の推定は主に**正規方程式**（normal equation）あるいは**勾配法**（gradient descent）によって行うことができる．いずれの手法でも，目的関数 $\mathcal{L}(\mathbf{w})$ の $\mathbf{w}$ について微分、すなわち勾配 (gradient) $\nabla \mathcal{L}(\mathbf{w})$ が必要となる．$\nabla \mathcal{L}(\mathbf{w})$ は以下のように計算できる：

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
目的関数の勾配について $\nabla \mathcal{L}(\mathbf{w})=0$ となる解を $\hat{\mathbf{w}}$ とすると，次の**正規方程式**（normal equation）が得られる：

$$
\begin{equation}
\mathbf{X}^\top \mathbf{X} \hat{\mathbf{w}} = \mathbf{X}^\top \mathbf{y}
\end{equation}
$$

この方程式を解くことで、パラメータの推定値 $\hat{\mathbf{w}}$ は次のように求まる：

$$
\begin{equation}
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\end{equation}
$$

なお、$A^+ := (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$ は $\mathbf{X}$ の**Moore–Penrose 擬似逆行列**（pseudoinverse）と呼ばれ、この表現を用いると $\hat{\mathbf{w}} = A^+ \mathbf{y}$ と簡潔に記述できる。

##### 勾配法による数値的推定
最小二乗法に基づくパラメータ推定は、数値的には**勾配法**（gradient descent）によっても実現できる。 目的関数の勾配 $\nabla \mathcal{L}(\mathbf{w})$ を用いると、更新式は次のように与えられる：

$$
\begin{align}
\Delta \mathbf{w} \propto - \nabla \mathcal{L}(\mathbf{w})= 2\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw})\\
\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \frac{1}{n} \mathbf{X}^\top (\mathbf{y} - \mathbf{Xw})
\end{align}
$$

ここで $\alpha$ は**学習率**（learning rate）と呼ばれるハイパーパラメータである。

### リッジ回帰
線形回帰においては、説明変数が高次元である場合や、多重共線性（説明変数間の相関）が存在する場合などに、最小二乗法による推定が不安定になることがある。これに対処する手法として、**L2 正則化**を加えた**リッジ回帰**（ridge regression）が用いられる。

リッジ回帰では、目的関数にパラメータの二乗ノルムを加えた正則化項を導入することにより、モデルの複雑さを抑制し、過学習の防止や推定の安定化を図る。具体的には、次のような正則化付き目的関数 $\mathcal{L}_\lambda(\mathbf{w})$ を最小化する：

$$
\begin{equation}
\mathcal{L}_\lambda(\mathbf{w}) = \|\mathbf{y} - \mathbf{Xw}\|^2 + \lambda \|\mathbf{w}\|^2 = (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} - \mathbf{Xw}) + \lambda \mathbf{w}^\top \mathbf{w},
\end{equation}
$$

ここで $\lambda \geq 0$ は**正則化係数**（regularization parameter）であり、モデルのあてはまりと複雑さのトレードオフを制御する。

> **注記：** 通常、$w_0$（切片）には正則化を加えないことが多いため、必要に応じて $\mathbf{w}$ の対象を $[w_1, \dots, w_p]^\top$ に限定する処理を行う。

##### 正規方程式による解
L2 正則化付きの目的関数を $\mathbf{w}$ で微分して0に等しいとおくと、次のような修正された正規方程式が得られる：

$$
\begin{equation}
(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \hat{\mathbf{w}} = \mathbf{X}^\top \mathbf{y},
\end{equation}
$$

ここで $\mathbf{I} \in \mathbb{R}^{(p+1)\times(p+1)}$ は単位行列である．ただし、$w_0$ を正則化対象から除く場合、$\lambda \mathbf{I}$ の最初の対角成分をゼロにすることで対処する。この式を解くと、リッジ回帰におけるパラメータの推定値は次のように求まる：

$$
\begin{equation}
\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}.
\end{equation}
$$

この推定式は、$\mathbf{X}^\top \mathbf{X}$ が特異（非正則）である場合でも、$\lambda > 0$ により逆行列の存在が保証される点で、最小二乗法に比べて数値的に安定であるという利点がある。

##### 勾配法による推定

リッジ回帰に対しても勾配法を適用できる。目的関数 $\mathcal{L}_\lambda(\mathbf{w})$ の勾配は次のように求まる：

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

または $\alpha$ を調整することで、$\lambda$ を勾配更新の一部として組み込む方法もある。いずれにしても、正則化項によって重みの更新が抑制されることで、過学習を防ぐ効果が得られる。

### ロジスティック回帰
本節では、非線形回帰の一種である**ロジスティック回帰** (logistic regression) について取り扱う。

### ロジスティック回帰

ロジスティック回帰は、入力 $\mathbf{x} \in \mathbb{R}^p$ に対して出力 $y \in \{0, 1\}$ を予測する**確率的な分類モデル**である。出力は事後確率 $\Pr(y=1 \mid \mathbf{x})$ を表し、その予測にはシグモイド関数（ロジスティック関数）を用いる。

#### モデルの定義

ロジスティック回帰では、まず説明変数の線形結合を求める：

$$
z = w_0 + \sum_{j=1}^p w_j x_j = \mathbf{w}^\top \mathbf{x}'
$$

ここで $\mathbf{x}' := [1, x_1, x_2, \dots, x_p]^\top \in \mathbb{R}^{p+1}$ はバイアス項を含んだ拡張入力ベクトル、$\mathbf{w} \in \mathbb{R}^{p+1}$ はパラメータベクトルである。

この線形出力 $z$ に対して、シグモイド関数 $\sigma(z)$ を適用することで、出力の確率的解釈が得られる：

$$
\Pr(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + \exp(-z)}
$$

したがって、クラスラベル $y \in \{0, 1\}$ の**確率モデル**は次のように表される：

$$
p(y \mid \mathbf{x}; \mathbf{w}) = \sigma(z)^y (1 - \sigma(z))^{1 - y}
$$

#### パラメータの推定：最尤推定

ロジスティック回帰のパラメータは**最尤推定**により求める。データ集合 $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ に対して、対数尤度関数は以下のように定義される：

$$
\ell(\mathbf{w}) = \sum_{i=1}^n \left[ y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log (1 - \sigma(z^{(i)})) \right]
$$

ここで $z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)}$ である。

この尤度を最大化することで $\mathbf{w}$ を学習する。一般には閉形式解を持たないため、**勾配降下法**などの最適化手法を用いて数値的に解く。

勾配は以下のように計算される：

$$
\nabla \ell(\mathbf{w}) = \sum_{i=1}^n (y^{(i)} - \sigma(z^{(i)})) \mathbf{x}^{(i)}
$$