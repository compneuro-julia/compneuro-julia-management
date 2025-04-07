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
### Julia言語のインストール方法
### Julia言語の基本構文

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

#### ラプラス変換

Laplace変換はFourier変換に似た手法であり，微分方程式を解く上で便利である．
ToDo: Laplace変換の詳細

$$
\begin{equation}
F(s):=\int_0^{\infty} f(t) e^{-st} dt=\mathcal{L}(f(t))
\end{equation}
$$

$e^{-st}$を引っ付けて積分することで，被積分関数が$t\to \infty$で収束し，積分可能となっている．

実用上は次の対応表を用いて計算すればよい．
ToDo: Laplace変換の対応表

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
#### 期待値 (Expectation)

$$
\begin{equation}
\mathbb{E}_{x\sim p(x)}\left[f(x)\right]:=\int f(x)p(x)dx
\end{equation}
$$

$x\sim p(x)$ が明示的な場合は $\mathbb{E}_{p(x)}\left[f(x)\right]$ や $\mathbb{E}\left[f(x)\right]$ と表す．

#### 情報量 (Information)
出現頻度が低い事象は多くの情報量を持つ (Shannon, 1948)．

$$
\begin{equation}
\mathbb{I}(x):=\ln\left(\frac{1}{p(x)}\right)=-\ln p(x)
\end{equation}
$$

$\mathbf{I}$は単位行列なので注意．

#### 平均情報量 (エントロピー, entropy)

$$
\begin{align}
\mathbb{H}(x)&:=\mathbb{E}[-\ln p(x)]\\
\mathbb{H}(x\vert y)&:=\mathbb{E}[-\ln p(x\vert y)]
\end{align}
$$

#### Kullback-Leibler 情報量
Kullback-Leibler (KL) divergence (Kullback and Leibler, 1951)

$$
\begin{align}
D_{\text{KL}}\left(p(x) \Vert\ q(x)\right)&:=\int p(x) \ln \frac{p(x)}{q(x)} dx\\
&=\int p(x) \ln p(x) dx-\int p(x) \ln q(x) dx\\
&=\mathbb{E}_{x\sim p(x)}[\ln p(x)]-\mathbb{E}_{x\sim p(x)}[\ln q(x)]\\
&=-\mathbb{H}(x)-\mathbb{E}_{x\sim p(x)}[\ln q(x)]
\end{align}
$$

#### 相互情報量 (Mutual information)

## 学習に関する基礎的概念
### モデルと学習・予測
**機械学習** (machine learning) における**モデル** (model) とは，2つの集合 $\mathcal{X}, \mathcal{Y}$ を仮定した際に，入力 $x\in \mathcal{X}$ を出力 $y\in \mathcal{Y}$ に変換する関数 (写像) $f: x \to y$ あるいは条件付き確率分布 $p(y|x)$ を意味する．モデルは内部に媒介変数あるいはパラメータ (parameter) $\theta$ を持ち，$\mathcal{Y}$ を設定した後に $y=f(x; \theta)$ あるいは $p(y|x; \theta)$ を満たすように $\theta$ を更新する．この過程を**学習** (learning) あるいは**訓練** (training) と呼ぶ．学習後のパラメータ $\theta^*$を用い，$x$が与えられた際の$y$ の推定値$\hat{y}$を $\hat{y}=f(x; \theta^*)$ あるいは $p(y|x; \theta^*)$ から取得する\footnote{取得の方法としてはサンプリング $\hat{y}\sim p(y|x; \theta^*)$ や $\hat{y}=\textrm{argmax}\ p(y|x; \theta^{*})$などが考えられる．}ことを**予測** (prediction) と呼ぶ．学習の際に用いられるデータを訓練データ (training data) と呼び，学習後のモデルの予測精度の評価に用いるデータを評価データ (test data) と呼ぶ．

ここ修正すべき

$y$が既知の場合は$D=\{(x,y)\}$は教師付きデータ ($y$がラベルの場合はラベル付きデータ) と呼ばれ，$x$ と $y$ の対応関係を学習する過程を教師あり学習 (supervised learning) と呼ぶ．$y$が未知の場合，$D=\{x\}$はラベルなしデータと呼ばれ，これのみでモデルを学習する過程を
教師なし学習 (unsupervised learning) と呼ぶ．この2つの学習の派生として，ラベルあり・なしデータを併用する半教師あり学習 (semi-supervised learning), 教師なし学習の一種であり，入力データの部分集合から他の部分集合を予測する自己教師あり学習 (self-supervised learning) などが存在する．

強化学習 (reinforcement learning) は

### 回帰と分類

\citep{2015-yz}.

### 識別モデル・生成モデル

オンライン・オフライン学習