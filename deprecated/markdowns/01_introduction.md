# 第1章：はじめに
## 本書の目的と構成
### 神経科学における計算論
本書では神経科学における数理モデルを主として取り扱う．初めに神経科学におけるモデルの役割について触れておこう．まず，神経科学の目標は端的に言えば「脳神経系を理解する」ことにある．神経科学に限らず，種々の学問分野においては実験と理論の2本柱で，対象とする現象や物質の理解が進められる．ここで実験は調査等も踏まえ実データを取る行為とする．理論の役割は複数あり，実験結果の抽象化および統合，仮説の提供，現象の予測等である \citep{Blohm2020-vc}．

「脳神経系を理解する」ということに関して，その定義は研究者により様々である．ここでは脳の計算処理に関する理論的理解を進めるための1つの方法として Marrの3レベル (Marr's Three Levels) を紹介する \citep{Marr1982-wk}．Marrの3レベルは視覚系における計算処理の理解を主としていたが，他でも適用可能である．3レベルとは(1)計算理論 (computational theory), (2) 表現・アルゴリズム (representation and algorithm), (3)実装 (implementation) であり，それぞれの段階での議論や理解を行う．(1)では脳の目的関数とそれを用いた最適化問題の設定を行う．(2)では(1)を実現するための表現およびアルゴリズムを解明する．(3)では(1,2)を神経回路・ハードウェア上で実装する方法を解明することを目標とし，平易には「脳」を作って理解すると言い換えることもできる\footnote{ここでの「作る」は計算機等でシミュレーションするという意味であり，脳オルガノイド (brain organoid) を作成するなどの意味ではない}．本書ではこの(3)を重視し，読者が自らの手で理論を検証し，数値計算による結果を再現できることを目標とした．また，本書は数式をプログラミングのコードに変換する具体例集としての役割も持っている．

モデルの中でも，本書では機械学習に関連する内容が多数登場する．これは神経科学と機械学習は互いに影響を及ぼし合ってきたためである \citep{Hassabis2017-zm}. 
神経科学から機械学習への応用は例えば，ニューラルネットワーク，記憶モデル，注意モデルなどがある．逆に機械学習から神経科学への応用は強化学習，運動制御，ベイズ脳仮説などが挙げられる．

筆者の立場としては，神経科学は機械学習の発展のためにあるわけではないので，後者の流れ，すなわち機械学習から神経科学への応用を重視して本書を執筆した次第である．

### 本書の構成
第1章では，Julia言語の使用法と用いる数学について簡単に説明する．第2章から第5章までは発火率モデルおよびニューラルネットワークについての説明を行う．第2章では，まず神経細胞の簡単な生理学について説明する．発火率モデルを説明したのち，局所学習則によって訓練されるネットワークの説明を行う．第3章では，同じく局所学習則ではあるが，ネットワーク全体のエネルギーを下げることを目的としたエネルギーベースモデルと呼ばれる枠組みのネットワークについて説明をする．第4章では，誤差逆伝播法に基づいたニューラルネットワークを説明し，貢献度分配問題の生理学的な解決策について説明をする．第5章では，さらに再起型ニューラルネットワークを説明し，経時的貢献度分配問題について説明を行う．

第6・7章ではスパイキングニューラルネットワークとその学習について取り扱う．第6章ではネットワークレベルの話から再び神経細胞とシナプスに回帰する．次に，シナプスのダイナミクスについて説明を補いながらモデルを構築する．第7章ではネットワークの構築と学習について，第2章から第5章までを踏まえて説明する．

第8章から12章は上記以外の内容について各論的に説明を行う．第8章のリザバーコンピューティングの章では，リザバーコンピューティングと呼ばれる枠組みのネットワークおよびその学習則について，発火率・スパイキングモデルの双方をまとめて紹介する．第9章ではベイズ推論の章では，神経回路網により，如何にして確率計算を行うかを説明する．第10章では運動学習では，最適制御問題の解決策について説明する．第11章の強化学習では，強化学習の基本的事項の説明と，大脳基底核との関連性について説明する．第12章は補足的な話題であり，ネットワーク・形態学・グリアについて説明を行う．

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

また，2025年3月以降，Google Colab (\url{https://colab.google/}) においてPythonやRに並んでJuliaを選択して使用することが可能となっている．

### 使用するライブラリ

REPL
で`]` を入力することで，パッケージ管理モードに移行する．

本書で使用するJuliaライブラリは以下の通りである．

- IJulia: 開発環境
- PyPlot: 描画用ライブラリ
- LinearAlgebra: 高度な線形代数演算
- Random: 

Pythonではnumpyで完結するところをライブラリをいくつも読み込む必要がある点は欠点ではある．

描画用のライブラリには `PyPlot.jl` を使用した．`PyPlot` はPythonライブラリである `matplotlib` に依存したライブラリである．Juliaで完結させたい場合は `Plot.jl` や `Makie.jl` を使用することが推奨されるが，`PyPlot` (`matplotlib`) の方が高機能であるため，

Pythonがない場合は

```julia
julia> ENV["PYTHON"] = ""
julia> ]
pkg> build PyCall
```

Pythonを既にインストールしている場合は，

```julia
julia> ENV["PYTHON"] = raw"C:\Users\TakutoYamamoto\AppData\Local\Programs\Python\Python312\python.exe"
julia> ]
pkg> build PyCall
```

Windowsの場合
例としてPythonの実行ファイル (python.exe) への完全なパスを


### 開発環境

インタプリタ型言語である

vscode

筆者は（Pythonユーザーでもあるため）Jupyter Labを使用している．

JuliaのみでJupyter Labを使用するには

```julia
using IJulia
jupyterlab(detached=true)
```

とすればよい．ただし，この際にCondaを入れることになるため，別途Pythonをインストールしておく方が推奨される．

p.33

`Pluto.jl` を用いることも可能である

### Julia言語の基本構文

https://docs.julialang.org/en/v1/manual/noteworthy-differences/

### 命名規則
この節では，本書で用いるJuliaの変数名や関数名等に関する基本的な取り決めをまとめる．

#### 変数名
- `nt`: 時間ステップ数 (number of time steps)
- `t`, `tt`: 時間ステップのインデント

## 基礎的数学とJuliaでの記法
本書で使用する数学的内容を整理する．

### 表記法
本書では次のような記号表記を用いる．
- 実数全体を$\mathbb{R}$，自然数全体を $\mathbb{N}$，複素数全体は$\mathbb{C}$と表記する．
- スカラーは小文字・斜体で$x$のように表記する．
- ベクトルは小文字・立体・太字で$\mathbf{x}$のように表記し，列ベクトル (縦ベクトル) として扱う．
- 行列は大文字・立体・太字で$\mathbf{X}$のように表記する．
- $n\times 1$の実ベクトルの集合を$\mathbb{R}^n$,$n\times m$の実行列の集合を$\mathbb{R}^{n\times m}$と表記する．
- 行列$\mathbf{X}$の置換は$\mathbf{X}^\top$と表記する．ベクトルの要素を表す場合は$\mathbf{x} = (x_1, x_2,\cdots, x_n)^\top$のように表記する．
- 単位行列を$\mathbf{I}$と表記する．$n \times n$ 次元の単位行列は $\mathbf{I}_n$ と表記する．
- ゼロベクトルは$\mathbf{0}$, 要素が全て1のベクトルは$\mathbf{1}$と表記する．
- ベクトル・行列の微分には分子レイアウト記法を使用する．
- 基本的に確率変数は大文字 $X$ のように表記し，確率変数の実現値は小文字 $x$ を用いる．ただし，大文字であっても確率変数でない場合や，実現値がベクトルの場合などがあるため，必ずしもこの規則に従うわけではない．
- $e$を自然対数の底とし，指数関数を$e^x=\exp(x)$と表記する．また，自然対数を$\ln(x)$と表記する．
- 定義を$\coloneqq$を用いて行う．例えば，$f(x)\coloneqq2x$ は $f(x)$ という関数を$2x$として定義するという意味である．定義する対象が右側である場合は，$\eqqcolon$を用いる．
- 比例関係を表す際は $\propto$ を使用する．例えば $a \propto b$ は $a$ と $b$ が比例関係にあることを意味する．
- 平均$\mu$, 標準偏差$\sigma$の正規分布を$\mathcal{N}(\mu, \sigma^2)$と表記する．

### 線形代数と微分

**線形代数 (Linear Algebra)** は、ベクトルや行列といった線形構造を持つ対象の性質を解析する数学の分野であり、現代のあらゆる数学・工学・情報科学の基礎をなしている。線形代数の中心的な対象は、**ベクトル空間**、**線形写像**、およびそれらの表現である**行列**である。


matrix cookbookに詳しいが，
https://arxiv.org/abs/2501.14787
Introduction to Applied Linear Algebra – Vectors, Matrices, and Least Squares
スタンフォード　ベクトル・行列からはじめる最適化数学

@book{Piaget1936,
  author = {Jean Piaget},
  title = {La naissance de l'intelligence chez l'enfant},
  year = {1936},
  publisher = {Delachaux et Niestlé},
  note = {田中寛一訳『児童の知能の誕生』, 岩波書店, 1970年}
}

訳本はnoteに記載．

まず、**ベクトル空間 (vector space)** とは、スカラー体（通常は実数$\mathbb{R}$または複素数$\mathbb{C}$）に対して定義された加法とスカラー倍という2つの演算に関して閉じている集合である。たとえば$\mathbb{R}^n$は、$n$個の実数からなるベクトル全体の集合であり、典型的なベクトル空間の例である。任意のベクトル$\mathbf{v}, \mathbf{w} \in \mathbb{R}^n$とスカラー$\alpha \in \mathbb{R}$に対して、

$$
\begin{equation}
\alpha\mathbf{v} + \mathbf{w} \in \mathbb{R}^n
\end{equation}
$$

が成り立つ。

**線形写像 (linear transformation)** とは、ベクトル空間からベクトル空間への写像$T: V \to W$であり、加法とスカラー倍に対して線形性を持つもの、すなわち

$$
\begin{equation}
T(\alpha \mathbf{v} + \beta \mathbf{w}) = \alpha T(\mathbf{v}) + \beta T(\mathbf{w})
\end{equation}
$$

が任意の$\mathbf{v}, \mathbf{w} \in V$とスカラー$\alpha, \beta$に対して成り立つ写像である。

このような線形写像は、基底を定めることで**行列 (matrix)** によって表現できる。たとえば、$n$次元から$m$次元への線形写像は、$m \times n$の行列$A$を用いて

$$
\begin{equation}
\mathbf{y} = A\mathbf{x}
\end{equation}
$$

という形で記述される。ここで$\mathbf{x} \in \mathbb{R}^n$,$\mathbf{y} \in \mathbb{R}^m$はそれぞれ入力および出力ベクトルである。

**行列の積**：$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$に対し、$AB \in \mathbb{R}^{m \times p}$を定義。

**転置**：$A^\top$は行列$A$の行と列を交換したもの。

**逆行列**：$A \in \mathbb{R}^{n \times n}$が正則（可逆）であれば、$A^{-1}$が存在し$AA^{-1} = A^{-1}A = I$を満たす。

**行列式**（determinant）：$\det A$は正方行列$A$に対するスカラー量で、行列の体積のスケーリング率や可逆性の指標となる。

**線形方程式系**の解法である。$A\mathbf{x} = \mathbf{b}$の形をした方程式において、$A$の逆行列が存在するならば、その解は

$$
\begin{equation}
\mathbf{x} = A^{-1}\mathbf{b}
\end{equation}
$$

と求められる。

**固有値問題**

ある正方行列$A$に対し、スカラー$\lambda$およびベクトル$\mathbf{v} \ne \mathbf{0}$が

$$
\begin{equation}
A\mathbf{v} = \lambda \mathbf{v}
\end{equation}
$$

を満たすとき、$\lambda$は$A$の**固有値 (eigenvalue)**、$\mathbf{v}$は**固有ベクトル (eigenvector)** と呼ばれる。固有値分解や対角化は、線形変換の構造解析や行列の関数（例：指数関数）を考える際に中心的な役割を果たす。


#### 外積・内積
まず、次のような行列とベクトルを考えます。
$$
\begin{equation}
M = 
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\in \mathbb{R}^{2\times 2},\quad \mathbf{v} = 
\begin{pmatrix}
5 \\ 6
\end{pmatrix}
\in \mathbb{R}^{2}
\end{equation}$$

これらの**テンソル積**（外積）$M \otimes \mathbf{v}$ を取る．

各成分について、単にすべてのペアを作り、**行列の成分 × ベクトルの成分**を計算して、新たな配列を作る。  
添え字を明示すると、
$$
\begin{equation}
(M\otimes \mathbf{v})_{ijk} = M_{ij}\, v_k
\end{equation}
$$
のように、行列$M_{ij}$（行$i$、列$j$）と、ベクトル$v_k$（要素$k$）をかけたものを並べる。よって $M\otimes \mathbf{v}$ は次のような形状は $(2,2,2)$ の**3階テンソル**になります。

$$\begin{equation}
M \otimes \mathbf{v} =
\begin{pmatrix}
\begin{pmatrix}
5 & 6 \\
10 & 12
\end{pmatrix}
\\
\begin{pmatrix}
15 & 18 \\
20 & 24
\end{pmatrix}
\end{pmatrix}
\end{equation}$$

直感的にいうと行列のそれぞれの成分をコピーし、ベクトルの各成分でスケールして、奥行き方向（新しい次元）に並べる．

テンソル積 (tensor product)
テンソル縮約 (tensor contraction)

ここで，$\tilde{\otimes}_1$ を次のように定義する．行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$、テンソル $\mathbf{B} \in \mathbb{R}^{n \times p \times q}$ とする．$\mathbf{C} \in \mathbb{R}^{m \times p \times q}$ を  

$$
\mathbf{C} := \mathbf{A} \,\tilde{\otimes}_1\, \mathbf{B}
$$

と定義する。ただし、$\tilde{\otimes}_1$ は $\mathbf{B}$ の第1軸に沿って $\mathbf{A}$ と縮約（行列積）を行う演算であり、これは $\mathbf{B}$ の各スライス $\mathbf{B}_{::\ell}$ に対して $\mathbf{A} \mathbf{B}_{::\ell}$ を逐次計算した結果を第3軸に沿って並べたものである。

#### テンソル縮約（Contraction）とアインシュタイン縮約記法（Einsum）

テンソルとは、多次元配列の一般化された概念である。一次元配列であるベクトルや二次元配列である行列を含むより高次の対象を統一的に扱う枠組みであり、階数（rank）と呼ばれる次元数によって分類される。階数が1のテンソルはベクトル、階数が2のテンソルは行列に対応し、それ以上の階数を持つものを一般に高次元テンソルと呼ぶ。テンソルは、座標変換に対して一定の変換則に従う数学的対象としても定義されるが、ここでは具体的な数値配列としての側面に焦点を当てる。

テンソルの縮約とは、多次元配列（テンソル）の特定の軸同士について和を取る操作を指す。これは線形代数における内積や行列積を高次元に拡張した概念であり、縮約によって元のテンソルの階数は減少し、より低次元のテンソルが得られる。縮約演算は、テンソルの特定の軸にわたる要素同士を対応させ、それらの積にわたる総和を計算することで実現される。

最も基本的な例として行列積が挙げられる。たとえば、行列 $A \in \mathbb{R}^{n \times m}$ と $B \in \mathbb{R}^{m \times p}$ の積 $C = AB \in \mathbb{R}^{n \times p}$ は、$A$ の第2軸と $B$ の第1軸を縮約することで得られる。このとき、各成分は

$$
\begin{equation}
C_{ik} = \sum_{j=1}^m A_{ij} B_{jk}
\end{equation}
$$

により与えられる。この縮約操作により、縮約された軸（ここでは $m$）は計算後に消失し、残された軸に対応する新たなテンソルが得られる。

テンソル縮約をより簡潔かつ体系的に記述する方法として、アインシュタイン縮約記法（Einstein summation convention）が存在する。アインシュタイン記法では、縮約される添え字を明示的に総和記号で表すことなく、単に同じ添え字が現れた場合にはその添え字について暗黙に総和を取る規則を採用する。これにより、数式表記が大幅に簡潔化される。

たとえば、行列積はアインシュタイン記法を用いると
$$
\begin{equation}
C_{ik} = A_{ij} B_{jk}
\end{equation}
$$

と書かれる。この表式では、添え字 $j$ が2回出現しているため、この添え字について縮約（すなわち総和を取る）が暗黙に行われるものと解釈される。

アインシュタイン縮約記法において、添え字には**ダミーインデックス**（dummy index）と **フリーインデックス**（free index）の2種類がある。ひとつはである。ダミーインデックスとは、式中に2回出現し、その添え字について縮約（総和）が行われるものを指す。たとえば先の例では添え字 $j$ がダミーインデックスである。一方、フリーインデックスとは、式中に1回しか現れず、縮約されずに最終的な結果の軸（次元）として残る添え字を指す。この例では $i$ と $k$ がフリーインデックスであり、結果のテンソル $C$ はフリーインデックス $(i,k)$ によってパラメータ付けされる。

ダミーインデックスは縮約により消失し、最終的なテンソルの構造に寄与しないが、フリーインデックスは結果のテンソルの階数や形状を決定するため、慎重に取り扱う必要がある。アインシュタイン記法においては、同じダミーインデックスを複数の場所で用いることはできず、また、フリーインデックスは各項で一貫して同じ意味を持たなければならない。この規則を守ることで、式の整合性と意味の明確性が保証される。

テンソル縮約は、二次元の行列同士の積に限らず、より高次元のテンソル間にも自然に拡張される。たとえば、テンソル $A \in \mathbb{R}^{n \times m \times p}$ と $B \in \mathbb{R}^{p \times m \times q}$ に対して、軸 $p$ と $m$ を縮約し、残る軸 $n$ と $q$ に対応するテンソル $C \in \mathbb{R}^{n \times q}$ を得ることができる。このときアインシュタイン記法では、

$$
\begin{equation}
C_{nq} = A_{imp} B_{pmq}
\end{equation}
$$

と表される。この表式では添え字 $m$ および $p$ がそれぞれ2回ずつ現れているため、それらについて縮約が行われ、フリーインデックス $n$ と $q$ に対応する次元が保持されることになる。

高次元テンソルの演算では、複数の軸の縮約を同時に行ったり、縮約と同時に軸の入れ替え（転置）や結合（reshape）を伴うことも一般的である。こうした複雑な操作もアインシュタイン記法を用いれば一貫した枠組みの中で簡潔に記述できる。特に、縮約する添え字と残す添え字を明確に区別することにより、結果となるテンソルの構造を容易に予測することができる。

アインシュタイン記法に基づいた計算は、プログラミングにおいても広く利用されている。特に、PythonのNumPyやJuliaなどの数値計算ライブラリでは、

ここで、$\tilde{\otimes}_1$ を次のように定める。行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$、テンソル $\mathbf{B} \in \mathbb{R}^{n \times p \times q}$ に対し、

$$
\begin{equation}
\mathbf{C} := \mathbf{A} \,\tilde{\otimes}_1\, \mathbf{B} \in \mathbb{R}^{m \times p \times q}
\end{equation}
$$

を、$\mathbf{B}$ の第1軸に沿って $\mathbf{A}$ と縮約（行列積）を行う操作として定義する。すなわち、各 $\ell = 1,\dots,q$ に対し $\mathbf{C}_{::\ell} = \mathbf{A} \mathbf{B}_{::\ell}$ が成り立つ。

TensorOperations.jl
Einsum.jl


$$
\begin{equation}
\delta _{ij}={\begin{cases}1&(i=j)\\0&(i\neq j)\end{cases}}
\end{equation}
$$



#### ベクトル・行列の微分
本書ではベクトルおよび行列の微分を多用する．これは成分ごとに記載するよりも，ベクトル・行列演算をコードに変換しやすいという実装上の利点があるためである．初めに注意したいこととして，ベクトル・行列の微分の記法には分子レイアウト記法 (numerator-layout notation) と分母レイアウト記法 (denominator-layout notation) の2種類が存在する．これらは，ベクトル関数やスカラー関数に対する微分の定義の仕方に違いがあり，特に勾配ベクトルの形（行ベクトルか列ベクトルか）や連鎖律の表記に影響を及ぼす．いずれが使用されているかは文献ごとにバラバラであり，中には両方の記法を採用している文献も存在する．本書では，本書では**分子レイアウト記法**を統一的に使用する．記法の例を記述するため，スカラー $x, y \in \mathbb{R}$, ベクトル $\mathbf{x}=[x_i] \in \mathbb{R}^n, \mathbf{y}=[y_j] \in \mathbb{R}^m$, 行列 $\mathbf{A}=[a_{ij}] \in \mathbb{R}^{p \times q}$ を使用する．分子（従属変数）と分母（独立変数）の組み合わせから，次の6通りの微分が定義される．

$$
\begin{align*}
\begin{array}{c|c|c|c}
& \text{スカラー} & \text{ベクトル} & \text{行列}\\
\hline
\text{スカラー} & \frac{\partial y}{\partial x} & \frac{\partial \mathbf{y}}{\partial x} & \frac{\partial \mathbf{A}}{\partial x} \\
\hline
\text{ベクトル} & \frac{\partial y}{\partial \mathbf{x}} & \frac{\partial \mathbf{y}}{\partial \mathbf{x}} &  \\
\hline
\text{行列} & \frac{\partial y}{\partial \mathbf{A}} & &  \\
\hline
\end{array}
\end{align*}
$$

行名が分子の変数の種類，列名が分母の変数の種類を表している．まず，スカラーで偏微分する場合は，

$$
\begin{align}
\dfrac{\partial y}{\partial x} \in \mathbb{R}, \quad
\frac{\partial \mathbf{y}}{\partial x}:=
\begin{bmatrix}
\frac{\partial y_{1}}{\partial x}\\
\frac{\partial y_{2}}{\partial x}\\
\vdots \\
\frac{\partial y_{m}}{\partial x}\\
\end{bmatrix}
\in \mathbb{R}^m, \quad
\frac{\partial \mathbf{A}}{\partial x}:=
\begin{bmatrix}
\frac{\partial a_{11}}{\partial x}&\frac{\partial a_{12}}{\partial x}&\cdots &\frac{\partial a_{1q}}{\partial x}\\
\frac{\partial a_{21}}{\partial x}&\frac{\partial a_{22}}{\partial x}&\cdots &\frac{\partial a_{2q}}{\partial x}\\
\vdots &\vdots &\ddots &\vdots \\
\frac{\partial a_{p1}}{\partial x}&\frac{\partial a_{p2}}{\partial x}&\cdots &\frac{\partial a_{pq}}{\partial x}\\
\end{bmatrix}
\in \mathbb{R}^{p \times q}
\end{align}
$$

である．次にベクトルで偏微分する場合，

$$
\begin{align}
\frac{\partial y}{\partial \mathbf{x}}:=
\begin{bmatrix}
\frac{\partial y}{\partial x_{1}}&\frac{\partial y}{\partial x_{2}}&\cdots &\frac{\partial y}{\partial x_{n}}
\end{bmatrix}
\in \mathbb{R}^{1\times n}, \quad
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}:=
\begin{bmatrix}
\frac{\partial y_{1}}{\partial x_{1}}&\frac{\partial y_{1}}{\partial x_{2}}&\cdots &\frac{\partial y_{1}}{\partial x_{n}}\\
\frac{\partial y_{2}}{\partial x_{1}}&\frac{\partial y_{2}}{\partial x_{2}}&\cdots &\frac{\partial y_{2}}{\partial x_{n}}\\
\vdots &\vdots &\ddots &\vdots\\
\frac{\partial y_{m}}{\partial x_{1}}&\frac{\partial y_{m}}{\partial x_{2}}&\cdots &\frac{\partial y_{m}}{\partial x_{n}}\\
\end{bmatrix}
\in \mathbb{R}^{m \times n}
\end{align}
$$

である．ここで $\nabla_\mathbf{x} y(\mathbf{x}):=\left(\dfrac{\partial y}{\partial \mathbf{x}}\right)^\top=\dfrac{\partial y}{\partial \mathbf{x}^\top}\in \mathbb{R}^{n}$ は $y$ の $\mathbf{x}$ に対する勾配 (gradient) と呼ばれ，分子レイアウト記法において，勾配は導関数（derivative）の転置として表される\footnote{$\nabla$ はナブラ (nabla) と呼ばれる演算子であり，Juliaでも`\nabla TAB` として入力可能である．}．すなわち、要素は同じだが，勾配は列ベクトル，導関数は行ベクトルになる。しかしながら、機械学習でのパラメータ更新では、転置の有無にかかわらず「勾配」という語が広く使われている。そこで本書では、転置前の導関数 $\frac{\partial y}{\partial \mathbf{x}}$ に対しても便宜的に「勾配」と呼ぶこととする。また，$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ は $\mathbf{y}$ に対する $\mathbf{x}$ のJacobian行列と呼ばれる．最後に，行列で偏微分する場合，

$$
\begin{align}
\frac{\partial y}{\partial \mathbf{A}}=
\begin{bmatrix}
\frac{\partial y}{\partial a_{11}}&\frac{\partial y}{\partial a_{21}}&\cdots &{\frac{\partial y}{\partial a_{p1}}}\\
\frac{\partial y}{\partial a_{12}}&\frac{\partial y}{\partial a_{22}}&\cdots &{\frac{\partial y}{\partial a_{p2}}}\\
\vdots &\vdots &\ddots &\vdots \\
\frac{\partial y}{\partial a_{1q}}&\frac{\partial y}{\partial a_{2q}}&\cdots &{\frac{\partial y}{\partial a_{pq}}}\\
\end{bmatrix}
\in \mathbb{R}^{q \times p}
\end{align}
$$

である．これも $\nabla_\mathbf{A} y(\mathbf{A}):= \left(\dfrac{\partial y}{\partial \mathbf{A}}\right)^\top=\dfrac{\partial y}{\partial \mathbf{A}^\top}$ とも表記でき，$y$ の $\mathbf{A}$ に対する勾配と呼ぶ．

微分係数 $f'$

| 微分対象 | 微分元 | 添え字 | 結果の次元 |
|:---|:---|:---|:---|
| $\frac{\partial y}{\partial x^j}$ | スカラー／ベクトル | $j$ | $1\times n$ |
| $\frac{\partial y}{\partial X^{ij}}$ | スカラー／行列 | $i,j$ | $m\times n$ |
| $\frac{\partial y^i}{\partial x^j}$ | ベクトル／ベクトル | $i,j$ | $m\times n$ |
| $\frac{\partial y^i}{\partial X^{pq}}$ | ベクトル／行列 | $i,p,q$ | $m\times m\times n$ |
| $\frac{\partial Y^{ij}}{\partial x^k}$ | 行列／ベクトル | $i,j,k$ | $m\times n\times n$ |
| $\frac{\partial Y^{ij}}{\partial X^{pq}}$ | 行列／行列 | $i,j,p,q$ | $m\times n\times m\times n$ |


### 微分方程式
微分方程式はある関数とそれを微分した導関数の関係式であり，関数の特定の変数に対する変化を記述することができる．まず，1階線形微分方程式を例として見てみよう．

$$
\begin{equation}
\frac{\mathrm{d}x(t)}{\mathrm{d}t}=a_c x(t)+b_c u(t)
\end{equation}
$$

状態変数$x(t)$は，時間$t$に対する関数である．

添え字の$c$は連続 (continuous) を意味するが，これは後で離散化する際に区別するためである．この方程式においては$b_c=0$の場合を**同次方程式**,$b_c\neq 0$の場合を**非同次方程式**という．

#### 微分方程式の解
微分方程式を解くとは$x(t)$のような関数の具体的な式を求めることである．上式の解は

$$
\begin{equation}
x(t)=e^{a_c t}x(0)+\int_0^t e^{a_c (t-\tau)}b_c u(\tau) \,\mathrm{d}\tau
\end{equation}
$$

として与えられる．微分方程式を解く手法は様々で，それぞれの方程式について適切な手法を選択する．本書ではLaplace変換を多用するが，細かい説明は付録にて行う．

#### 連立線形微分方程式
$n$個の微分方程式

連立線形微分方程式という．これをベクトル，行列を用いて

時不変 (time-invariant) の定数行列を$\mathbf{A}_c \in \mathbb{R}^{n\times n}, \mathbf{B}_c \in \mathbb{R}^{n\times m}$, 状態ベクトルを$\mathbf{x}(t)\in\mathbb{R}^n$, 入力ベクトルを$\mathbf{u}(t)\in\mathbb{R}^m$とする．

$$
\begin{equation}
\frac{d\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{A}_c\mathbf{x}(t) + \mathbf{B}_c\mathbf{u}(t)
\end{equation}
$$

解は

$$
\begin{equation}
\mathbf{x}(t)=e^{t\mathbf{A}_c}\mathbf{x}(0)+\int_0^t e^{(t-\tau)\mathbf{A}_c}\mathbf{B}_c\mathbf{u}(\tau) \,\mathrm{d}\tau
\end{equation}
$$

#### Laplace変換
**Laplace変換**は、与えられた時間領域の関数$f(t)$を複素数変数$s$の関数$F(s)$に写像する積分変換である。特に、線形微分方程式の解析や制御工学において非常に有効な手法であり、Fourier変換と密接な関係をもつ。

Laplace変換は、実時間領域$t \ge 0$上で定義された関数$f(t)$に対して、以下のように定義される：

$$
\begin{equation}
F(s) \coloneqq \mathscr{L}(f(t)) = \int_0^{\infty} f(t)\, e^{-st} \mathrm{d}t
\end{equation}
$$

ここで$s \in \mathbb{C}$は複素数変数であり、通常は$s = \sigma + i\omega$の形をとる。変換核$e^{-st}$を掛けて積分することにより、関数$f(t)$の無限大での振る舞いを抑制し、積分を収束させる効果を持つ。特に、$f(t)$が指数関数的増加を含む場合でも、$e^{-st}$による減衰によってその成分を抑えることが可能となる。

Laplace変換の大きな利点の一つは、**微分演算を代数演算に変換できる**という性質にある。すなわち、$f(t)$の微分$\frac{d}{\mathrm{d}t}f(t)$に対するLaplace変換は、次のように与えられる：

$$
\begin{equation}
\mathscr{L}\left(\frac{df}{\mathrm{d}t}\right) = sF(s) - f(0)
\end{equation}
$$

この性質により、常微分方程式はLaplace変換の下で代数方程式に変換され、解の導出が容易となる。初期値を含んだ微分方程式を直接的に解くことができるため、初期値問題への応用にも適している。

さらに、Laplace変換には線形性：

$$
\begin{equation}
\mathscr{L}(af(t) + bg(t)) = a\mathscr{L}(f(t)) + b\mathscr{L}(g(t))
\end{equation}
$$

および畳み込みに関する定理：

$$
\begin{equation}
\mathscr{L}(f * g)(t) = F(s)G(s)
\end{equation}
$$

など、多くの有用な性質がある。これにより、時間領域での複雑な演算が周波数領域で簡単な演算として扱えるようになる．

#### 1階線形行列微分方程式の解
時不変 (time-invariant) の定数行列を$\mathbf{A} \in \mathbb{R}^{n\times n}, \mathbf{B} \in \mathbb{R}^{n\times m}$, 状態ベクトルを$\mathbf{x}(t)\in\mathbb{R}^n$, 入力ベクトルを$\mathbf{u}(t)\in\mathbb{R}^m$とする．

$$
\begin{equation}
\frac{d\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)
\end{equation}
$$

この線形行列微分方程式をLaplace変換$\mathscr{L}$を用いて解こう．$\boldsymbol{X}(s) \coloneqq \mathscr{L}(\mathbf{x}(t)), \boldsymbol{U}(s) \coloneqq \mathscr{L}(\mathbf{u}(t))$とすると，

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
e^\mathbf{A} = \exp(\mathbf{A}) \coloneqq \sum_{k=0}^\infty \frac{1}{k!}\mathbf{A}^k = \mathbf{I}+\mathbf{A}+\frac{\mathbf{A}^2}{2!}+\cdots
\end{equation}
$$

として定義される．天下り的だが，

$$
\begin{align}
\mathscr{L}(e^{at})&=\frac{1}{s-a}\\
\mathscr{L}(e^{t\mathbf{A}})&=(s\mathbf{I} - \mathbf{A})^{-1}\\
\end{align}
$$

であるので．よって

$$
\begin{align}
\boldsymbol{X}(s) &= (s\mathbf{I} - \mathbf{A})^{-1}(\mathbf{x}(0) + \mathbf{B}\boldsymbol{U}(s))\\
&= (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{x}(0) + (s\mathbf{I} - \mathbf{A})^{-1}\mathbf{B}\boldsymbol{U}(s)\\
\mathbf{x}(t)&=e^{t\mathbf{A}}\mathbf{x}(0)+\int_0^t e^{(t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) \,\mathrm{d}\tau
\end{align}
$$

となる．最後の式は両辺を逆Laplace変換した．ここで，$\mathscr{L}^{-1}(F(s)G(s))=\int_0^tf(\tau)g(t-\tau)\,\mathrm{d}\tau$であることを用いた．区間$[t, t+\Delta t]$において入力$\mathbf{u}(t)$が一定であると仮定すると，

$$
\begin{align}
\mathbf{x}(t+\Delta t)&=e^{(t+\Delta t)\mathbf{A}}\mathbf{x}(0)+\int_0^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) \,\mathrm{d}\tau\\
&=e^{\Delta t\mathbf{A}}e^{t\mathbf{A}}\mathbf{x}(0)+e^{\Delta t\mathbf{A}}\int_0^{t} e^{(t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) \,\mathrm{d}\tau + \int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}}\mathbf{B}\mathbf{u}(\tau) \,\mathrm{d}\tau\\
&\approx \underbrace{e^{\Delta t\mathbf{A}}}_{=: \mathbf{A}_d}\mathbf{x}(t)+\underbrace{\left[\int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}} \,\mathrm{d}\tau\right] \mathbf{B}}_{=: \mathbf{B}_d}\mathbf{u}(t)\\
&=\mathbf{A}_d\mathbf{x}(t)+\mathbf{B}_d\mathbf{u}(t)\\
\end{align}
$$

となる．添え字の$d$は離散化(discretization)を意味する．$\mathbf{A}_c$が正則行列の場合，

$$
\begin{align}
\mathbf{B}_d &= \left[\int_t^{t+\Delta t} e^{(t+\Delta t-\tau)\mathbf{A}} \,\mathrm{d}\tau\right] \mathbf{B}\\
&=\mathbf{A}^{-1}\left[e^{\Delta t \mathbf{A}}-\mathbf{I}\right]\mathbf{B}
\end{align}
$$

が成り立つ．

### 確率論
確率論の基本的な対象は**確率分布 (probability distribution)** である。確率分布は、ある確率変数がどのような値をどの程度の確率でとるかを定量的に記述するものである。確率変数$x$は、離散的あるいは連続的な値をとる場合があり、それぞれに応じて確率分布の定義も異なる。

離散的な場合、確率分布は**確率質量関数** (probability mass function; PMF) により定義され、任意の値$x$に対して$p(x)$はその値が観測される確率を与える。このとき、全ての確率の総和は 1 に等しくなければならない：

$$
\begin{equation}
\sum_x p(x) = 1
\end{equation}
$$

この代表例として**ポアソン分布 (Poisson distribution)** がある。ポアソン分布は、ある固定時間・空間内における稀な離散事象の発生回数をモデル化するものであり、以下のように定義される：

$$
\begin{equation}
p(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0,1,2,\dots
\end{equation}
$$

ここで$\lambda > 0$は単位時間（または空間）あたりの平均発生回数を表す。この分布は事象が独立かつ一定の発生率で起きると仮定する場面で用いられる。

一方、連続的な場合には**確率密度関数** (probability density function; PDF) を用いて定義される。確率密度関数$p(x)$は特定の値における確率そのものではなく、ある範囲に入る確率を積分によって与える関数である。たとえば、区間$[a,b]$における確率は次のように表される：

$$
\begin{equation}
\mathbb{P}(a \leq x \leq b) = \int_a^b p(x)\,\mathrm{d}x
\end{equation}
$$

確率密度関数もまた、定義域全体にわたる積分が 1 でなければならない：

$$
\begin{equation}
\int p(x)\,\mathrm{d}x = 1
\end{equation}
$$

この典型例として**正規分布 (normal distribution)** が挙げられる。正規分布は、多くの自然現象や測定誤差の分布を記述するのに適しており、平均$\mu$、分散$\sigma^2$をパラメータとして次のように定義される：

$$
\begin{equation}
p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\end{equation}
$$

この分布は平均値$\mu$を中心に左右対称であり、確率変数の分布が「中央に集まり、端にいくほど稀になる」という性質を持つ。特に$\mu = 0$,$\sigma^2 = 1$の場合は標準正規分布と呼ばれる．また，正規分布の概念は一変数の場合に限らず、多次元の確率変数にも拡張される。これが**多変量正規分布 (multivariate normal distribution)** であり、ベクトル値の確率変数がとる値の分布を記述する。

$d$次元の確率変数$\mathbf{x} \in \mathbb{R}^d$が平均ベクトル$\boldsymbol{\mu} \in \mathbb{R}^d$、共分散行列$\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$をもつ多変量正規分布に従うとき、その確率密度関数は以下のように定義される：

$$
\begin{equation}
p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma})^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
\end{equation}
$$

ここで、$\mathbf{x}$は$d$次元の確率変数ベクトル、$\boldsymbol{\mu}$は平均ベクトルであり、各成分が$\mathbf{x}$の平均値，$\boldsymbol{\Sigma}$は対象対称かつ正定値な共分散行列であり、成分$\Sigma_{ij}$は$\mathrm{Cov}(x_i, x_j)$を表す．

この分布は、各変数が正規分布に従い、かつそれらの間の線形な関係（共分散）もモデル化できる点で、非常に広範に用いられる。特に共分散行列が対角行列のとき、すなわち変数間が独立な場合には、各変数は独立な一変量正規分布に従う。


確率論においては、不確実性を定量的に扱うための基本的な概念がいくつか存在する。以下では、期待値、情報量、エントロピー、Kullback-Leibler情報量、そして相互情報量について簡単に説明を行う。

まず、**期待値 (Expectation)** は、確率変数$x$に関する関数$f(x)$の平均値を、$x$の確率分布$p(x)$に基づいて計算する操作である。連続値の場合、期待値は次のように定義される。

$$
\begin{equation}
\mathbb{E}_{x\sim p(x)}\left[f(x)\right] \coloneqq \int f(x)p(x)\,\mathrm{d}x
\end{equation}
$$

ここで$x \sim p(x)$は、$x$が分布$p(x)$に従うことを表す。文脈が明確な場合には、簡略に$\mathbb{E}_{p(x)}[f(x)]$や$\mathbb{E}[f(x)]$と表記する。

次に、**情報量 (Information)** は、ある特定の事象$x$の出現がどれほどの「驚き」や「情報」をもたらすかを定量化するものである。情報理論の創始者であるShannon (1948) によって導入された。出現確率が低い事象ほど、多くの情報を含むと考えられる。情報量は次のように定義される。

$$
\begin{equation}
\mathbb{I}(x) \coloneqq \ln\left(\frac{1}{p(x)}\right) = -\ln p(x)
\end{equation}
$$

**エントロピー (Entropy)** は、確率変数の持つ平均的な不確実性、すなわち平均情報量を表す。離散的な場合には和を、連続的な場合には積分を用いて定義されるが、ここでは連続的な場合を考える。エントロピーは以下のように定義される。

$$
\begin{equation}
\mathbb{H}(x) \coloneqq \mathbb{E}[-\ln p(x)] = -\int p(x) \ln p(x)\,\mathrm{d}x
\end{equation}
$$

また、条件付きエントロピー$\mathbb{H}(x|y)$は、$y$が与えられたときの$x$の不確実性を測る指標であり、次のように定義される。

$$
\begin{equation}
\mathbb{H}(x \vert y) \coloneqq \mathbb{E}_{x,y}[-\ln p(x \vert y)]
\end{equation}
$$

この期待値は、$p(x,y)$に基づいて計算される。

次に、**Kullback-Leibler情報量 (KL divergence)** は、ある確率分布$p(x)$と別の分布$q(x)$の間の「距離」あるいは「ずれ」を測る尺度である。対称性は持たないため、厳密には距離ではないが、情報理論や機械学習において極めて重要な概念である。KLダイバージェンスは以下のように定義される。

$$
\begin{align}
D_{\text{KL}}(p(x)\Vert q(x)) &\coloneqq \int p(x) \ln \frac{p(x)}{q(x)} \mathrm{d}x \\
&= \int p(x) \ln p(x)\,\mathrm{d}x - \int p(x) \ln q(x)\,\mathrm{d}x \\
&= \mathbb{E}_{x\sim p(x)}[\ln p(x)] - \mathbb{E}_{x\sim p(x)}[\ln q(x)] \\
&= -\mathbb{H}(x) - \mathbb{E}_{x\sim p(x)}[\ln q(x)]
\end{align}
$$

最後に、**相互情報量 (Mutual Information)** は、二つの確率変数$x$と$y$の間にどれほどの情報的関連性があるか、すなわち$y$を知ることによって$x$の不確実性がどれほど減少するかを定量化する。相互情報量は、エントロピーの差として次のように定義される。

$$
\begin{equation}
\mathbb{I}(x;y) \coloneqq \mathbb{H}(x) - \mathbb{H}(x\vert y)
\end{equation}
$$

これはまた、対称的な形でも書ける。

$$
\begin{equation}
\mathbb{I}(x;y) = \mathbb{H}(x) + \mathbb{H}(y) - \mathbb{H}(x,y)
\end{equation}
$$

あるいは、確率分布の比を使って次のようにも表現される。

$$
\begin{equation}
\mathbb{I}(x;y) = \int p(x,y) \ln \frac{p(x,y)}{p(x)p(y)} \mathrm{d}xdy
\end{equation}
$$

この表現は、相互情報量が、$p(x,y)$と$p(x)p(y)$のKLダイバージェンスであることを示しており、すなわち独立であれば情報共有はゼロであることを意味する。

### 確率過程
**確率過程**（stochastic process）とは、時間とともに変化する確率変数の集まりを指す。日常においても、株価の変動、気温の変化、人の行動など、未来の状態が確実には予測できず、ある程度の不確実性を含む現象は数多く存在する。これらの確率的な時間変化を数学的に扱う枠組みが確率過程である。

形式的には、確率過程とは、ある時間 $t$ における確率変数$X_t$の集まり$\{X_t\}_{t \in \mathcal{T}}$のことである。ここで、$\mathcal{T}$は時間を表す集，たとえば$\mathcal{T} = \{0,1,2,\dots\}$や $\mathcal{T} = [0,\infty)$ であり，各$X_t$はある量（たとえば位置や価格など）を表す確率変数である。

**Markov過程**（Markov process）は、確率過程の中でも特に重要なクラスに属する。最大の特徴は、「現在の状態が分かれば、未来の状態の予測に過去の情報（履歴）は不要である」という性質、すなわち**Markov性**を持つことである。

この性質は次のように定式化される：

$$
\begin{equation}
P(X_{t+1} = x \mid X_t = x_t, X_{t-1} = x_{t-1}, \dots, X_0 = x_0) = P(X_{t+1} = x \mid X_t = x_t)
\end{equation}
$$

すなわち、未来の状態$X_{t+1}$の確率は、現在の状態$X_t$のみに依存し、それ以前の状態には依存しない。これにより、Markov過程は過去の情報を逐一保持する必要がなく、解析やシミュレーションが容易になるという利点を持つ。

**Wiener過程** (Wiener process) は、連続時間Markov過程の代表例であり、ブラウン運動とも呼ばれる。微粒子が液体中で不規則に運動する現象（ブラウン運動）を数学的に記述するために導入されたものであるが、現在では金融や情報理論をはじめとした多くの分野において基本的なモデルとなっている。

Wiener過程$\{W_t\}_{t \ge 0}$は、以下の性質を満たす確率過程である：

- 初期値が 0 である：$W_0 = 0$。
- 増分の独立性：任意の$0 \le t_1 < t_2 < \dots < t_n$に対し、各増分$W_{t_{i+1}} - W_{t_i}$は互いに独立である。
- 増分の分布が正規分布に従う：任意の$s < t$に対して、増分$W_t - W_s$は平均 0、分散$t - s$の正規分布$N(0, t - s)$に従う。
- パスが連続である：時間$t$に対する関数$W_t$は、ほとんど確実に連続である。

このような性質により、Wiener過程は時間的に連続かつ不規則に変化するランダムな運動を表現することができる。また、Wiener過程はガウス過程でもあり、その共分散関数は$\mathbb{E}[W_s W_t] = \min(s, t)$で与えられる。

さらに、Wiener過程はスケーリングの性質も持つ。すなわち、任意の$c > 0$に対して、$\{\sqrt{c}W_{t/c}\}_{t \ge 0}$もまたWiener過程である。このような対称性により、Wiener過程は確率解析における中心的な対象として位置づけられている。

以上のように、確率過程はランダムな時間変化を記述するための基本概念であり、Markov過程やWiener過程はその中でも特に重要な例である。それぞれの性質を理解することは、確率論や統計物理、数理ファイナンス、機械学習などの応用分野において基礎的かつ不可欠である。

### 確率微分方程式
神経活動を含む生体活動には様々なゆらぎ（ノイズ）が常に存在しており、神経モデルにおいてもこれを考慮する必要がある。神経活動のダイナミクスを連続時間で記述する際には，決定論的な時間変化に加えて確率的なゆらぎ（ノイズ）を含む微分方程式，すなわち確率微分方程式（Stochastic Differential Equation; SDE）を用いることがある。SDEの一般的な形は以下のように与えられる：

$$
\begin{equation}
\mathrm{d}x(t) = f(X(t), t)\,\mathrm{d}t + g(X(t), t)\,dW(t)
\end{equation}
$$

ここで、$f(X(t), t)$はドリフト項と呼ばれる決定論的な変化、$g(X(t), t)$は拡散項と呼ばれるノイズの強度を表す関数、$W(t)$は標準ブラウン運動（Wiener過程）である。$W(t)$は連続時間の確率過程であり、その増分$W(t + \Delta t) - W(t)$は平均0、分散$\Delta t$の正規分布$\mathcal{N}(0, \Delta t)$に従う。


- 離散モデルでのノイズ$\mathbf{w}_t$,$\mathbf{v}_t$は各離散時刻ごとにガウス分布から独立にサンプルされるもので、各離散時点に明示的に加えられる雑音です。

- 一方、連続モデルのノイズ$d\mathbf{w}(t), d\mathbf{v}(t)$はブラウン運動の微小増分を表しており、連続時間での微小な変動をモデル化しています。

- このように連続時間モデルと離散時間モデルは形式上対応しています。離散化する際には通常、
$$
\begin{equation}
\mathbf{w}_{t} \approx \int_{t}^{t+\Delta t} d\mathbf{w}(s),\quad
\mathbf{v}_{t} \approx \int_{t}^{t+\Delta t} d\mathbf{v}(s)
\end{equation}
$$
という対応を用いて導出します。

神経活動には**ノイズ**（neuronal noise）が常に存在しており、神経モデルにおいてもこれを考慮する必要がある。そのため、シナプス入力にノイズを加えることがある。たとえば、Leaky Integrate-and-Fire（LIF）モデルにおける膜電位の力学にノイズを加える場合を考える。ノイズ$\xi(t)$を平均$\tilde{\mu}$、分散$\tilde{\sigma}^2$の正規分布$\mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)$に従うガウシアンノイズとすると、膜電位$V_m(t)$の時間発展は次式で記述される：

$$
\begin{equation}
\tau_m \frac{dV_m(t)}{\mathrm{d}t} = -(V_m(t) - V_\text{rest}) + R_m I(t) + \xi(t)
\end{equation}
$$

このように、線形のドリフト項$-(V_m(t) - V_\text{rest})$とガウシアンノイズ項$\xi(t)$を含む確率微分方程式（stochastic differential equation; SDE）で表される確率過程は、**Ornstein–Uhlenbeck（OU）過程** と呼ばれる。ノイズ$\xi(t)$が標準正規分布$\mathcal{N}(0, 1)$に従うホワイトノイズ$\eta(t)$を用いて$\xi(t) = \tilde{\mu} + \tilde{\sigma} \eta(t)$と表すこともできる。

さらに、$\xi(t)$が発火率$\lambda$のポアソン過程に従う場合を考える。シナプス前細胞の数を$N_\text{pre}$、$i$番目のシナプスにおけるシナプス強度に比例する定数を$J_i$とすると、ノイズの平均と分散はそれぞれ$\tilde{\mu} = \langle J_i \rangle N_\text{pre} \cdot \lambda$、$\tilde{\sigma}^2 = \langle J_i^2 \rangle N_\text{pre} \cdot \lambda$と書ける。ただし、$\langle \cdot \rangle$は平均を意味する。このような連続的なガウス過程でポアソン入力を近似する手法を**拡散近似**（diffusion approximation）と呼び、これは**Campbellの定理**に基づいて導かれる。

このような確率微分方程式を数値的にシミュレーションするためには、時間離散化が必要となるが、その際には注意が必要である。たとえば、ドリフト項を省略し、ノイズ項のみを残した場合、

$$
\begin{equation}
\tau_m \frac{dV_m(t)}{\mathrm{d}t} = \xi(t)
\end{equation}
$$

となる。この式を時間ステップ$\Delta t$でEuler法により離散化すると、

$$
\begin{equation}
V_m(t + \Delta t) = V_m(t) + \frac{1}{\tau_m} \xi_1(t)
\end{equation}
$$

と書ける。ここで、時間ステップを$\Delta t$から$\Delta t/2$に変更して同様に離散化すると、

$$
\begin{align}
V_m(t + \Delta t) &= V_m(t + \Delta t/2) + \frac{1}{\tau_m} \xi_1(t) \\
&= V_m(t) + \frac{1}{\tau_m} \left[ \xi_1(t) + \xi_2(t) \right]
\end{align}
$$

となる。ノイズ項$\xi_1(t)$と$\xi_2(t)$は互いに独立と仮定すると、それぞれの標準偏差は$\tilde{\sigma}/\tau_m$であり、その和$\xi_1(t) + \xi_2(t)$の分散は$2\tilde{\sigma}^2$、すなわち標準偏差は$\sqrt{2} \tilde{\sigma}/\tau_m$となる。これは時間ステップの取り方によってノイズ項の大きさが変化することを意味しており、正確なシミュレーションのためには問題となる。したがって、時間ステップに依存しないようノイズ項をスケーリングする必要があり、そのためにはノイズに$\sqrt{\Delta t}$を掛けることで対処できる。すなわち、離散化式は以下のように修正するのが望ましい：

$$
\begin{equation}
V_m(t + \Delta t) = V_m(t) + \frac{\sqrt{\Delta t}}{\tau_m} \xi_1(t)
\end{equation}
$$

このように修正することで、時間ステップに依存しない安定なノイズスケーリングが可能となる。このように確率微分方程式をEuler法で離散化する方法は、**Euler–Maruyama法**と呼ばれる．他の離散化手法としては、Milstein法なども存在する。

このようなSDEを解析的に解くことは一般に困難であるため、数値的な近似解法が必要となる。Euler–Maruyama法は、その最も基本的な手法の一つであり、常微分方程式に対するEuler法の自然な拡張である。時間を刻み幅$\Delta t$で離散化し、$t_n = n \Delta t$、$X_n \approx X(t_n)$とおくと、Euler–Maruyama法は次のように与えられる：

$$
\begin{equation}
X_{n+1} = X_n + f(X_n, t_n)\, \Delta t + g(X_n, t_n)\, \Delta W_n
\end{equation}
$$

ここで$\Delta W_n = W(t_{n+1}) - W(t_n)$は、正規分布$\mathcal{N}(0, \Delta t)$に従う確率変数である。実装上は、$\Delta W_n$を標準正規分布$\mathcal{N}(0, 1)$に従う独立な乱数$\eta_n$を用いて$\Delta W_n = \sqrt{\Delta t} \cdot \eta_n$と近似する。この結果、Euler–Maruyama法は以下のように書き換えられる：

$$
\begin{equation}
X_{n+1} = X_n + f(X_n, t_n)\, \Delta t + g(X_n, t_n)\, \sqrt{\Delta t} \cdot \eta_n
\end{equation}
$$

この手法は簡便でありながら、確率過程の時間発展を模倣するのに有用であり、多くのシミュレーションにおいて第一選択となる。ただし、その強収束次数は$0.5$であり、すなわち刻み幅$\Delta t$を小さくしても誤差は$\mathcal{O}(\sqrt{\Delta t})$でしか減少しない。そのため、より高精度な数値解が必要な場合には、Milstein法などの高次手法の導入が検討される。Euler–Maruyama法は、その単純さと広範な適用性から、確率微分方程式の数値解析における基本的な出発点となる手法である。

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
**数理モデル**とは、実世界の現象、システム、または過程を数学的対象を用いて表現する枠組みである。モデルは変数とパラメータから構成され、これらの間の関係性は数式、方程式、確率分布、または集合論的記述によって定式化される。数理モデルは現象の分析、予測、制御、最適化などに用いられる。

**機械学習** (machine learning) における**モデル** (model) とは，2つの集合$\mathcal{X}, \mathcal{Y}$ を仮定したとき，入力 $x \in \mathcal{X}$ を出力 $y \in \mathcal{Y}$ に対応づける関数 $f: \mathcal{X} \to \mathcal{Y}$ または条件付き確率分布 $p(y\mid x)$ を指す．モデルは内部に学習可能なパラメータ (parameter) $\theta$を持ち，このパラメータを調整することによって入力と出力の関係を最適化する．

機械学習での**学習** (learning) とは，観測データに基づき，$y = f(x; \theta)$ あるいは $p(y\mid x; \theta)$ をよりよく近似するパラメータ $\theta$ を求める過程である．モデルを学習させることを**訓練** (training) と呼ぶ．このとき，データ集合 $D = \{(x_i, y_i)\}$ を用いて，目的関数（損失関数）を最小化または尤度を最大化することでパラメータ $\theta$ を更新する．学習によって得られた最適なパラメータ $\theta^*$ を用い，未知の入力 $x$ に対して出力 $\hat{y}$ を推定することを**予測** (prediction) と呼ぶ．推定値 $\hat{y}$ の取得には，$p(y \mid x;\theta^*)$ からのサンプリング，$\mathrm{argmax}_y\ p(y \mid x; \theta^*)$ の計算，あるいは期待値 $\mathbb{E}[y \mid x; \theta^*]$ の計算などが用いられる．

$y$ が既知の場合，データ集合 $D=\{(x,y)\}$ は**教師付きデータ** (labeled data) と呼ばれ，この対応関係を学習する過程を**教師あり学習** (supervised learning) という．一方，$y$ が未知で $x$ のみが与えられる場合，$D=\{x\}$ は**ラベルなしデータ**と呼ばれ，その潜在構造や分布を推定する過程を**教師なし学習** (unsupervised learning) という．教師なし学習の典型例にはクラスタリングや次元削減がある．また，ラベルありデータとラベルなしデータの両方を用いる学習を**半教師あり学習** (semi-supervised learning)，ラベルなしデータから自己生成したラベルを用いて教師あり学習の形式で学習する手法を**自己教師あり学習** (self-supervised learning) という．さらに，機械学習の重要な分野のひとつに**強化学習** (reinforcement learning) がある．強化学習では，環境と相互作用しながら行動を選択するエージェントを仮定し，逐次的な意思決定過程において，累積報酬を最大化するための方策を学習する．詳細は第11章で述べる．

### 回帰と分類
機械学習の課題は、大きく回帰 (regression) と分類 (classification) に分けられる。回帰とは、入力$x$に対して連続的な出力$y \in \mathbb{R}$または$\mathbb{R}^d$を予測する問題を指す。典型的な例としては、住宅価格の予測や気温の予測などが挙げられる。これに対し分類とは、入力$x$に対して離散的なクラスラベル$y \in {1, \dots, K}$を予測する問題を指し、例えば画像から猫・犬・鳥を識別する場合がこれに相当する。回帰と分類はいずれも入力と出力の関係を学習するが、出力空間の性質（連続 vs 離散）により目的関数やモデルの設計が異なる。なお、回帰問題においても、出力を閾値によって離散化することで分類問題に転換することが可能であり、両者の境界は必ずしも絶対的ではない。

### 識別モデルと生成モデル
入力と出力の関係を学習する方法には、識別モデル (discriminative model) と生成モデル (generative model) という分類がある。識別モデルとは、入力$x$が与えられたときに出力$y$を直接推定する条件付き確率分布$p(y|x)$を学習するモデルを指す。これに対し生成モデルとは、入力と出力の同時確率分布$p(x,y)$、あるいは入力$x$の分布$p(x)$と出力$y$に条件づけた生成分布$p(x|y)$を学習するモデルを指す。生成モデルを用いることで、データのサンプリングや異常検知、データ補完など幅広い応用が可能となる。識別モデルは予測性能に特化する一方、生成モデルはデータ分布そのものの理解と生成に重点を置くという違いがある。

### オフライン学習とオンライン学習
学習の方法は、利用可能なデータと学習過程の違いに応じて、**オフライン学習** (offline learning) と**オンライン学習** (online learning) に分類できる。オフライン学習では、すべての訓練データ $D=\{(x_i, y_i)\}$ があらかじめ揃っており、この固定されたデータに対して繰り返し学習を行う。オフライン学習は一括学習あるいはバッチ学習 (batch learning) とも呼ばれ、典型的な深層学習や統計的推論はこの枠組みに基づいている。

これに対してオンライン学習では、データは逐次的に到着し、到着するたびにモデルのパラメータを即座に更新する。このためオンライン学習は逐次学習 (sequential learning) とも呼ばれ、環境が変動する場合やリアルタイム処理が求められる場合に有効である。

このオフライン学習とオンライン学習の違いは、学習の**相** (phase) の構造にも現れる。オフライン学習は、モデルから推定値を取得する**推論相** (inference phase) と、パラメータを更新する**訓練相** (training phase) の二つの相を明確に区別して持つ。例えばニューラルネットワークにおいては、順伝播が推論相、逆伝播とパラメータ更新が訓練相に対応する。一方、オンライン学習は基本的に**単相** (single phase) であり、推論と更新が逐次的かつ並行して行われるため、明示的に分離された相を持たない。

生物学的な脳において、オフライン学習とオンライン学習のいずれが行われているかについては、現代の神経科学においても明確な結論は得られていない。ただし、一般的な知見として、睡眠中に記憶再生を通じて運動記憶が固定されるオフライン学習過程（Shadmehr & Brashers-Krug, 1997; Albouy et al., 2008）と、覚醒中の感覚入力に応じて逐次的な誤差修正が行われるオンライン学習過程（Shadmehr et al., 2010）がともに存在すると考えられている。

実際、最近のメタ分析研究（Byczynski et al., 2025）は、運動学習課題における脳活動のパターンを体系的に比較することにより、試行中に進行する**オンライン学習**と試行間に進行する**オフライン学習**がいずれも明確に存在することを示した。この研究では、両者に共通する領域（補足運動野や体性感覚野）と、それぞれに特有な領域が同定され、運動獲得と記憶固定という異なるプロセスにおいて異なる神経基盤が働いていることが示唆された。これにより、脳におけるオンライン・オフライン学習の並立モデルが一層支持された。

https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00457/127405/Neural-signatures-of-online-and-offline-motor

## 線形回帰
**線形回帰モデル**（linear regression）は、与えられた説明変数（explanatory variable）$\mathbf{x}$に基づいて、目的変数（objective variable）$y$を線形に予測することを目的とする。

説明変数の次元が$p$であるとき、線形回帰モデルは次のように表される：

$$
\begin{equation}
y = w_0 + w_1x_1 + \cdots + w_px_p + \varepsilon = w_0 + \sum_{j=1}^p w_j x_j + \varepsilon
\end{equation}
$$

ここで$w_0$は切片（バイアス項）、$w_1, \dots, w_p$は各説明変数に対する重み、$\varepsilon$は誤差項を表す。$p = 1$の場合を**単回帰**（*simple regression*）、$p > 1$の場合を**重回帰**（*multiple regression*）と呼ぶ。

### 回帰モデルの行列表現

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

### 最小二乗法
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

#### 正規方程式による解析解
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

#### 勾配法による数値的推定
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