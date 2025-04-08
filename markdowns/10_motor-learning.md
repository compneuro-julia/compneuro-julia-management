# 第10章：運動学習と最適制御
## 運動制御と最適制御
ここまでは神経系のみで閉じていたが，本章と次章「強化学習」では，神経とその他外界との相互作用について説明を行う．

身体の運動には、四肢運動，頭部運動，眼球運動（サッケード）など多様な形式が存在する。これらの運動は、大脳皮質、大脳基底核、小脳、脊髄路、末梢運動神経など、複数の神経系の協調的な活動によって制御されている。運動が制御されているとは，実質的には複数の筋肉の収縮を制御するということを意味する．

そのため，筋疾患や整形外科的疾患による運動障害を除外した場合，運動にかかわる神経の障害は運動の障害として外界に表出される．脳の画像検査や侵襲的検査なしで神経の様子を大まかに推し量れるため，運動は神経の「窓」であるともいえる．
（もちろん，診断のためには上記の検査も必要不可欠である）

このような視点は、臨床医学においても重要である。すなわち、運動機能の障害の様子を丁寧に観察することで、どの神経系に異常が生じているのかを推定する手がかりとなる。また、一般論として、他者の精神状態や意図は行動に表れることが知られており、運動もその例外ではない。したがって、運動は神経学的にも心理学的にも、内部状態を外部に表現する媒体であると捉えることができる。

では、病的な運動と正常な運動は何が異なるのだろうか。病的な運動には、しばしば「粗大である」「時間がかかる」「非効率である」といった特徴が見られる。具体的には、目的に対して必要以上の動作が含まれる、あるいは目的達成までに過剰なエネルギーや時間を要する、といった観察がなされる。一方、正常な運動は滑らかで効率的であり、目的に沿った最小限のエネルギーで遂行される傾向がある。「正常」と「病的」の比較以外にも，例えば日常生活で行わない動作，武道や器械体操，バスケットボールをゴールに投げ入れるなどであれば，初心者と熟練者で運動の効率性は異なる．

この運動における「効率性」や「非効率性」を定量的に評価するための理論枠組みとして、最適制御理論がある。最適制御においては、所定の目的関数を最小化するような運動を求めることが基本的な問題設定となる。したがって、実際の動物やヒトの運動がどのような目的関数の下で最適化されているかを明らかにすることは、運動の本質的理解に寄与すると考えられる。

ただし、ここで重要なのは、脳内に目的関数が明示的にコードされており、それを数学的な最適化手法によって最小化している、という機構的仮説を主張するものではないという点である。本章では、あくまでも運動の特徴を理解するための理論的手段として最適制御の枠組みを用いることに留意されたい。

## 状態空間モデル
最適制御のモデルは状態空間モデル (state-space model) により記述される．
(修正する)

状態空間モデルとは、時間発展する力学系の状態の変化と出力を、ベクトルと行列を用いて表現する数学的枠組みである。一般に、あるシステムが時刻$t$において内部状態$\mathbf{x}(t) \in \mathbb{R}^n$、外部からの入力$\mathbf{u}(t) \in \mathbb{R}^m$、および観測可能な出力$\mathbf{y}(t) \in \mathbb{R}^p$をもつとき、その振る舞いは以下のような**一階の微分方程式**と**代数方程式**によって記述される：

$$
\begin{aligned}
\dot{\mathbf{x}}(t) &= \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t), \\
\mathbf{y}(t) &= \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t).
\end{aligned}
$$

ここで、行列$\mathbf{A} \in \mathbb{R}^{n \times n}$は状態遷移行列と呼ばれ、状態がどのように時間発展するかを決定する。行列$\mathbf{B} \in \mathbb{R}^{n \times m}$は入力が状態に与える影響を、$\mathbf{C} \in \mathbb{R}^{p \times n}$は状態が出力に与える影響を、$\mathbf{D} \in \mathbb{R}^{p \times m}$は入力が直接出力に与える影響を表す。これは連続時間系における線形状態空間モデルであり、離散時間系の場合には微分方程式を差分方程式に置き換えることで表現される：

$$
\begin{aligned}
\mathbf{x}_{k+1} &= \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k, \\
\mathbf{y}_k &= \mathbf{C}\mathbf{x}_k + \mathbf{D}\mathbf{u}_k,
\end{aligned}
$$

ここで$k$は離散時刻を表す。

このようなモデルは、制御理論や信号処理、ロボティクス、経済学、生体計測などさまざまな分野で応用されており、特に複雑な高次元システムをコンパクトに扱うことができる点が大きな利点である。また、モデルを非線形系に拡張することで、一般の非線形動的システムも同様の枠組みで扱うことができる。非線形状態空間モデルでは、状態と出力の式は次のように一般化される：

$$
\begin{aligned}
\dot{\mathbf{x}}(t) &= f(\mathbf{x}(t), \mathbf{u}(t)), \\
\mathbf{y}(t) &= g(\mathbf{x}(t), \mathbf{u}(t)),
\end{aligned}
$$

ここで$f$および$g$は非線形関数である。さらに、観測ノイズやシステムノイズを導入することで確率的な状態空間モデルを定式化でき、これに基づく代表的な手法としてカルマンフィルタや拡張カルマンフィルタ、粒子フィルタなどが存在する。

このように、状態空間モデルは、システムの内部状態と出力の関係を時間的に記述するための統一的かつ強力な理論的枠組みである。

## 躍度最小モデル
躍度最小モデル (minimum-jerk model; {cite:p}`Flash1985-vj`)を実装する．解析的に求まるが以下では二次計画法を用いて数値的に求める．

### 変分法による解法
位置のベクトルを$\mathbf{x}(t) \in \mathbb{R}^n$とする．位置を1, 2, 3回微分したものをそれぞれ，速度，加速度，躍度と呼ぶ．躍度最小モデルでは，運動過程における躍度のノルムの二乗の総和を最小化することを目的とする．目的関数 $J$は

$$
J[\mathbf{x}(t)] := \int_0^T \left\| \frac{d^3 \mathbf{x}(t)}{dt^3} \right\|^2 dt = \int_0^T \sum_{i=1}^n \left( {\dddot{x}}_i(t) \right)^2 dt = \sum_{i=1}^n  \int_0^T \left( {\dddot{x}}_i(t) \right)^2 dt
$$

ここで，$J$は $L=\left\| \dfrac{d^3 \mathbf{x}(t)}{dt^3} \right\|^2$の汎関数であり，$L$はLagrangianである．

各成分は独立なので，各成分ごとに最小化問題を解けば良いことがわかる．$x_i$を $x$と単純に表記する．関数 $x(t)$に微小な摂動 $\epsilon \eta (t)$を加え，

$$
x(t) \to x(t) + \epsilon \eta (t)
$$

とする．この場合，汎関数は

$$
\begin{align}
J[x + \epsilon \eta] &= \int_0^T \left( \dddot{x}(t) + \epsilon \dddot{\eta}(t) \right)^2 dt
\end{align}
$$

となるので，汎関数微分をすると，

$$
\begin{align}
\delta J = \left. \frac{d}{d\epsilon} J[x + \epsilon \eta] \right|_{\epsilon = 0}&=\lim_{\epsilon \to 0} \int_0^T \frac{(\dddot{x}(t))^2 + 2 \epsilon \dddot{\eta}(t) + (\epsilon \dddot{\eta}(t) )^2}{\epsilon}\\
&= \int_0^T 2 \dddot{x}(t) \dddot{\eta}(t) dt
\end{align}
$$

ここで，$\delta J=\int f(t)\eta(t) dt$の形を目指す．こうすると任意の $\eta$に対する極値条件を書くことができる．
部分積分を3回繰り返すことにより，

$$
\begin{align}
\int_0^T \dddot{x} \dddot{\eta} dt &= \left[\dddot{x}\ddot{\eta}\right]_{0}^T-\int_0^T x^{(4)} \ddot{\eta} dt\\
&=\left[\dddot{x}\ddot{\eta}\right]_{0}^T - \left[x^{(4)} \dot{\eta}\right]_{0}^T + \int_0^T x^{(5)} \dot{\eta} dt\\
&=\left[\dddot{x}\ddot{\eta}\right]_{0}^T - \left[x^{(4)} \dot{\eta}\right]_{0}^T + \left[x^{(5)} \eta\right]_{0}^T - \int_0^T x^{(6)} \eta dt\\
\end{align}
$$

境界条件 $\eta^{(i)}(t')=0\ (i=0, 1, 2;\ t'=0, T)$とすることで，

$$
\delta J=- 2\int_0^T x^{(6)} \eta dt
$$

となる．任意の$\eta$に対して，$\delta J=0$である場合，$x^{(6)}=0$となる必要があるので，$x$は高々5次の多項式で表せる．

$$
x(t)=\sum_{i=0}^5 a_i t^i=a_0+a_1 t+a_2 t^2+a_3 t^3+a_4 t^4+a_5 t^5
$$

微分すると 

$$
\begin{align}
\dot{x}(t)&=a_1 + 2a_2 t+3a_3 t^2+4a_4 t^3+5a_5 t^4\\
\ddot{x}(t)&=2a_2+6a_3 t+12a_4 t^2+20a_5 t^3\\
\end{align}
$$

となるので，境界条件より，$\dot{x}(0)=\dot{x}(T)=0$および $\ddot{x}(0)=\ddot{x}(T)=0$とすると，$a_0=x_0, a_1=a_2=0$であり，$T>0$より，

$$
\begin{align}
\begin{cases}
a_3+T a_4 +T^2 a_5 &=\dfrac{x_T-x_0}{T^3}\\
3a_3 +4T a_4 +5T^2 a_5 &=0\\
6a_3+12T a_4 +20T^2 a_5 &=0\\
\end{cases}
\end{align}
$$

これを解くと，$a_3=\dfrac{10(x_T-x_0)}{T^3}, a_4=-\dfrac{15(x_T-x_0)}{T^4}, a_5=\dfrac{6(x_T-x_0)}{T^5}$
となるので，

$$
x(t)=x_0+(x_T-x_0) \left[10 \left(\frac{t}{T}\right)^3-15 \left(\frac{t}{T}\right)^4-6 \left(\frac{t}{T}\right)^5\right]
$$

と求まる．


### 二次計画法による解法

#### 等式制約下の二次計画法 (Equality Constrained Quadratic Programming)

$n$個の変数があり，$m$個の制約条件がある等式制約下の二次計画問題を考える．$\mathbf {x}\in \mathbb{R}^n$, 対称行列$\mathbf{P}\in \mathbb{R}^{n\times n}$,  $\mathbf {q}\in \mathbb{R}^{n}$, $\mathbf{A}\in \mathbb{R}^{m\times n}$, $\mathbf {b}\in \mathbb{R}^m$．このとき，問題は次のようになる．

$$
\begin{align}
&{\text{Minimize}}\quad {\frac {1}{2}}\mathbf {x}^\top \mathbf{P}\mathbf {x} +\mathbf {q} ^{\top}\mathbf {x}\\
&{\text{subject to}}\quad \mathbf{A}\mathbf {x} =\mathbf {b}
\end{align}
$$

Lagrangeの未定乗数法を用いると解は

$$
\begin{equation}
{\begin{bmatrix}\mathbf{P}&\mathbf{A}^\top\\\mathbf{A}&\mathbf{0}\end{bmatrix}}{\begin{bmatrix}\mathbf {x} \\
\lambda \end{bmatrix}}={\begin{bmatrix}-\mathbf {q} \\\mathbf {b} \end{bmatrix}}
\end{equation}
$$

の解として与えられる．ここで $\lambda \in \mathbb{R}^{m}$ はLagrange乗数のベクトルである．

#### 躍度最小モデルの実装
1次元における運動を考えよう．この仮定ではサッカードするときの眼球運動などが当てはまる．以下では {cite:p}`Yazdani2012-sx` での問題設定を用いる．Toeplitz行列を用いた実装はYazdaniらのPythonでcvxoptを用いた実装を参考にして作成した．

問題設定は以下のようにする．

$$
\begin{align}
&\underset{u(t)}{\operatorname{minimize}}\quad \|u(t)\|_2 \\
&\text{subject to} \quad \dot{\mathbf{x}}(t)=A \mathbf{x}(t)+B u(t)
\end{align}
$$

ただし，$\|\cdot\|_{2}$は $L_{2}$ノルムを意味し，$A=\left[\begin{array}{lll}0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0\end{array}\right], B=\left[\begin{array}{l}0 \\ 0 \\ 1\end{array}\right], \mathbf{x}(t)=\left[\begin{array}{l}x(t) \\ \dot{x}(t) \\ \ddot{x}(t)\end{array}\right], u(t)=\dddot x(t)$とする．すなわち，制御信号 $u(t)$は躍度 $\dddot x(t)$と等しいとする．

### 中継点問題

## トルク変化最小モデル
ダイナミクスのみキネティクス

腕のモデルを考えていない場合，

## 終点誤差分散最小モデル
HarrisおよびWolpertは制御信号の大きさに従い，ノイズが生じるモデルを提案した．さらにこのモデルにおいて，状態の分散が可能な限り小さくなるような制御信号を求めた．これを**終点誤差分散最小モデル** (minimum-variance model; {cite:p}`Harris1998-gj`)と呼ぶ．

終点誤差分散最小モデルは状態$\mathbf{x}_t\in \mathbb{R}^n$, 制御信号$\mathbf{u}_t \in \mathbb{R}^p$とし，$\mathbf{A}\in \mathbb{R}^{n\times n}$, $\mathbf{B}\in \mathbb{R}^{n \times p}$とすると，

$$
\begin{equation}
\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \mathbf{B}\mathbf{u}_t (1+\boldsymbol{\xi}_t)
\end{equation}
$$

と表せる．ただし，$\boldsymbol{\xi}_t \sim \mathcal{N}(0, k\mathbf{I})\ (k>0)$である．このため，$\mathbf{u}_t (1+\xi_t)$の平均は $\mathbf{u}_t$, 分散共分散行列は $k\mathbf{u}_t \mathbf{u}_t^\top$となる．$\mathbf{x}_t$を過去の状態 $\mathbf{x}_{t'}\ (t'=0, \ldots, t-1)$で表すと，

$$
\begin{align}
\mathbf{x}_{t} &= \mathbf{A} \mathbf{x}_{t-1} + \mathbf{B}\mathbf{u}_{t-1} (1+\boldsymbol{\xi}_{t-1})\\
&=\mathbf{A}^2 \mathbf{x}_{t-2} + \mathbf{A}\mathbf{B}\mathbf{u}_{t-2} (1+\boldsymbol{\xi}_{t-2}) + \mathbf{B}\mathbf{u}_{t-1} (1+\boldsymbol{\xi}_{t-1})\\
&=\cdots\\
&=\mathbf{A}^{t} \mathbf{x}_{0} + \sum_{t'=0}^{t-1} \mathbf{A}^{t-t'-1}\mathbf{B}\mathbf{u}_{t'} (1+\boldsymbol{\xi}_{t'})\\
\end{align}
$$

となるので，$\mathbf{x}_t$の平均と分散共分散行列はそれぞれ，

$$
\begin{align}
\mathbb{E}\left[\mathbf{x}_{t}\right]&=\mathbf{A}^{t} \mathbf{x}_{0}+\sum_{t'=0}^{t-1} \mathbf{A}^{t-t'-1} \mathbf{B} \mathbf{u}_{t'}\\
\operatorname{Cov}\left[\mathbf{x}_{t}\right]&=k\sum_{t'=0}^{t-1}\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right) \mathbf{u}_{t'} \mathbf{u}_{t'}^\top \left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)^{\top}
\end{align}
$$

となる．制御信号の時系列 $\{\mathbf{u}_t\}$が与えられている場合，状態 $\mathbf{x}_t$の平均と分散共分散行列は，$\mathbb{E}\left[\mathbf{x}_{0}\right]=\mathbf{x}_0, \operatorname{Cov}\left[\mathbf{x}_{0}\right]=\mathbf{0}\in\mathbb{R}^{n\times n}$として，

$$
\begin{align}
\mathbb{E}\left[\mathbf{x}_{t}\right] &=\mathbf{A}\mathbb{E}\left[\mathbf{x}_{t-1}\right] + \mathbf{B} \mathbf{u}_{t-1}\\
\operatorname{Cov}\left[\mathbf{x}_{t}\right]&=\mathbf{A}\operatorname{Cov}\left[\mathbf{x}_{t-1}\right]\mathbf{A}^\top + k\mathbf{B} \mathbf{u}_{t-1} \mathbf{u}_{t-1}^\top \mathbf{B}^\top
\end{align}
$$

と逐次的に計算が可能である．

このようなモデルにおいて，次の条件を満たす制御信号を求めることを考える．まず，初期状態を$\mathbf{x}_0$, 目標状態を $\mathbf{x}_f$とする．また，運動時間を $T_m$, 運動後時間 (post-movement time) を $T_p$とする．よって1試行にかかる時間は$T:=T_m + T_p$となる．以下では時間は離散化されており，$T_m, T_p, T$は自然数を取るとする．運動後の停留期間である時刻 $T_m\leq t \leq T$において，状態の平均が目標状態と一致する，すなわち

$$
\mathbb{E}\left[\mathbf{x}_{t}\right] = \mathbf{x}_f\quad (T_m\leq t \leq T)
$$

を満たし，位置の分散

$$
\mathcal{F}=\sum_{i\in \mathrm{Pos.}}\left[\sum_{t=T_m}^{T} \operatorname{Cov}\left[\mathbf{x}_{t}\right]\right]_{i, i}
$$

を最小にするような制御信号 $\mathbf{u}_t$を求める．ただし，$\mathrm{Pos.}$は状態 $\mathbf{x}_t$の中で位置を表す次元の番号 (インデックス) の集合を意味し，$[\cdot]_{i,i}$は行列の$(i,i)$成分を取り出す操作を意味する．この最適化問題を（躍度最小モデルの際にも用いた）等式制約下の二次計画問題で解くことを考える．二次計画問題で解くには，最小化する目的関数と等式制約をそれぞれ

$$
\begin{align}
&{\text{目的関数}}\quad {\frac {1}{2}}\mathbf{u}^\top \mathbf{P}\mathbf{u} +\mathbf{q} ^{\top}\mathbf{u}\\
&{\text{等式制約}}\quad \mathbf{C}\mathbf{u} =\mathbf{d}
\end{align}
$$

の形にする必要がある．ただし，$\mathbf{P}, \mathbf{C}$は行列，$\mathbf{q}, \mathbf{d}$はベクトルである．簡単のため，$p=1$の場合を考慮すると，$\mathbf{u}_t \to u_{t} \in \mathbb{R}$となる．状態信号の時系列をベクトル化し，$\mathbf{u}=[u_t]_{t=0, \ldots, T-1} \in \mathbf{R}^{T}$とする．また，後の結果に影響しないため，$k=1$とする．さらに位置のインデックスを$\mathrm{Pos.}=\{1\}$のみとする．この条件の下，式変形を行うと，目的関数 $\mathcal{F}$は

$$
\begin{align}
\mathcal{F}=\left[\sum_{t=T_m}^{T} \operatorname{Cov}\left[\mathbf{x}_{t}\right]\right]_{1,1}
&=\left[\sum_{t=T_m}^{T}\sum_{t'=0}^{t-1}u_{t'}^2\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right) \left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)^{\top}\right]_{1,1}\\
&=\sum_{t'=0}^{T-1} u_{t'}^2 \sum_{t=\max(t'+1, T_m)}^{T} \left[\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)^{\top} \right]_{1,1}
\end{align}
$$

と書ける．最後の式変形は $u_{t'}^2$を二重総和の外に出すために行った．この操作は次の図における横方向と縦方向の和の順番を交換することに該当する．

ここで $V_{t'}:=\sum_{t=\max(t'+1, T_m)}^{T} \left[\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)\left(\mathbf{A}^{t-t'-1} \mathbf{B}\right)^{\top} \right]_{1,1}$とすると，$\mathbf{P}=\mathrm{diag}(V_0, \ldots, V_{T-1})\in \mathbf{R}^{T\times T}$および $\mathbf{q}=\mathbf{0} \in \mathbf{R}^{T}$と置くことで，$\mathcal{F}=\mathbf{u}^\top \mathbf{P}\mathbf{u}+\mathbf{q} ^{\top}\mathbf{u}$と書ける．この場合，第2項は0であるので，第1項の係数は結果に影響しない．

次に等式制約を求める．$\mathbb{E}\left[\mathbf{x}_{t}\right] = \mathbf{x}_f\quad (T_m\leq t \leq T)$を変形すると，

$$
\begin{equation}
\sum_{t'=0}^{t-1} \mathbf{A}^{t-t'-1} \mathbf{B} u_{t'}=\mathbf{x}_f-\mathbf{A}^{t} \mathbf{x}_{0}
\end{equation}
$$

となる．左辺について

$$
\begin{equation}
\mathbf{C}_{(t-T_m)n+1:(t-T_m+1)n+1,\ t'}=
\begin{cases}
    \mathbf{A}^{t-t'-1} \mathbf{B} & (0\leq t'\leq t-1) \\
    \mathbf{0} & (t\leq t'\leq T-1)
\end{cases}\in \mathbb{R}^n 
\end{equation}
$$

および，右辺について

$$
\begin{equation}
\mathbf{d}_{(t-T_m)n+1:(t-T_m+1)n+1}=\mathbf{x}_f-\mathbf{A}^{t} \mathbf{x}_{0} \in \mathbb{R}^n 
\end{equation}
$$

とすることで，等式制約が書き下せる．ただし，$[\cdot]_{i:j}$はベクトルあるいは行列の $i$番目から $j$番目までを取り出す操作を意味する．このように，$\mathbf{P}, \mathbf{q}, \mathbf{C}, \mathbf{d}$を設定すると，等式制約下の二次計画問題を用いて $\mathbf{u}$を求めることができる．の二次計画問題を用いて $\mathbf{u}$を求める．

## 最適フィードバック制御モデル
ToDo: infiniteOFCと数式の統一を行う．

### 最適フィードバック制御モデルの構造
**最適フィードバック制御モデル** (optimal feedback control; OFC) の特徴として目標軌道を必要としないことが挙げられる．**Kalman フィルタ**による状態推定と**線形2次レギュレーター** (linear-quadratic regurator; LQR) により推定された状態に基づいて運動指令を生成という2つの流れが基本となる．

### 系の状態変化

$$
\begin{align}
&\text {Dynamics} \quad \mathbf{x}_{t+1}=A \mathbf{x}_{t}+B \mathbf{u}_{t}+\boldsymbol{\xi}_{t}+\sum_{i=1}^{c} \varepsilon_{t}^{i} C_{i} \mathbf{u}_{t}\\
&\text {Feedback} \quad \mathbf{y}_{t}=H \mathbf{x}_{t}+\omega_{t}+\sum_{i=1}^{d} \epsilon_{t}^{i} D_{i} \mathbf{x}_{t}\\
&\text{Cost per step}\quad \mathbf{x}_{t}^\top Q_{t} \mathbf{x}_{t}+\mathbf{u}_{t}^\top R \mathbf{u}_{t}
\end{align}
$$

### LQG
加法ノイズしかない場合($C=D=0$)，制御問題は**線形2次ガウシアン** (linear-quadratic-Gaussian; LQG)制御と呼ばれる．


#### 運動制御 (Linear-Quadratic Regulator)

$$
\begin{align}
\mathbf{u}_{t}&=-L_{t} \widehat{\mathbf{x}}_{t}\\
L_{t}&=\left(R+B^{\top} S_{t+1} B\right)^{-1} B^{\top} S_{t+1} A\\
S_{t}&=Q_{t}+A^{\top} S_{t+1}\left(A-B L_{t}\right)\\
s_t &= \mathrm{tr}(S_{t+1}\Omega^\xi) + s_{t+1}; s_T=0
\end{align}
$$

$\boldsymbol{S}_{T}=Q$

#### 状態推定 (Kalman Filter)

$$
\begin{align}
\widehat{\mathbf{x}}_{t+1}&=A \widehat{\mathbf{x}}_{t}+B \mathbf{u}_{t}+K_{t}\left(\mathbf{y}_{t}-H \widehat{\mathbf{x}}_{t}\right)+\boldsymbol{\eta}_{t} \\ 
K_{t}&=A \Sigma_{t} H^{\top}\left(H \Sigma_{t} H^{\top}+\Omega^{\omega}\right)^{-1} \\ 
\Sigma_{t+1}&=\Omega^{\xi}+\left(A-K_{t} H\right) \Sigma_{t} A^{\top}
\end{align}
$$

この場合に限り，運動制御と状態推定を独立させることができる．

#### 一般化LQG
状態および制御依存ノイズがある場合，

#### シミュレーション
信号依存ノイズ Yが入っている場合はLQGとは異なってくる．

$$
\begin{align}
&\mathbf{u}_{t}=-L_{t} \hat{\mathbf{x}}_{t} \\
&L_{t}=\left(B^\top S_{t+1}^{\mathbf{x}} B+R+\sum_{n} C_{n}^\top\left(S_{t+1}^{\mathbf{x}}+S_{t+1}^{\mathrm{e}}\right) C_{n}\right)^{-1} B^\top S_{t+1}^{\mathbf{x}} A \\
&S_{t}^{\mathbf{x}}=Q_{t}+A^\top S_{t+1}^{\mathbf{x}}\left(A-B L_{t}\right) ; \quad S_{T}^{\mathbf{x}}=Q_{T} \\
&S_{t}^{\mathrm{e}}=A^\top S_{t+1}^{\mathbf{x}} B L_t+\left(A-K_{t} H\right)^\top S_{t+1}^{\mathrm{e}}\left(A-K_{t} H\right) ; \quad S_{T}^{\mathrm{e}}=0\\
&s_{t}=\operatorname{tr}\left(S_{t+1}^{\mathrm{x}}\Omega^{\xi}+S_{t+1}^{\mathrm{e}}\left(\Omega^{\xi}+\Omega^{\eta}+K_{t} \Omega^{\omega} K_{t}^{\top}\right)\right)+s_{t+1} ; \quad s_{n}=0 .
\end{align}
$$

$$
\begin{align}
\hat{\mathbf{x}}_{t+1} &=A \hat{\mathbf{x}}_{t}+B \mathbf{u}_{t}+K_{t}\left(\mathbf{y}_{t}-H \hat{\mathbf{x}}_{t}\right) \\
K_{t} &=A \Sigma_{t}^{\mathrm{e}} H^\top\left(H \Sigma_{t}^{\mathrm{e}} H^\top+\Omega^{\omega}\right)^{-1} \\
\Sigma_{t+1}^{\mathrm{e}} &=\left(A-K_{t} H\right) \Sigma_{t}^{\mathrm{e}} A^\top+\sum_{n} C_{n} L_{t} \Sigma_{t}^{\hat{x}} L_{t}^\top C_{n}^\top ; \quad \Sigma_{1}^{\mathrm{e}}=\Sigma_{1} \\
\Sigma_{t+1}^{\hat{\mathbf{x}}} &=K_{t} H \Sigma_{t}^{\mathrm{e}} A^\top+\left(A-B L_{t}\right) \Sigma_{t}^{\hat{\mathbf{x}}}\left(A-B L_{t}\right)^\top ; \quad \Sigma_{1}^{\hat{\mathbf{x}}}=\hat{\mathbf{x}}_{1} \hat{\mathbf{x}}_{1}^\top
\end{align}
$$

### costの計算と誤差の脳内表現
cost per stepは脳内で計算できるのか？

## 無限時間最適フィードバック制御モデル
### モデルの構造
**無限時間最適フィードバック制御モデル** (infinite-horizon optimal feedback control model) {cite:p}`Qian2013-zy`

$$
\begin{align}
d x&=(\mathbf{A} x+\mathbf{B} u) dt +\mathbf{Y} u d \gamma+\mathbf{G} d \omega \\
d y&=\mathbf{C} x dt+\mathbf{D} d \xi\\
d \hat{x}&=(\mathbf{A} \hat{x}+\mathbf{B} u) dt+\mathbf{K}(dy-\mathbf{C} \hat{x} dt)
\end{align}
$$

$$
\begin{align}
\mathbf{X}:=\begin{bmatrix}
x \\
\tilde{x}
\end{bmatrix}, d \bar{\omega} :=\begin{bmatrix}
d \omega \\
d \xi
\end{bmatrix}, \bar{\mathbf{A}} :=\begin{bmatrix}
\mathbf{A}-\mathbf{B} \mathbf{L} & \mathbf{B} \mathbf{L} \\
\mathbf{0} & \mathbf{A}-\mathbf{K} \mathbf{C}
\end{bmatrix}\\
\bar{\mathbf{Y}} :=\begin{bmatrix}
-\mathbf{Y} \mathbf{L} & \mathbf{Y} \mathbf{L} \\
-\mathbf{Y} \mathbf{L} & \mathbf{Y} \mathbf{L}
\end{bmatrix}, \bar{G} :=\begin{bmatrix}
\mathbf{G} & \mathbf{0} \\
\mathbf{G} & -\mathbf{K} \mathbf{D}
\end{bmatrix}
\end{align}
$$

とする．元論文では$F, \bar{F}$が定義されていたが，$F=0$とするため，以後の式から削除した．

$$
\begin{align}
\mathbf{P} &:=\begin{bmatrix}
\mathbf{P}_{11} & \mathbf{P}_{12} \\
\mathbf{P}_{12} & \mathbf{P}_{22}
\end{bmatrix} = \mathbb{E}\left[\mathbf{X} \mathbf{X}^\top\right] \\
\mathbf{V} &:=\begin{bmatrix}
\mathbf{Q}+\mathbf{L}^\top R \mathbf{L} & -\mathbf{L}^\top R \mathbf{L} \\
-\mathbf{L}^\top R \mathbf{L} & \mathbf{L}^\top R \mathbf{L}+\mathbf{U}
\end{bmatrix}
\end{align}
$$

aaa
$$
\begin{align}
&K=\mathbf{P}_{22} \mathbf{C}^\top\left(\mathbf{D} \mathbf{D}^\top\right)^{-1} \\
&\mathbf{L}=\left(R+\mathbf{Y}^\top\left(\mathbf{S}_{11}+\mathbf{S}_{22}\right) \mathbf{Y}\right)^{-1} \mathbf{B}^\top \mathbf{S}_{11} \\
&\bar{\mathbf{A}}^\top \mathbf{S}+\mathbf{S} \bar{\mathbf{A}}+\bar{\mathbf{Y}}^\top \mathbf{S} \bar{\mathbf{Y}}+\mathbf{V}=0 \\
&\bar{\mathbf{A}} \mathbf{P}+\mathbf{P} \bar{\mathbf{A}}^\top+\bar{\mathbf{Y}} \mathbf{P} \bar{\mathbf{Y}}^\top+\bar{\mathbf{G}} \bar{\mathbf{G}}^\top=0
\end{align}
$$


$\mathbf{A} = (a_{ij})$を $m \times n$行列，$\mathbf{B} = (b_{kl})$を $p \times q$行列とすると、それらのクロネッカー積 $\mathbf{A} \otimes \mathbf{B}$は

$$
\begin{equation}
\mathbf{A}\otimes \mathbf{B}={\begin{bmatrix}a_{11}\mathbf{B}&\cdots &a_{1n}\mathbf{B}\\\vdots &\ddots &\vdots \\a_{m1}\mathbf{B}&\cdots &a_{mn}\mathbf{B}\end{bmatrix}}
\end{equation}
$$

で与えられる $mp \times nq$区分行列である．

Roth's column lemma (vec-trick) 

$$
\begin{equation}
(\mathbf{B}^\top \otimes \mathbf{A})\text{vec}(\mathbf{X}) = \text{vec}(\mathbf{A}\mathbf{X}\mathbf{B})=\text{vec}(\mathbf{C})
\end{equation}
$$

によりこれを解くと，

$$
\begin{align}
\mathbf{S} &= -\text{vec}^{-1}\left(\left(\mathbf{I} \otimes \bar{\mathbf{A}}^\top + \bar{\mathbf{A}}^\top \otimes \mathbf{I} + \bar{\mathbf{Y}}^\top \otimes \bar{\mathbf{Y}}^\top\right)^{-1}\text{vec}(\mathbf{V})\right)\\
\mathbf{P} &= -\text{vec}^{-1}\left(\left(\mathbf{I} \otimes \bar{\mathbf{A}} + \bar{\mathbf{A}} \otimes \mathbf{I} + \bar{\mathbf{Y}} \otimes \bar{\mathbf{Y}}\right)^{-1}\text{vec}(\bar{\mathbf{G}}\bar{\mathbf{G}}^\top)\right)
\end{align}
$$

となる．ここで$\mathbf{I}=\mathbf{I}^\top$を用いた．

#### K, L, S, Pの計算
K, L, S, Pの計算は次のようにする．
1. LとKをランダムに初期化
1. SとPを計算
1. LとKを更新
1. 収束するまで2と3を繰り返す．

収束スピードはかなり速い．

### Target jump

target jumpする場合の最適制御 {cite:p}`Li2018-qt`. 状態にtarget位置も含むモデルであればtarget位置をずらせばよいが，ここでは自己位置をずらしtargetとの相対位置を変化させることでtarget jumpを実現する．

## げっ歯類の自由行動の現象論的モデル
本節は運動学習や最適制御から離れた話題であるが，「運動」という枠組みに入れることができるためにここで紹介を行う．ラットが自由行動下において箱の中を探索する際の軌跡をシミュレーションする {cite:p}`Raudies2012-gp`．これまでと異なり，現象論的に運動を生成する．場所細胞・格子細胞等自己位置と神経活動が相関する細胞のシミュレーションにおいて用いられる {cite:p}`George2024-rv`．

並進速度をレイリー分布，回転速度を正規分布に従うようにランダムサンプリングする．壁の接ベクトルとラットの距離を`dist_wall`, 壁の法線ベクトルとラットの頭方向の角度の差を`angle_wall`とする．なお，壁とはラットの自己速度ベクトルと壁全体との交点である．

- ラットの自己速度ベクトルと壁全体との交点を求める．
- 交点の接ベクトルと法線ベクトルを求める．
- 接ベクトルとの距離を`dist_wall`とする．
- 法線ベクトルと成す角を`angle_wall`とする．