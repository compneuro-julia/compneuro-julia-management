## 躍度最小モデル

目的関数として

**躍度最小モデル** (minimum-jerk model; {cite:p}`Flash1985-vj`) という．

### 変分法による解法
位置のベクトルを$\mathbf{x}(t) \in \mathbb{R}^n$とする．位置を1, 2, 3回微分したものをそれぞれ，速度，加速度，躍度（加加速度）と呼ぶ．躍度最小モデルでは，運動過程における躍度のノルムの二乗の総和を最小化することを目的とする．目的関数 $J$は

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