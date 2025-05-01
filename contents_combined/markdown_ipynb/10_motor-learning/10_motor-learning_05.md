## 最適制御（文字修正）


Todorovの論文からはノイズ項を減らす．

離散時間で

$$
\begin{align}
\mathbf{x}_{t+1}&=\mathbf{A} \mathbf{x}_{t}+\mathbf{B} \mathbf{u}_{t}+\mathbf{w}_{t}\\
\mathbf{y}_{t+1}&=\mathbf{C} \mathbf{x}_{t}+\mathbf{v}_{t}
\end{align}
$$

ただし，$\mathbf{w}_{t}\sim \mathcal{N(\mathbf{0}, \Omega_w)}, \mathbf{v}_{t}\sim \mathcal{N(\mathbf{0}, \Omega_v)}$ である．このとき，
連続時間で

$$
\begin{align}
d\mathbf{x}(t)&=[\mathbf{A} \mathbf{x}(t)+\mathbf{B} \mathbf{u}(t)]dt+d\mathbf{w}(t)\\
d\mathbf{y}(t)&=\mathbf{C} \mathbf{x}(t)dt+d\mathbf{v}(t)
\end{align}
$$

ただし，$\mathbf{E}[d\mathbf{w}(t)]=\mathbf{0}, \mathbf{E}[d\mathbf{w}(t)d\mathbf{w}(t)^\top]=\Omega_w dt, \mathbf{E}[d\mathbf{v}(t)]=\mathbf{0}, \mathbf{E}[d\mathbf{v}(t)d\mathbf{v}(t)^\top]=\Omega_v dt$

と表記するのは正しいですか？


$$
\begin{align}
&\mathbf{x}_{t+1}=\mathbf{A} \mathbf{x}_{t}+\mathbf{B} \mathbf{u}_{t} +\boldsymbol{\xi}_{t}+\mathbf{Y}\mathbf{u}_{t}\gamma_t\\
&\mathbf{y}_{t}=\mathbf{C} \mathbf{x}_{t}+\omega_{t}\\
&\mathbf{x}_{t}^\top Q_{t} \mathbf{x}_{t}+\mathbf{u}_{t}^\top R \mathbf{u}_{t}\\
&\mathbf{u}_{t}=-\mathbf{L}_{t} \hat{\mathbf{x}}_{t}\\
&\hat{\mathbf{x}}_{t+1}=\mathbf{A} \hat{\mathbf{x}}_{t}+\mathbf{B} \mathbf{u}_{t}+\mathbf{K}_{t}\left(\mathbf{y}_{t}-\mathbf{C} \hat{\mathbf{x}}_{t}\right)+\boldsymbol{\eta}_{t} \\ 
\end{align}
$$

$$
\begin{align}
\mathbf{L}_{t}&=\left(R+\mathbf{B}^{\top} \mathbf{S}_{t+1} \mathbf{B}\right)^{-1} \mathbf{B}^{\top} \mathbf{S}_{t+1} \mathbf{A}\\
\mathbf{S}_{t}&=Q_{t}+\mathbf{A}^{\top} \mathbf{S}_{t+1}\left(\mathbf{A}-\mathbf{B} \mathbf{L}_{t}\right)\\
s_t &= \mathrm{tr}(\mathbf{S}_{t+1}\Omega^\xi) + s_{t+1}
\end{align}
$$

$\mathbf{S}_{T}=Q_{T}, s_T=0$

定数項が田中本と異なる．

$$
\begin{align}
\mathbf{K}_{t}&=\mathbf{A} \Sigma_{t} \mathbf{C}^{\top}\left(\mathbf{C} \Sigma_{t} \mathbf{C}^{\top}+\Omega^{\omega}\right)^{-1} \\ 
\Sigma_{t+1}&=\Omega^{\xi}+\left(\mathbf{A}-\mathbf{K}_{t} \mathbf{C}\right) \Sigma_{t} \mathbf{A}^{\top}
\end{align}
$$


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


動的計画法（Dynamic Programming, DP）で解くと、**最適フィードバック制御**が導かれる．

cost-to-go関数

$$
V_t(\mathbf{x}_t) = \min_{\{\mathbf{u}_\tau\}_{\tau=t}^{T-1}} \sum_{\tau=t}^{T-1} \left[ \mathbf{x}_\tau^\top Q \mathbf{x}_\tau + \mathbf{u}_\tau^\top R \mathbf{u}_\tau \right] + \mathbf{x}_T^\top Q_f \mathbf{x}_T
$$

Bellman方程式により，逐次的に以下のように解ける
$$
V_t(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \mathbf{x}_t^\top Q \mathbf{x}_t + \mathbf{u}_t^\top R \mathbf{u}_t + V_{t+1}(A \mathbf{x}_t + B \mathbf{u}_t) \right]
$$

加法ノイズしかない場合($C=D=0$)，制御問題は**線形二次ガウシアン** (linear-quadratic-Gaussian; LQG)制御と呼ばれる．

システムが線形で、コストが2次である場合を**線形二次制御** (linear-quadratic regulator; LQR) と呼ぶ．

$$
\begin{align}
\mathbf{u}_{t}&=-L_{t} \widehat{\mathbf{x}}_{t}\\
L_{t}&=\left(R+B^{\top} S_{t+1} B\right)^{-1} B^{\top} S_{t+1} A\\
S_{t}&=Q_{t}+A^{\top} S_{t+1}\left(A-B L_{t}\right)\\
s_t &= \mathrm{tr}(S_{t+1}\Omega^\xi) + s_{t+1}; s_T=0
\end{align}
$$

**リカッチ方程式**（Riccati recursion）によって逐次的に求まる．

$\boldsymbol{S}_{T}=Q$

#### 状態推定 (Kalman Filter)

$$
\begin{align}
\widehat{\mathbf{x}}_{t+1}&=A \widehat{\mathbf{x}}_{t}+B \mathbf{u}_{t}+K_{t}\left(\mathbf{y}_{t}-H \widehat{\mathbf{x}}_{t}\right)+\boldsymbol{\eta}_{t} \\ 
K_{t}&=A \Sigma_{t} H^{\top}\left(H \Sigma_{t} H^{\top}+\Omega^{\omega}\right)^{-1} \\ 
\Sigma_{t+1}&=\Omega^{\xi}+\left(A-K_{t} H\right) \Sigma_{t} A^{\top}
\end{align}
$$

**遠心性コピー** (efference copy)
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