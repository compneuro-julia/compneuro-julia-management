## 無限時間最適フィードバック制御モデル
### モデルの構造
**無限時間最適フィードバック制御モデル** (infinite-horizon optimal feedback control model) {cite:p}`Qian2013-zy`

$$
\begin{align}
d \mathbf{x}&=(\mathbf{A} \mathbf{x}+\mathbf{B} \mathbf{u}) dt +\mathbf{Y} \mathbf{u} d \gamma+\mathbf{G} d \omega \\
d \mathbf{y}&=\mathbf{C} \mathbf{x} dt+\mathbf{D} d \xi\\
d \hat{\mathbf{x}}&=(\mathbf{A} \hat{\mathbf{x}}+\mathbf{B} \mathbf{u}) dt+\mathbf{K}(d\mathbf{y}-\mathbf{C} \hat{\mathbf{x}} dt)
\end{align}
$$

$$
\begin{align}
\mathbf{X}:=\begin{bmatrix}
\mathbf{x} \\
\tilde{\mathbf{x}}
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