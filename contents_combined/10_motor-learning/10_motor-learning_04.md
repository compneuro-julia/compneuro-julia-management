## 状態空間モデル
観測過程も考慮した最適制御のモデルは状態空間モデル (state-space model) により記述される．この節では，後のモデルの基礎として触れる．

状態空間モデルとは、時間発展する力学系の状態の変化と出力を、ベクトルと行列を用いて表現する数学的枠組みである。一般に、あるシステムが時刻$t$において内部状態$\mathbf{x}(t) \in \mathbb{R}^n$、外部からの入力$\mathbf{u}(t) \in \mathbb{R}^m$、および観測可能な出力$\mathbf{y}(t) \in \mathbb{R}^p$をもつとき、その振る舞いは以下のような**一階の微分方程式**と**代数方程式**によって記述される：

$$
\begin{aligned}
\dot{\mathbf{x}}(t) &= f(\mathbf{x}(t), \mathbf{u}(t)), \\
\mathbf{y}(t) &= g(\mathbf{x}(t), \mathbf{u}(t)),
\end{aligned}
$$

ここで$f$および$g$は非線形関数である。
観測ノイズやシステムノイズを導入することで確率的な状態空間モデルを定式化でき、これに基づく代表的な手法としてカルマンフィルタなどがある．線形かつ連続な場合は，

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

ここで$k$は離散時刻を表す。連続時間のモデルを離散時間に変換する場合は，行列を補正する必要がある．このことについては第1章にて触れている．

$$
A \leftarrow A
$$

このように、状態空間モデルは、システムの内部状態と出力の関係を時間的に記述するための統一的かつ強力な理論的枠組みである。