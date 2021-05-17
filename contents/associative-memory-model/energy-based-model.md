# エネルギーベースモデル (Energy-based model)
入力 $\mathbf{x}\in \mathbb{R}^d$, エネルギー関数 $E_\theta: \mathbb{R}^d\to \mathbb{R}$を考える．

$$
\begin{align}
p_\theta(\mathbf{x})&=\frac{\exp(-E_\theta(\mathbf{x})}{Z_\theta}\\
Z_\theta &= \int \exp(-E_\theta(\mathbf{x})) d\mathbf{x}
\end{align}
$$

$Z_\theta$は分配関数．

