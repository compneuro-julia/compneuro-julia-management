# エネルギーベースモデル (Energy-based model)
本章では**エネルギーベースモデル (energy-based models; EBMs)** という枠組みに含まれるモデルを紹介する．エネルギーベースモデルではネットワークの状態をスカラー値に変換するエネルギー関数 (あるいはコスト関数) を定義し，推論時と学習時の双方においてエネルギーを最小化するようにネットワークの状態を更新する．{cite:p}`LeCun2006-dt`

入力 $\mathbf{x}\in \mathbb{R}^d$, エネルギー関数 $E_\theta: \mathbb{R}^d\to \mathbb{R}$を考える．

$$
\begin{align}
p_\theta(\mathbf{x})&=\frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}\\
Z_\theta &= \int \exp(-E_\theta(\mathbf{x})) d\mathbf{x}
\end{align}
$$

$Z_\theta$は分配関数．

## 参考文献
```{bibliography}
:filter: docname in docnames
```
