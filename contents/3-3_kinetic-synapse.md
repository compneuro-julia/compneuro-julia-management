## 3.3 動力学モデル
指数関数型シナプスとモデルの振る舞いはほぼ同一だが, 式の構成が少し異なるモデルとして**動力学モデル**(Kinetic model)がある ([Destexhe et al., 1994](https://link.springer.com/article/10.1007/BF00961734))。動力学モデルはHHモデルのゲート変数の式と類似した式で表される。このモデルではチャネルが開いた状態(Open)と閉じた状態(Close), および神経伝達物質(neurotransmitter)の放出状態(T)の2つの要素に関する状態がある。また, 閉$\to$開の反応速度を$\alpha$, 開$\to$閉の反応速度を$\beta$とする。このとき、これらを表す状態遷移の式は次のようになる。

$$
\begin{equation}
\text{Close}+\text{T}  \underset{\beta}{\overset{\alpha}{\rightleftharpoons}}\text{Open}    
\end{equation}
$$

ここで, シナプス動態を$r$とすると

$$
\begin{equation}
\frac{dr}{dt}=\alpha T (1-r) - \beta r
\end{equation}
$$

となる。ただし, Tはシナプス前細胞が発火したときにインパルス的に1だけ増加するとする。また, $\alpha, \beta$は速度なので, 時定数の逆数であることに注意しよう。 $\alpha=2000 \text{ms}^{-1}$, $\beta=200 \text{ms}^{-1}$とすると, シナプス動態は図のようになる。