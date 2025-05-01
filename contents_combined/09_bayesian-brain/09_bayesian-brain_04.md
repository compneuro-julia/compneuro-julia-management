## 確率的集団符号化
### 確率的集団符号化 (probabilistic population coding)

Distributional Population Coding or distributed distributional codes (DDCs)

ポアソン分布

$$
\begin{equation}
P(X=k)={\frac  {e^{-\lambda} \lambda^k}{k!}}
\end{equation}
$$

より，

$$
\begin{equation}
p(y \mid \mathbf{x}) \propto \prod_{i} \frac{e^{-f_{i}(y)} f_{i}(y)^{x_{i}}}{x_{i} !} p(y)
\end{equation}
$$