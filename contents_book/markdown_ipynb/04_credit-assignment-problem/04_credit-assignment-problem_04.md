## 予測符号化による活動と結合の共調整

本節では予測符号化による

### 予測符号化による訓練
PCには"Standard" Generative PC と "Reverse" Discriminative PCが存在する．
Millidge, B., Seth, A., & Buckley, C. L. (2021). Predictive Coding: a Theoretical and Experimental Review. In arXiv [cs.AI]. arXiv. http://arxiv.org/abs/2107.12979


ここでのPCは"Reverse" Discriminative PC

状態をdecayすることで，generativeにもdiscriminativeにもすることが可能．
A Predictive-Coding Network That Is Both Discriminative and Generative
https://direct.mit.edu/neco/article/32/10/1836/95621/A-Predictive-Coding-Network-That-Is-Both


入出力を固定 (clamp) する．電位固定法のようなものか？predictive codingと文字を合わせる．(Song et al., 2023)

$x_0=s_{in}, x_{L+1}=s_{target}$とする．状態$x_l(t=0)=\mathbf{0} (l=2, \ldots, L)$に初期化する．予測誤差 $\mathbf{\epsilon}_l$ を次式で計算する．

$$
\begin{equation}
\mathbf{\epsilon}_l(t)=\mathbf{z}_l(t)-\mathbf{w}_{l-1}f(\mathbf{z}_{l-1}(t))\quad(l=1, \ldots, L)
\end{equation}
$$

次に状態 $\mathbf{z}_l(t)\ (t=0, \ldots, \mathcal{T}-1)$ を次式で更新する．

$$
\begin{equation}
\mathbf{z}_l(t+1)=\mathbf{z}_l(t)+\gamma (-\mathbf{\epsilon}_l + f'(\mathbf{z}_l(t))) \circ (\mathbf{w}_l^\top \mathbf{\epsilon}_{l+1}(t))
\end{equation}
$$

収束後，重みを次式で更新する．$n$を一つのsampleの番号として，

$$
\begin{equation}
\mathbf{w}_l(n+1)=\mathbf{w}_l(n)+\eta \mathbf{\epsilon}_l(\mathcal{T}) f(\mathbf{z}_l(\mathcal{T}))^\top
\end{equation}
$$

として重みを更新する．
### 順伝播 (forward propagation)
$f(\cdot)$を活性化関数とする．順伝播(feedforward propagation)は以下のようになる．$(\ell=1,\ldots,L)$

$$
\begin{align}
\text{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{u}_\ell=W_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{u}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{z}_{L+1}
\end{align}
$$

#### 予測符号化による訓練
入出力を固定 (clamp) する．電位固定法のようなものか？predictive codingと文字を合わせる．(Rosebvbaum 2022, Song et al., 2023)

Rosenbaum, R. (2022). On the relationship between predictive coding and backpropagation. PloS One, 17(3), e0266102.

固定点解析によりbackpropと同等であることがわかる．
$\mathbf{z}_1=\mathbf{x}_{\textrm{in}}, \mathbf{z}_{L+1}=\mathbf{x}_{\textrm{target}}$とする．状態$\mathbf{z}_\ell(t=0)=\mathbf{0}\ (\ell=2, \ldots, L)$に初期化する．予測誤差 $\boldsymbol{\epsilon}_\ell(t)$ を次式で計算する．

$$
\begin{equation}
\boldsymbol{\epsilon}_{\ell}(t)=\mathbf{z}_{\ell+1}(t)-\mathbf{W}_{\ell}f(\mathbf{z}_{\ell}(t))\quad(\ell=1, \ldots, L-1)
\end{equation}
$$

$$
\boldsymbol{\epsilon}_{L} = \frac{\partial \mathcal{L} (\mathbf{z}_{L+1}, \mathbf{x}_{\textrm{target}})}{\partial \mathbf{z}_{L+1}}
$$

次に状態 $\mathbf{z}_\ell(t)\quad (\ell=2, \ldots, L;\  t=0, \ldots, \mathcal{T}-1)$ を次式で更新する．

$$
\begin{equation}
\mathbf{z}_\ell(t+1)=\mathbf{z}_\ell(t)+\gamma (-\boldsymbol{\epsilon}_{\ell-1} + f'(\mathbf{z}_\ell(t))) \circ (\mathbf{w}_\ell^\top \boldsymbol{\epsilon}_{\ell}(t))
\end{equation}
$$

収束後，重みを次式で更新する．$n$を一つのsampleの番号として，

$$
\begin{equation}
\mathbf{w}_l(n+1)=\mathbf{w}_l(n)+\eta \mathbf{\epsilon}_l(\mathcal{T}) f(\mathbf{z}_l(\mathcal{T}))^\top
\end{equation}
$$

として重みを更新する．


fixed prediction assumptionという (Millidge etal., 2022. Rosebvbaum 2022) 修正もある．

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}}\\
\boldsymbol{\delta}_L&:=\frac{\partial \mathcal{L}}{\partial \mathbf{u}_L}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{u}_L}\\
\boldsymbol{\delta}_\ell&:=\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell}}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell+1}}\frac{\partial \mathbf{u}_{\ell+1}}{\partial \mathbf{z}_{\ell+1}}\right)\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_{\ell}}\\
&={\mathbf{W}_{\ell+1}}^\top \boldsymbol{\delta}_{\ell+1} \odot f_\ell^{\prime}\left(\mathbf{u}_{\ell}\right)\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{W}_\ell}=\boldsymbol{\delta}_\ell \mathbf{z}_\ell^\top\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{b}_\ell}=\boldsymbol{\delta}_\ell
\end{align}
$$