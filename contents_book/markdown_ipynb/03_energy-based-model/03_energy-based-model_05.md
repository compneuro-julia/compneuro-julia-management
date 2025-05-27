## 予測符号化モデル
$u$ を $w$ に変更．


Annotated Bibliographyはもう一度確認する．

Pece, AEC (1992) Redundancy reduction of a Gabor representation: A possible computational role for feedback from primary visual cortex to lateral geniculate nucleus. In I Aleksander, & J Taylor, eds., Artificial Neural Networks, 2, 865–868. Amsterdam: Elsevier

Kawato, M, Hayakama, H, & Inui, T (1993) A forward-inverse optics model of reciprocal connections between visual cortical areas. Network: Computation in Neural Systems 4:415–422.


https://arxiv.org/abs/2011.07464
https://arxiv.org/abs/2112.10048

https://arxiv.org/abs/2410.19315
Predictive coding as variational inference

Srinivasan, M. V., Laughlin, S., & Dubs, A. (1982). Predictive coding: a fresh view of
inhibition in the retina. Proceedings of the Royal Society of London. Series B. Biological
Sciences, 216(1205), 427–459.

Dong, D. W., & Atick, J. J. (1995). Temporal decorrelation: a theory of lagged and
nonlagged responses in the lateral geniculate nucleus. Network: Computation in Neural
Systems, 6(2), 159–178.

A forward-inverse optics model of reciprocal connections between visual cortical areas
https://www.tandfonline.com/doi/abs/10.1088/0954-898X_4_4_001

https://pmc.ncbi.nlm.nih.gov/articles/PMC1569488/#bib45
https://pubmed.ncbi.nlm.nih.gov/15937014/


https://arxiv.org/abs/2212.00720

Kalman filter

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal
of Basic Engineering, 82(1), 35–45.

### 観測世界の階層的予測
**階層的予測符号化(hierarchical predictive coding; HPC)** は\citep{`Rao1999-zv`により導入された．構築するネットワークは入力層を含め，3層のネットワークとする．LGNへの入力として画像 $\mathbf{x} \in \mathbb{R}^{n_0}$を考える．画像 $\mathbf{x}$ の観測世界における隠れ変数，すなわち**潜在変数** (latent variable)を$\mathbf{r} \in \mathbb{R}^{n_1}$とし，ニューロン群によって発火率で表現されているとする (真の変数と $\mathbf{r}$は異なるので文字を分けるべきだが簡単のためにこう表す)．このとき，

$$
\begin{equation}
\mathbf{x} = f(\mathbf{U}\mathbf{r}) + \boldsymbol{\epsilon}
\end{equation}
$$

が成立しているとする．ただし，$f(\cdot)$は活性化関数 (activation function)，$\mathbf{U} \in \mathbb{R}^{n_0 \times n_1}$は重み行列である．$\boldsymbol{\epsilon} \in \mathbb{R}^{n_0}$ は $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ からサンプリングされるとする．潜在変数 $\mathbf{r}$はさらに高次 (higher-level)の潜在変数 $\mathbf{r}^h$により，次式で表現される．

$$
\begin{equation}
\mathbf{r} = \mathbf{r}^{td}+\boldsymbol{\epsilon}^{td}=f(\mathbf{U}^h \mathbf{r}^h)+\boldsymbol{\epsilon}^{td}
\end{equation}
$$

ただし，Top-downの予測信号を $\mathbf{r}^{td}:=f(\mathbf{U}^h \mathbf{r}^h)$ とした．また，$\mathbf{r}^{td} \in \mathbb{R}^{n_1}$, $\mathbf{r}^{h} \in \mathbb{R}^{n_2}$, $\mathbf{U}^h \in \mathbb{R}^{n_1 \times n_2}$ である．$\boldsymbol{\epsilon}^{td} \in \mathbb{R}^{n_1}$は$\mathcal{N}(\mathbf{0}$, $\sigma_{td}^2 \mathbf{I}$) からサンプリングされるとする．

話は飛ぶが，Predictive codingのネットワークの特徴は
- 階層的な構造
- 高次による低次の予測 (Feedback or Top-down信号)
- 低次から高次への誤差信号の伝搬 (Feedforward or Bottom-up 信号)

である．ここまでは高次表現による低次表現の予測，というFeedback信号について説明してきたが，この部分はSparse codingでも同じである．それではPredictive codingのもう一つの要となる，低次から高次への予測誤差の伝搬というFeedforward信号はどのように導かれるのだろうか．結論から言えば，これは復元誤差 (reconstruction error)の最小化を行う再帰的ネットワーク (recurrent network)を考慮することで自然に導かれる．

### 損失関数と学習則
#### 事前分布の設定
$\mathbf{r}$の事前分布$p(\mathbf{r})$はCauchy分布を用いる．$p(\mathbf{r})$の負の対数事前分布を$g(\mathbf{r}):=-\log p(\mathbf{r})$としておく．

$$
\begin{align}
p(\mathbf{r})&=\prod_i p(r_i)=\prod_i \exp\left[-\alpha \ln(1+r_i^2)\right]\\
g(\mathbf{r})&=-\ln p(\mathbf{r})=\alpha \sum_i \ln(1+r_i^2)\\
g'(\mathbf{r})&=\frac{\partial g(\mathbf{r})}{\partial \mathbf{r}}=\left[\frac{2\alpha r_i}{1+r_i^2}\right]_i
\end{align}
$$

次に重み行列$\mathbf{U}$の事前分布 $p(\mathbf{U})$はGaussian分布とする．$p(\mathbf{U})$の負の対数事前分布を$h(\mathbf{U}):=-\ln p(\mathbf{U})$とすると，次のように表される．

$$
\begin{align}
p(\mathbf{U})&=\exp(-\lambda\|\mathbf{U}\|^2_F)\\
h(\mathbf{U})&=-\ln p(\mathbf{U})=\lambda\|\mathbf{U}\|^2_F\\
h'(\mathbf{U})&=\frac{\partial h(\mathbf{U})}{\partial \mathbf{U}}=2\lambda \mathbf{U}
\end{align}
$$

ただし，$\|\cdot \| _ F^2$はフロベニウスノルムを意味する．

#### 損失関数の設定
Sparse codingと同様に考えることにより，損失関数 $E$を次のように定義する．

$$
\begin{align}
E=\underbrace{\frac{1}{\sigma^{2}}\|\mathbf{x}-f(\mathbf{U} \mathbf{r})\|^2+\frac{1}{\sigma_{t d}^{2}}\left\|\mathbf{r}-f(\mathbf{U}^h \mathbf{r}^h)\right\|^2}_{\text{reconstruction error}}+\underbrace{g(\mathbf{r})+g(\mathbf{r}^{h})+h(\mathbf{U})+h(\mathbf{U}^h)}_{\text{sparsity penalty}}
\end{align}
$$

潜在変数 $\mathbf{r}, \mathbf{r}^h$ と 重み行列 $\mathbf{U}, \mathbf{U}^h$ のそれぞれに事前分布を仮定しているため，これらについてのMAP推定を行うことに相当する．

#### 再帰ネットワークの更新則
簡単のために$\mathbf{z}:=\mathbf{U}\mathbf{r}, \mathbf{z}^h:=\mathbf{U}^h\mathbf{r}^h$とする．

$$
\begin{align}
\frac{d \mathbf{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} \mathbf{U}^{\top}\bigg[\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\odot\underbrace{(\mathbf{x}-f(\mathbf{z}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\mathbf{r})\Bigg)\\
\frac{d \mathbf{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(\mathbf{U}^h)^\top\bigg[\frac{\partial f(\mathbf{z}^h)}{\partial \mathbf{z}^h}\odot\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\mathbf{r}^h)\Bigg)
\end{align}
$$

ただし，$k_1$は更新率 (updating rate)である．または，発火率の時定数を$\tau:=1/k_1$として，$k_1$は発火率の時定数$\tau$の逆数であると捉えることもできる．ここで1番目の式において，中間表現 $\mathbf{r}$ のダイナミクスはbottom-up errorとtop-down errorで記述されている．このようにbottom-up errorが $\mathbf{r}$ への入力となることは自然に導出される．なお，top-down errorに関しては高次からの予測 (prediction)の項 $f(\mathbf{x}^h)$とleaky-integratorとしての項 $-\mathbf{r}$に分割することができる．また$\mathbf{U}^\top, (\mathbf{U}^h)^\top$は重み行列の転置となっており，bottom-upとtop-downの投射において対称な重み行列を用いることを意味している．$-g'(\mathbf{r})$は発火率を抑制してスパースにすることを目的とする項だが，無理やり解釈をすると自己再帰的な抑制と言える．