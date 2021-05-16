# ベイズ統計の基礎

この節では本章で用いるベイズ統計の基礎的概念の説明を行う (予定)。

```{note}
悪いこと言わないので[渡辺澄夫先生のHP](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/index-j.html)の講義録、特に[ベイズ統計入門](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/joho-gakushu6.html)を読もう。
```

### 期待値 (Expectation)

$$
\mathbb{E}_{x\sim p(x)}\left[f(x)\right]:=\int f(x)p(x)dx
$$

$x\sim p(x)$ が明示的な場合は $\mathbb{E}_{p(x)}\left[f(x)\right]$ や $\mathbb{E}\left[f(x)\right]$ と表す。

### 情報量 (Information)
出現頻度が低い事象は多くの情報量を持つ (Shannon, 1948)。

$$
\mathbb{I}(x):=\ln\left(\frac{1}{p(x)}\right)=-\ln p(x)
$$

$\mathbf{I}$は単位行列なので注意。

### 平均情報量 (エントロピー, entropy)

$$
\begin{align}
\mathbb{H}(x)&:=\mathbb{E}[-\ln p(x)]\\
\mathbb{H}(x\vert y)&:=\mathbb{E}[-\ln p(x\vert y)]
\end{align}
$$

### Kullback-Leibler 情報量
Kullback-Leibler (KL) divergence (Kullback and Leibler, 1951)

$$
\begin{align}
D_{\text{KL}}\left(p(x) \Vert\ q(x)\right)&:=\int p(x) \ln \frac{p(x)}{q(x)} dx\\
&=\int p(x) \ln p(x) dx-\int p(x) \ln q(x) dx\\
&=\mathbb{E}_{x\sim p(x)}[\ln p(x)]-\mathbb{E}_{x\sim p(x)}[\ln q(x)]\\
&=-\mathbb{H}(x)-\mathbb{E}_{x\sim p(x)}[\ln q(x)]
\end{align}
$$

### 相互情報量 (Mutual information)
