- [第2章：発火率モデルと局所学習則](#第2章発火率モデルと局所学習則)
  - [神経細胞の生理](#神経細胞の生理)
  - [発火率モデルとHebb則　※](#発火率モデルとhebb則)
  - [線形回帰※](#線形回帰)
    - [最小二乗法によるパラメータの推定](#最小二乗法によるパラメータの推定)
      - [正規方程式を用いた推定](#正規方程式を用いた推定)
      - [勾配法を用いた推定](#勾配法を用いた推定)
  - [ロジスティック回帰とパーセプトロン※](#ロジスティック回帰とパーセプトロン)
  - [主成分分析](#主成分分析)
  - [独立成分分析](#独立成分分析)
  - [自己組織化マップと競合学習](#自己組織化マップと競合学習)
    - [自己組織化マップと視覚野の構造](#自己組織化マップと視覚野の構造)
    - [単純なデータセット](#単純なデータセット)

---

# 第2章：発火率モデルと局所学習則
## 神経細胞の生理
## 発火率モデルとHebb則　※


## 線形回帰※
線形回帰モデル (linear regression) では説明変数 (explanatory variable) $\mathbf{x}$ を線形変換し，目的変数 (objective variable) $y$を予測することを目的とする．説明変数$p$個の線形モデル 

$$
\begin{equation}
y=w_0+w_1x_1+\cdots+w_px_p+\varepsilon=w_0+\sum_{j=1}^p w_jx_j+\varepsilon
\end{equation}
$$

で説明することを考える．説明変数が単一 $(p=1)$ の場合を単回帰，複数 $(p>1)$ の場合を重回帰と呼ぶことがある．



次に，データセット $\mathcal{D}=\left\{\mathbf{x}^{(i)}, y^{(i)}\right\}_{i=1}^n$ を考える．ただし，$\mathbf{x}^{(i)}=\left[x_1^{(i)}, x_2^{(i)}, \ldots, x_p^{(i)}\right]^\top\in \mathbb{R}^p,\ y^{(i)}\in \mathbb{R}$とする．ここで添え字 $(i)$ が付いている場合は観測値を，無い場合はモデル内変数を表すことに注意しよう．
ここで，
$$
\mathbf{y}= \left[ \begin{array}{c} y^{(1)}\\ y^{(2)}\\ \vdots \\ y^{(n)} \end{array} \right] \in \mathbb{R}^n,\quad 
\mathbf{X}=\left[ \begin{array}{ccccc} 1 & x_{1}^{(1)}& x_{2}^{(1)} &\cdots & x_{p}^{(1)} \\ 1& x_{1}^{(2)}& x_{2}^{(2)}&\cdots & x_{p}^{(2)}\\ \vdots & \vdots& \vdots& \ddots & \vdots \\1 &x_{1}^{(n)} & x_{2}^{(n)} &\cdots & x_{p}^{(n)} \end{array} \right] \in \mathbb{R}^{n\times (p+1)}, \quad \mathbf{w}= \left[ \begin{array}{c} w_0\\ w_1\\ \vdots \\ w_p \end{array} \right] \in \mathbb{R}^{p+1}
$$

この場合，回帰モデルは $\mathbf{y}=\mathbf{X}\mathbf{w}+\mathbf{\varepsilon}$と書ける．ただし，$\mathbf{X}$は計画行列 (design matrix)，$\boldsymbol{\varepsilon}$は誤差項である．特に，$\mathbf{\varepsilon}$が平均0, 分散$\sigma^2$の独立な正規分布に従う場合，$\mathbf{y}\sim \mathcal{N}(\mathbf{X}\mathbf{w}, \sigma^2\mathbf{I})$と表せる．

### 最小二乗法によるパラメータの推定
最小二乗法 (ordinary least squares)により線形回帰のパラメータを推定する．$y$の予測値は$\mathbf{X} \mathbf{w}$なので，誤差 $\mathbf{\delta} \in \mathbb{R}^n$は
$\mathbf{\delta} = \mathbf{y}-\mathbf{X} \mathbf{w}$と表せる．ゆえに目的関数$L(\mathbf{w})$は 

$$
\begin{equation}
L(w)=\sum_{i=1}^n \delta_i^2 = \|\mathbf{\delta}\|^2=\mathbf{\delta}^\top \mathbf{\delta}
\end{equation}
$$

となり， $L(\mathbf{w})$を最小化する$\mathbf{w}$, つまり $\hat {\mathbf {w }}={\underset {\mathbf {w}}{\operatorname {arg min} }}\,L({\mathbf{w}})$
を求める．

#### 正規方程式を用いた推定
条件に基づいて目的関数$L(\mathbf{w})$を微分すると次のような方程式が得られる．

$$
\begin{equation}
\mathbf{X}^\top\mathbf{X}\mathbf{\hat w}=\mathbf{X}^\top\mathbf{y}
\end{equation}
$$

これを**正規方程式** (normal equation)と呼ぶ．この正規方程式より、係数の推定値は$\mathbf{\hat w}={(\mathbf{X}^\top\mathbf{X})}^{-1}X^\top\mathbf{y}$という式で得られる．なお，正規方程式自体は$\mathbf{y}=\mathbf{X}\mathbf{w}$の左から$\mathbf{X}^\top$をかける，と覚えると良い．

#### 勾配法を用いた推定
最小二乗法による回帰直線を勾配法で求めてみよう．$w$の更新式は$w \leftarrow w + \alpha\cdot \dfrac{1}{n} \delta \mathbf{X}$と書ける．ただし，$\alpha$は学習率である．

## ロジスティック回帰とパーセプトロン※


## 主成分分析

## 独立成分分析

## 自己組織化マップと競合学習

### 自己組織化マップと視覚野の構造
視覚野にはコラム構造が存在する．こうした構造は神経活動依存的な発生  (activity dependent development) により獲得される．本節では視覚野のコラム構造を生み出す数理モデルの中で，**自己組織化マップ (self-organizing map)** {cite:p}`Kohonen1982-mn`, {cite:p}`Kohonen2013-yt`を取り上げる．

自己組織化マップを視覚野の構造に適応したのは{cite:p}`Obermayer1990-gq` {cite:p}`N_V_Swindale1998-ri`などの研究である．視覚野マップの数理モデルとして自己組織化マップは受容野を考慮しないなどの簡略化がなされているが，単純な手法にして視覚野の構造に関する良い予測を与える．他の数理モデルとしては自己組織化マップと発想が類似している **Elastic net**  {cite:p}`Durbin1987-bp` {cite:p}`Durbin1990-xx` {cite:p}`Carreira-Perpinan2005-gy`　(ここでのElastic netは正則化手法としてのElastic net regularizationとは異なる)や受容野を明示的に設定した {cite:p}`Tanaka2004-vz`， {cite:p}`Ringach2007-oe`などのモデルがある．総説としては{cite:p}`Das2005-mq`，{cite:p}`Goodhill2007-va` ，数理モデル同士の関係については{cite:p}`2002-nm`が詳しい．

自己組織化マップでは「抹消から中枢への伝達過程で損失される情報量」，および「近い性質を持ったニューロン同士が結合するような配線長」の両者を最小化するような学習が行われる．包括性 (coverage) と連続性 (continuity) のトレードオフとも呼ばれる {cite:p}`Carreira-Perpinan2005-gy`　 (Elastic netは両者を明示的に計算し，線形結合で表されるエネルギー関数を最小化する．Elastic netは本書では取り扱わないが，MATLAB実装が公開されている
<https://faculty.ucmerced.edu/mcarreira-perpinan/research/EN.html>) ． 連続性と関連する事項として，近い性質を持つ細胞が脳内で近傍に存在するような発生/発達過程を**トポグラフィックマッピング (topographic mapping)** と呼ぶ．トポグラフィックマッピングの数理モデルの初期の研究としては{cite:p}`Von_der_Malsburg1973-bz` {cite:p}`Willshaw1976-zo` {cite:p}`Takeuchi1979-mi`などがある．

発生の数理モデルに関する総説 {cite:p}`Van_Ooyen2011-fz`, {cite:p}`Goodhill2018-ho`

### 単純なデータセット
SOMにおける$n$番目の入力を $\mathbf{v}(t)=\mathbf{v}_n\in \mathbb{R}^{D} (n=1, \ldots, N)$，$m$番目のニューロン$ (m=1, \ldots, M) $の重みベクトル (または活動ベクトル, 参照ベクトル) を$\mathbf{w}_m(t)\in \mathbb{R}^{D}$とする {cite:p}`Kohonen2013-yt`．また，各ニューロンの物理的な位置を$\mathbf{x}_m$とする．このとき，$\mathbf{v}(t)$に対して$\mathbf{w}_m(t)$を次のように更新する．

まず，$\mathbf{v}(t)$と$\mathbf{w}_m(t)$の間の距離が最も小さい (類似度が最も大きい) ニューロンを見つける．距離や類似度としてはユークリッド距離やコサイン類似度などが考えられる．

$$
\begin{align}
&[\text{ユークリッド距離}]: c = \underset{m}{\operatorname{argmin}}\left[\|\mathbf{v}(t)-\mathbf{w}_m(t)\|^2\right]\\
&[\text{コサイン類似度}]: c  = \underset{m}{\operatorname{argmax}}\left[\frac{\mathbf{w}_m(t)^\top\mathbf{v}(t)}{\|\mathbf{w}_m(t)\|\|\mathbf{v}(t)\|}\right]
\end{align}
$$

この，$c$番目のニューロンを**勝者ユニット(best matching unit; BMU)** と呼ぶ．コサイン類似度において，$\mathbf{w}_m(t)^\top\mathbf{v}(t)$は線形ニューロンモデルの出力となる．このため，コサイン距離を採用する方が生理学的に妥当でありSOMの初期の研究ではコサイン類似度が用いられている {cite:p}`Kohonen1982-mn`．しかし，コサイン類似度を用いる場合は$\mathbf{w}_m$および$\mathbf{v}$を正規化する必要がある．ユークリッド距離を用いると正規化なしでも学習できるため，SOMを応用する上ではユークリッド距離が採用される事が多い．ユークリッド距離を用いる場合，$\mathbf{w}_m$は重みベクトルではなくなるため，活動ベクトルや参照ベクトルと呼ばれる．ここでは結果の安定性を優先してユークリッド距離を用いることとする．

こうして得られた$c$を用いて$\mathbf{w}_m$を次のように更新する．

$$
\begin{equation}
\mathbf{w}_m(t+1)=\mathbf{w}_m(t)+h_{cm}(t)[\mathbf{v}(t)-\mathbf{w}_m(t)]
\end{equation}
$$

ここで$h_{cm}(t)$は近傍関数 (neighborhood function) と呼ばれ，$c$番目と$m$番目のニューロンの距離が近いほど大きな値を取る．ガウス関数を用いるのが一般的である．

$$
\begin{equation}
h_{cm}(t)=\alpha(t)\exp\left(-\frac{\|\mathbf{x}_c-\mathbf{x}_m\|^2}{2\sigma^2(t)}\right)
\end{equation}
$$

ここで$\mathbf{x}$はニューロンの位置を表すベクトルである．また，$\alpha(t), \sigma(t)$は単調に減少するように設定する．\footnote{Generative topographic map (GTM)を用いれば$\alpha(t), \sigma(t)$の縮小は必要ない．また，SOMとGTMの間を取ったモデルとしてS-mapがある．}