## 低速特徴分析
**Slow Feature Analysis (SFA)** とは, 複数の時系列データの中から低速に変化する成分 (slow feature) を抽出する教師なし学習のアルゴリズムである \citep{Wiskott2002-vb,Wiskott2011-uz}．潜在変数 $y$ の時間変化の2乗である $\left(\frac{dy}{\mathrm{d}t}\right)^2$を最小にするように教師なし学習を行う．初期視覚野の受容野 \citep{Berkes2005-i} や格子細胞・場所細胞などのモデルに応用がされている \citep{Franzius2007-sf}．

生理学的妥当性についてはいくつかの検討がされている．\citep{Sprekeler2007-qm} ではSTDP則によりSFAが実現できることを報告している．古典的な線形Recurrent neural networkでの実装も提案されている \citep{Lipshutz2020-uj}．

より具体的には，観測された高次元の入力信号 $\mathbf{x}(t) \in \mathbb{R}^n$ から，できるだけゆっくりと変化するスカラー出力 $y(t) = g(\mathbf{x}(t))$ を学習によって導出することが目的である．このとき，関数 $g(\cdot)$ は通常，入力に対して線形または非線形な写像である．

SFAの基本的な最適化問題は以下のように定式化される：

$$
\min_{g} \left\langle \left( \frac{d}{\mathrm{d}t} g(\mathbf{x}(t)) \right)^2 \right\rangle_t
$$

ただし，$\langle \cdot \rangle_t$ は時間平均を意味する．このままでは自明な定数解（全く変化しない出力）が得られるため，以下のような制約条件を課す：

1. **零平均**：$\langle y(t) \rangle_t = 0$
2. **単位分散**：$\langle y(t)^2 \rangle_t = 1$
3. **異なる特徴間の直交性**（複数のslow featureを抽出する場合）：$\langle y_i(t) y_j(t) \rangle_t = 0\quad (i \ne j)$

これらの制約により，情報量がありながらも変化の遅い特徴を抽出することが可能となる．実際のアルゴリズムでは，まず入力信号に対して一定の非線形写像（例えば多項式基底関数など）を適用した後，主成分分析（PCA）によって前処理を行い，その後時間的変化の最小化問題を一般化固有値問題として解くことでslow featuresを得る．

まずデータセットの生成を行う．\citep{Wiskott2002-vb}で用いられているトイデータを用いる．

Slow Feature Analysis (SFA) は，時系列データに含まれる情報のうち，時間的に最もゆっくりと変化する成分（slow features）を抽出するための教師なし学習アルゴリズムである．このアルゴリズムでは，観測された高次元の信号 $\mathbf{x}(t) \in \mathbb{R}^n$ に対して，線形または非線形な写像 $y(t) = g(\mathbf{x}(t))$ を学習し，その出力が時間的に滑らかになるように設計される．特に線形SFAの場合，写像 $g(\mathbf{x})$ は線形関数 $\mathbf{w}^\top \mathbf{x}(t)$ として表され，その時間微分の2乗平均 $\left\langle \left( \frac{d}{\mathrm{d}t} \mathbf{w}^\top \mathbf{x}(t) \right)^2 \right\rangle_t$ を最小化することが目的となる．

この最適化問題を解くためには，まず入力データ $\mathbf{x}(t)$ を前処理し，時間平均を引くことでゼロ平均化する．次に，共分散行列 $\mathbf{C}_x = \langle \tilde{\mathbf{x}}(t) \tilde{\mathbf{x}}(t)^\top \rangle_t$ を求め，これに対して固有値分解 $\mathbf{C}_x = \mathbf{E} \mathbf{D} \mathbf{E}^\top$ を適用することで主成分空間を構成し，白色化変換 $\mathbf{z}(t) = \mathbf{D}^{-1/2} \mathbf{E}^\top \tilde{\mathbf{x}}(t)$ を得る．この変換により，$\mathbf{z}(t)$ は単位分散かつ直交性を持つ特徴ベクトルとなる．

白色化されたデータに対して時間微分を近似的に計算し，$\dot{\mathbf{z}}(t) = \mathbf{z}(t+1) - \mathbf{z}(t)$ と定義することで，その共分散行列 $\mathbf{C}_{\dot{z}} = \langle \dot{\mathbf{z}}(t) \dot{\mathbf{z}}(t)^\top \rangle_t$ を構築することができる．SFAにおける主たる目的は，この微分共分散行列に関する最小固有値問題を解くことである．すなわち，$\mathbf{C}_{\dot{z}}$ に対する固有値分解または特異値分解（SVD）を行い，最小固有値に対応する固有ベクトル $\mathbf{u}_1$ を求めることで，最もゆっくりと変化する成分 $y(t) = \mathbf{u}_1^\top \mathbf{z}(t)$ を得ることができる．複数のslow featuresを得たい場合は，対応する小さい固有値順に固有ベクトルを選択することで可能となる．

最終的に，元のデータ空間におけるslow featuresを得るためには，逆変換を施して $\mathbf{W} = \mathbf{E} \mathbf{D}^{-1/2} \mathbf{U}$ とし，$\mathbf{U}$ は選択された固有ベクトルからなる行列である．この射影行列 $\mathbf{W}$ を用いることで，元の信号 $\tilde{\mathbf{x}}(t)$ からslow feature $y(t) = \mathbf{W}^\top \tilde{\mathbf{x}}(t)$ を得ることができる．このようにして，SFAはSVDを通じて効率的に解くことが可能であり，低速に変化する潜在表現を抽出するための強力な手法となっている．

**要約（箇条書きなし）**：

Bio-SFAは，入力信号 $\{\mathbf{x}_t\}$ を時系列的に与えられたとき，その中から「時間的に変化の遅い成分（slow features）」を線形写像 $\mathbf{y}_t = \mathbf{W} \mathbf{x}_t$ により抽出するための，生物学的にもっともらしいニューラルネットワーク学習則である．このアルゴリズムは，拡張された入力信号の和 $\mathbf{x}_t + \mathbf{x}_{t-1}$，出力信号の和 $\mathbf{y}_t + \mathbf{y}_{t-1}$ を用いてシナプス重みを更新するという点で特徴的である．具体的には，前回と今回の出力・入力の合計を掛け合わせた項（Hebbian項）と，入力に対応する樹状突起電流（feedforward入力）との積を減じる項（whitening項）からなる学習則に従い，出力が自己相関の高い，すなわち変化の遅い特徴になるように学習が行われる．また，出力ニューロン間には側方抑制的なシナプス結合（行列 $\mathbf{M}$）があり，それも出力の自己相関に基づいて更新されることで，抽出される slow features が互いに重複しないように整えられる．

**なぜ「差」ではなく「和」による学習が可能なのか**：

SFAの目的は，出力信号の「時間的変化の速さ（slowness）」すなわち離散時間微分の大きさ $\|\mathbf{y}_t - \mathbf{y}_{t-1}\|$ を最小化することにある．一見するとこれは差分に基づく学習を示唆するが，本論文ではこの最小化問題が，古典的な多次元尺度構成法（classical multidimensional scaling）を介して，入力の自己共分散行列 $\mathbf{C}_{\bar{x}\bar{x}}$（入力の和 $\mathbf{x}_t + \mathbf{x}_{t-1}$ を用いたもの）の主成分を最大化する問題と等価であることが示される．この等価性により，時間差 $(\mathbf{x}_t - \mathbf{x}_{t-1})$ ではなく時間和 $(\mathbf{x}_t + \mathbf{x}_{t-1})$ による学習則に置き換えることができる．特にこの形式はオンライン更新に適しており，またシナプス可塑性の局所性を保つことができる．

すなわち，「差分」ではなく「和」を用いる理由は、SFAの目的関数が等価に「時間的自己相関を最大化する」形式に書き換えられるためであり、これが時間和 $\mathbf{x}_t + \mathbf{x}_{t-1}$ を含む学習則を導くのである．また，この形式により，出力の内積 $\mathbf{y}_t^\top \mathbf{y}_{t-1}$ を最大化することができるが，これはまさに「変化が少ない」ことを意味するため，目的関数と整合的である．

では，以下に「SFAの目的関数が時間差分ではなく時間和に基づく形式に変換できる」ことの導出を行います．この導出は，Bio-SFA 論文中の Eq. (6) → Eq. (7) → Eq. (9) の流れを整理したものです．

---

### 1. SFA の元の目的関数（差分ベース）

SFAの目標は，入力の線形変換によって得られる出力 $\{\mathbf{y}_t\}$ に対して，その時間変化を最小化することです．このとき，

$$
\min_{\mathbf{V} \in \mathbb{R}^{m \times k}} \quad \frac{1}{T} \sum_{t=1}^T \|\mathbf{y}_t - \mathbf{y}_{t-1}\|^2 \quad \text{subject to } \frac{1}{T} \sum_{t=1}^T \mathbf{y}_t \mathbf{y}_t^\top = \mathbf{I}_k
$$

ここで，$\mathbf{y}_t = \mathbf{V}^\top \mathbf{x}_t$ と置くと，目的関数は

$$
\min_{\mathbf{V}} \operatorname{Tr} \left( \mathbf{V}^\top \mathbf{C}_{\dot{x} \dot{x}} \mathbf{V} \right) \quad \text{subject to } \mathbf{V}^\top \mathbf{C}_{x x} \mathbf{V} = \mathbf{I}_k
\tag{1}
$$

となります．ここで，

$$
\mathbf{C}_{\dot{x} \dot{x}} := \frac{1}{T} \sum_{t=1}^T (\mathbf{x}_t - \mathbf{x}_{t-1})(\mathbf{x}_t - \mathbf{x}_{t-1})^\top
$$

は時間差分の共分散行列です．

---

### 2. 時間差分と時間和の関係

次に，以下の恒等式を導入します：

$$
\begin{align*}
\mathbf{x}_t - \mathbf{x}_{t-1} &= \dot{\mathbf{x}}_t \\
\mathbf{x}_t + \mathbf{x}_{t-1} &= \bar{\mathbf{x}}_t
\end{align*}
$$

このとき，

$$
\mathbf{x}_t \mathbf{x}_t^\top + \mathbf{x}_{t-1} \mathbf{x}_{t-1}^\top = \frac{1}{2} \left( \bar{\mathbf{x}}_t \bar{\mathbf{x}}_t^\top + \dot{\mathbf{x}}_t \dot{\mathbf{x}}_t^\top \right)
$$

が成立することから，$\mathbf{C}_{\dot{x}\dot{x}}$ を使った最小化は，$\mathbf{C}_{\bar{x}\bar{x}}$ を使った最大化に等価です．実際，次のような等式が成立します（論文Appendix Aより）：

$$
\operatorname{Tr} \left( \mathbf{V}^\top \mathbf{C}_{\dot{x} \dot{x}} \mathbf{V} \right) = 4k - \operatorname{Tr} \left( \mathbf{V}^\top \mathbf{C}_{\bar{x} \bar{x}} \mathbf{V} \right)
$$

従って，目的関数 (1) は以下と等価です：

$$
\max_{\mathbf{V}} \operatorname{Tr} \left( \mathbf{V}^\top \mathbf{C}_{\bar{x} \bar{x}} \mathbf{V} \right) \quad \text{subject to } \mathbf{V}^\top \mathbf{C}_{x x} \mathbf{V} = \mathbf{I}_k
\tag{2}
$$

これは，「時間差が小さい特徴量を求める」問題が，「時間和の自己相関が大きい特徴量を求める」問題に変換できることを意味します．

---

### 3. 正規化入力による形式（時間和ベースの主成分分析）

$\mathbf{C}_{x x}$ が正則であると仮定すると，以下の変換を導入できます：

$$
\hat{\mathbf{x}}_t := \mathbf{C}_{x x}^{-1/2} \bar{\mathbf{x}}_t
$$

$$
\hat{\mathbf{V}} := \mathbf{C}_{x x}^{1/2} \mathbf{V}
$$

これにより，目的関数 (2) は単なる主成分分析の形式になります：

$$
\max_{\hat{\mathbf{V}}} \operatorname{Tr} \left( \hat{\mathbf{V}}^\top \mathbf{C}_{\hat{x} \hat{x}} \hat{\mathbf{V}} \right) \quad \text{subject to } \hat{\mathbf{V}}^\top \hat{\mathbf{V}} = \mathbf{I}_k
\tag{3}
$$

ここで $\mathbf{C}_{\hat{x} \hat{x}} = \frac{1}{T} \sum_t \hat{\mathbf{x}}_t \hat{\mathbf{x}}_t^\top$ は whitened 和入力の共分散です．

---

### 4. 距離最小化による最終目的関数

さらに，これを古典的多次元尺度構成（classical MDS）の形式に書き換えると，以下が得られます：

$$
\min_{\mathbf{Y}} \frac{1}{2T^2} \left\| \mathbf{Y}^\top \mathbf{Y} - \mathbf{X}^\top \mathbf{C}_{x x}^{-1} \mathbf{X} \right\|_{\mathrm{F}}^2
\tag{4}
$$

ここで $\mathbf{X} = [\bar{\mathbf{x}}_1, \ldots, \bar{\mathbf{x}}_T]$，$\mathbf{Y} = [\bar{\mathbf{y}}_1, \ldots, \bar{\mathbf{y}}_T]$，であり，時間和ベクトルの自己内積を一致させるように最適化する形式になっています．

---

### 結論

差分の自己共分散を最小化する（slownessの最大化）という本来の目的は，時間和ベクトルの内積を最大化することと等価であるため，入力・出力ともに「和」を使った局所学習則によって最適化が可能となります．

この変換によって，$\bar{\mathbf{x}}_t = \mathbf{x}_t + \mathbf{x}_{t-1}$，$\bar{\mathbf{y}}_t = \mathbf{y}_t + \mathbf{y}_{t-1} $ を用いた Hebbian 型学習則

$$
\Delta \mathbf{W} \propto \bar{\mathbf{y}}_t \bar{\mathbf{x}}_t^\top - \mathbf{a}_t \mathbf{x}_t^\top
$$

が導出されるわけです．

### ラテラル重み（側方抑制結合）$\mathbf{M}$ の更新則の導出

出力信号 $\mathbf{y}_t$ の時間変化を抑えつつ，出力の次元ごとに異なる特徴（＝非冗長な slow features）を学習させるため，SFAでは **出力の自己共分散** に対して白色化（直交化）制約を課します：

$$
\frac{1}{T} \sum_{t=1}^T \mathbf{y}_t \mathbf{y}_t^\top = \mathbf{I}_k
$$

この制約を Lagrange の未定乗数法で目的関数に組み込み，ラテラル重み $\mathbf{M}$ に相当する項を最適化すると，以下のような **勾配上昇（gradient ascent）ステップ** が得られます：

$$
\Delta \mathbf{M} = \frac{\eta}{\tau} \left( \bar{\mathbf{y}}_t \bar{\mathbf{y}}_t^\top - \mathbf{M} \right)
$$

ここで，

* $\eta$ は学習率
* $\tau$ は $\mathbf{M}$ の更新の時間スケール（$\mathbf{W}$ よりも遅く）
* $\bar{\mathbf{y}}_t = \mathbf{y}_t + \mathbf{y}_{t-1}$ は時間和

この更新則は、以下の観点から意味があります：

* $\bar{\mathbf{y}}_t \bar{\mathbf{y}}_t^\top$ により **現在と直前の出力間の相関行列** が測られ、
* それを $\mathbf{M}$ が徐々に近似していくことで、
* 出力ニューロン同士の活動が同じ方向を向かないように（直交的に）調整される

したがって、$\mathbf{M}$ は**抑制的ラテラル結合**を担い、異なる出力ユニットが「異なるslow features」に感度を持つようにする役割を果たします。

---

## 初心者向けの要約

**Slow Feature Analysis (SFA)** は、「ゆっくり変化する重要な特徴（slow features）」を時系列データから抽出する学習方法です。例えば、映像の中で変化しない背景や、空間的な位置情報などがこれに該当します。

この研究で提案された **Bio-SFA** は、SFAを脳の神経回路のように「オンラインで」かつ「局所的なシナプスルール」で実装できるようにしたものです。

### なぜ「差」ではなく「和」なのか？

普通、ゆっくり変化するとは「隣り合う時点の差が小さい」ことなので、差（$\mathbf{x}_t - \mathbf{x}_{t-1}$）を小さくするのがよさそうに思えます。でもこのままでは神経回路っぽく実装できません。

そこで「数学的トリック」で、差を小さくすることが、「和」（$\mathbf{x}_t + \mathbf{x}_{t-1}$）の内積を最大にすることと同じになると証明できます。

すると、**今と1つ前の活動の和**だけを使った計算で学習ができるようになり、シナプスの局所更新で実装できるのです。

### 学習の仕組み

* **入力から出力へ**の結合（重み $\mathbf{W}$）は、出力と入力の和（$\bar{\mathbf{y}}_t$, $\bar{\mathbf{x}}_t$）を使って Hebb 型に更新されます。
* **出力ニューロン同士のつながり**（重み $\mathbf{M}$）は、出力同士の和の相関を使って、重複のないslow featuresを学習します。

このようにして、脳のような仕組みで、環境中の変化の少ない重要な情報を学習できるのです。

https://github.com/flatironinstitute/bio-sfa