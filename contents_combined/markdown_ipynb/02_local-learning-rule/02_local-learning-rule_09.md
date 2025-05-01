## 自己組織化マップ
### 自己組織化マップと視覚野の構造
**自己組織化マップ**（Self-Organizing Map; SOM）は，Kohonenによって提案された教師なし学習アルゴリズムであり，高次元データを低次元（通常は2次元）の格子状マップに写像することにより，データのトポロジ的構造を保ちながら可視化する手法である．SOMは，**競合学習**（competitive learning）と呼ばれる学習規則に基づいており，入力パターンに最も近い出力ユニット（ニューロン）が「勝者」となり，その近傍のユニットとともに重みが更新される．競合学習はSOMに限らず，出力ニューロンが互いに競い合い，最も適合するものだけが活性化されるような学習機構を指す．SOMではこの競合に加えて，空間的な隣接性を重視した協調的な重み更新が行われる点が特徴的である．これにより，類似した入力はマップ上の近い位置に投影されるようになり，結果として**トポグラフィックマッピング** (topographic mapping) が実現される．

視覚野にはコラム構造が存在する．こうした構造は神経活動依存的な発生  (activity dependent development) により獲得される．本節では視覚野のコラム構造を生み出す数理モデルの中で，**自己組織化マップ** (self-organizing map) \citep{Kohonen1982-mn`, \citep{Kohonen2013-yt`を取り上げる．

自己組織化マップを視覚野の構造に適応したのは\citep{Obermayer1990-gq` \citep{N_V_Swindale1998-ri`などの研究である．視覚野マップの数理モデルとして自己組織化マップは受容野を考慮しないなどの簡略化がなされているが，単純な手法にして視覚野の構造に関する良い予測を与える．他の数理モデルとしては自己組織化マップと発想が類似している **Elastic net**  \citep{Durbin1987-bp` \citep{Durbin1990-xx` \citep{Carreira-Perpinan2005-gy`　(ここでのElastic netは正則化手法としてのElastic net regularizationとは異なる)や受容野を明示的に設定した \citep{Tanaka2004-vz`， \citep{Ringach2007-oe`などのモデルがある．総説としては\citep{Das2005-mq`，\citep{Goodhill2007-va` ，数理モデル同士の関係については\citep{2002-nm`が詳しい．

自己組織化マップでは「抹消から中枢への伝達過程で損失される情報量」，および「近い性質を持ったニューロン同士が結合するような配線長」の両者を最小化するような学習が行われる．包括性 (coverage) と連続性 (continuity) のトレードオフとも呼ばれる \citep{Carreira-Perpinan2005-gy` (Elastic netは両者を明示的に計算し，線形結合で表されるエネルギー関数を最小化する．Elastic netは本書では取り扱わないが，MATLAB実装が公開されている
<https://faculty.ucmerced.edu/mcarreira-perpinan/research/EN.html>) ． 連続性と関連する事項として，近い性質を持つ細胞が脳内で近傍に存在するような発生/発達過程を**トポグラフィックマッピング (topographic mapping)** と呼ぶ．トポグラフィックマッピングの数理モデルの初期の研究としては\citep{Von_der_Malsburg1973-bz` \citep{Willshaw1976-zo` \citep{Takeuchi1979-mi`などがある．

発生の数理モデルに関する総説 \citep{Van_Ooyen2011-fz`, \citep{Goodhill2018-ho`

### 単純なデータセット
SOMにおける $n$ 番目の入力を $\mathbf{v}(t)=\mathbf{v}_n\in \mathbb{R}^{D} (n=1, \ldots, N)$，$m$番目のニューロン $(m=1, \ldots, M)$ の重みベクトル（または活動ベクトル, 参照ベクトル）を $\mathbf{w}_m(t)\in \mathbb{R}^{D}$ とする \citep{Kohonen2013-yt`．また，各ニューロンの物理的な位置を $\mathbf{x}_m$ とする．このとき，$\mathbf{v}(t)$ に対して $\mathbf{w}_m(t)$ を次のように更新する．

まず，$\mathbf{v}(t)$ と $\mathbf{w}_m(t)$ の間の距離が最も小さい (類似度が最も大きい) ニューロンを見つける．距離や類似度としてはユークリッド距離やコサイン類似度などが考えられる．

$$
\begin{align}
&[\text{ユークリッド距離}]: c = \underset{m}{\operatorname{argmin}}\left[\|\mathbf{v}(t)-\mathbf{w}_m(t)\|^2\right]\\
&[\text{コサイン類似度}]: c  = \underset{m}{\operatorname{argmax}}\left[\frac{\mathbf{w}_m(t)^\top\mathbf{v}(t)}{\|\mathbf{w}_m(t)\|\|\mathbf{v}(t)\|}\right]
\end{align}
$$

この，$c$ 番目のニューロンを **勝者ユニット** (best matching unit; BMU) と呼ぶ．コサイン類似度において，$\mathbf{w}_m(t)^\top\mathbf{v}(t)$ は線形ニューロンモデルの出力となる．このため，コサイン距離を採用する方が生理学的に妥当でありSOMの初期の研究ではコサイン類似度が用いられている \citep{Kohonen1982-mn`．しかし，コサイン類似度を用いる場合は $\mathbf{w}_m$ および $\mathbf{v}$ を正規化する必要がある．ユークリッド距離を用いると正規化なしでも学習できるため，SOMを応用する上ではユークリッド距離が採用される事が多い．ユークリッド距離を用いる場合，$\mathbf{w}_m$ は重みベクトルではなくなるため，活動ベクトルや参照ベクトルと呼ばれる．ここでは結果の安定性を優先してユークリッド距離を用いることとする．

こうして得られた $c$ を用いて $\mathbf{w}_m$ を次のように更新する．

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