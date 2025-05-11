## ニューラルネットワークと誤差逆伝播法
### 順伝播
**ニューラルネットワーク** (neural network) \footnote{ニューラルネットワーク（neural network）」という用語は，実際の神経細胞から構成されるネットワークを指すのか，人工的に構築された数理モデルを指すのかが，字面だけでは判別しにくい．両者を区別する必要がある文脈では，人工的なモデルには「ニューラルネットワーク (neural network)」，「人工ニューラルネットワーク (artificial neural network; ANN)」，あるいは「人工神経回路網」といった用語が用いられる．一方，生体の神経回路を指す場合には，「ニューロナルネットワーク (neuronal network)」，「生物的ニューラルネットワーク (biological neural network; BNN)」，あるいは「神経回路網」といった用語を使用し，両者を区別することができる．}

順伝播 (forward propagation)
$f(\cdot)$を活性化関数とする．順伝播(feedforward propagation)は以下のようになる 
$(\ell=1,\ldots,L)$．ただし，活動を $\mathbf{z}_\ell \in \mathbb{R}^{n_\ell}$, 結合重みを $\mathbf{W}_\ell \in \mathbb{R}^{n_{\ell+1} \times n_{\ell}}$，定常項を $\mathbf{b}_\ell \in \mathbb{R}^{n_{\ell+1}}$ とする．

$$
\begin{align}
\textrm{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\textrm{隠れ層 : }&\mathbf{u}_\ell=\mathbf{W}_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{u}_\ell\right)\\
\textrm{出力層 : }&\mathbf{y}=\mathbf{z}_{L+1}
\end{align}
$$

2章で導入した活性化関数について，導関数と併記する．

代表的な活性化関数を紹介する．なお，`backward` における `y` は `forward` での出力に対応する．これは活性化関数を作用させる前の変数 ($x$であり，膜電位に対応する) を保持しておかなくても良いようにするためである．

### 活性化関数
シグモイド関数 (sigmoid function) あるいはロジスティック関数 (logistic function) の場合，

$$
\begin{align}
&\textrm{forward: } y = \frac{1}{1+e^{-x}}\\
&\textrm{backward: } \frac{dy}{dx} =\frac{e^{-x}}{(1+e^{-x})^2}= y\cdot (1-y)\\
\end{align}
$$

tanh関数の場合，

$$
\begin{align}
&\textrm{forward: } y = \frac{e^x-e^{-x}}{e^x+e^{-x}}\\
&\textrm{backward: } \frac{dy}{dx} =\frac{(e^x+e^{-x})(e^x+e^{-x})-(e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2}= 1-y^2\\
\end{align}
$$

ReLU関数 (rectified linear unit function, 正規化線形関数) あるいはランプ関数 (ramp function)の場合，

$$
\begin{align}
&\textrm{forward: } y = \max(x, 0)\\
&\textrm{backward: } \frac{dy}{dx} = \mathbf{1}_{x > 0}(x) = \mathbf{1}_{y > 0}(y)\\
\end{align}
$$

ただし，$\max(a, b)$は, $a, b$のうち，大きい値を返す関数である．また，$\mathbf{1}_{A}(x)$ は指示関数 (indicator function)であり，$x\in A$ ならば $\mathbf{1}_A(x)=1$ であり，それ以外の場合は $\mathbf{1}_A(x)=0$ となる関数である．

これらの活性化関数を構造体 `ActivationFunction`を用いて実装する．

以下では$z=f(a), g(z)=f'(a)$として膜電位を使わず，発火率情報のみを使うようにしている．このようにできない関数もあるが，今回はこのように書き下せる活性化関数のみを扱う．

### 重みの初期化
struct `MLP`を用意し，**重みの初期化** (weight initialization) を行う同名の関数`MLP`を用意する．重みの初期化に関しては，各層の出力および勾配の分散が一定となるような初期化をすることで学習が進行することが知られている．出力は活性化関数に依存するため，初期化についても活性化関数に応じて変更することが推奨され，sigmoid関数やtanh関数を用いる場合はXavierの初期化 \citep{Glorot2010-iu}，ReLU関数を用いる場合はHeの初期化 \citep{He2015-fs} が用いられる．入力ユニット数を $n_{\textrm{in}}$, 出力ユニット数を $n_{\textrm{out}}$ とすると，Xavierの初期化では重み $w$ の平均が0, 分散が $\frac{2}{n_{\textrm{in}}+n_{\textrm{out}}}$ となるように一様分布 $U\left(-\sqrt{\frac{6}{n_{\textrm{in}}+n_{\textrm{out}}}}, \sqrt{\frac{6}{n_{\textrm{in}}+n_{\textrm{out}}}}\right)$ や正規分布 $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{\textrm{in}}+n_{\textrm{out}}}}\right)$ 等から重みをサンプリングする．Heの初期化ではReLUを用いる場合，重み $w$ の平均が0, 分散が$\frac{2}{n_{\textrm{in}}}$ あるいは $\frac{2}{n_{\textrm{out}}}$ となるようにし，前者の分散を使用する場合は一様分布 $U\left(-\sqrt{\frac{6}{n_{\textrm{in}}}}, \sqrt{\frac{6}{n_{\textrm{in}}}}\right)$ や正規分布 $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{\textrm{in}}}}\right)$ 等から重みをサンプリングする．

## 誤差逆伝播法
ニューラルネットワークにおいて，効率よく各重みの勾配を推定することで貢献度割り当て問題を解決する方法が**誤差逆伝播法** (backpropagation) である．本節では入力層，隠れ層，出力層からなる多層ニューラルネットワークを実装し，誤差逆伝播法による勾配推定を用いて学習を行う．

本書では誤差逆伝播法を用いない学習法を実施することも考慮し，数式と対応するような実装を行う．そのため，Deep Learningライブラリ (PyTorch, Flux.jl等) のようにLayerを定義し，それを繋げてモデルを定義するということや計算グラフの構築は行わない．

### 逆伝播 (backward propagation)
ニューラルネットワークの学習 (learning) あるいは訓練 (training) とは，目的関数 (objective function) あるいは損失関数 (loss function) と呼ばれる評価指標を可能な限り小さく\footnote{場合によっては大きくすることも求められるが，そのような最大化問題であっても，目的関数の符号を逆にすれば最小化問題に帰着する．}するようなパラメータ集合 $\Theta = \{W_\ell, b_\ell\}_{\ell=1}^{L}$ を求める過程のことである．学習においてパラメータを最適化するアルゴリズムを**オプティマイザ** (optimizer) という．オプティマイザは多数提案されており，代表的なものを後ほど紹介する．まず，最も単純なオプティマイザである **勾配降下法** (gradient descent; GD) を紹介する．勾配降下法では全データを用いてパラメータ $\theta \in \Theta$ の更新量 $\Delta \theta$ を 

$$
\begin{equation}
\Delta \theta = -\eta \nabla_\theta \mathcal{L}_{\textrm{GD}} = -\eta \left(\frac{\partial \mathcal{L}_{\textrm{GD}}}{\partial \theta}\right)^\top = -\frac{\eta}{N} \sum_{i=1}^N \left(\frac{\partial \mathcal{L}^{(i)}}{\partial \theta}\right)^\top
\end{equation}
$$

として計算する（パラメータは$\theta\leftarrow \theta + \Delta \theta$により更新される）．ただし，$\mathcal{L}_{\textrm{GD}}:=\frac{1}{N}\sum_{i=1}^N \mathcal{L}^{(i)}$ であり，$\mathcal{L}^{(i)}$ は $i$ 番目のサンプルに対する目的関数であり，$N$ は全データのサンプル数を意味する．$\eta$ は学習率 (learning rate) である．勾配降下法のようにオプティマイザは一般的に損失のパラメータに対する勾配 $\nabla_\theta \mathcal{L}=\left(\frac{\partial \mathcal{L}}{\partial \theta}\right)^\top$ の計算を必要とする．**誤差逆伝播法**（backpropagation; BP）は、損失関数 $\mathcal{L}$ の各層のパラメータに対する勾配を効率的に計算するアルゴリズムであり、その導出には合成関数の微分則である**連鎖律**（chain rule）が用いられる。以下では分子レイアウト記法\footnote{第1章参照}を用いて、各層における勾配を順に求める．

まず、出力層 ($L+1$ 層) における損失 $\mathcal{L}$ の出力 $\mathbf{y} = \mathbf{z}_{L+1}$ に関する勾配は、
ここで $\boldsymbol{\delta}_L\in \mathbb{R}^{n_{L+1}}$ は**誤差信号**（error signal）と呼ばれ、以下のように再帰的に中間層へ伝播させる。

$$
\begin{equation}
\boldsymbol{\delta}_L:=\,\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_L}\right)^\top=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{u}_L}\right)^\top
\end{equation}
$$

となる。$\boldsymbol{\delta}_L$ の具体的な導出には損失関数の定義が必要となる．次に，$\ell$ 番目の中間層においては、1つ上の $\ell+1$ 番目の層における誤差信号 $\boldsymbol{\delta}_{\ell+1} \in \mathbb{R}^{n_{\ell+2}}$ を用いて、以下のように誤差信号 $\boldsymbol{\delta}_\ell \in \mathbb{R}^{n_{\ell+1}}$ を更新する：

$$
\begin{align}
\boldsymbol{\delta}_\ell:=&\,\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell}}\right)^\top=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{\ell+1}}\frac{\partial \mathbf{u}_{\ell+1}}{\partial \mathbf{z}_{\ell+1}}\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_{\ell}}\right)^\top\\
=&\,\mathrm{diag}\left(f_\ell^{\prime}\left(\mathbf{u}_{\ell}\right)\right){\mathbf{W}_{\ell+1}^\top} \boldsymbol{\delta}_{\ell+1}\\
=&\,f_\ell^{\prime}\left(\mathbf{u}_{\ell}\right)\odot \left({\mathbf{W}_{\ell+1}^\top} \boldsymbol{\delta}_{\ell+1}\right)
\end{align}
$$

ここで，$\mathrm{diag}(\cdot)$ は，ベクトルの各成分を対角要素に配置した対角行列を生成する演算子である．また，$\odot$ は同じ次元のベクトル同士の要素積（Hadamard積）を表す記号である．任意の $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ に対して，$\mathrm{diag}(\mathbf{a}) \mathbf{b} = \mathbf{a} \odot \mathbf{b}$ が成り立つことから，最後の式変形ではこの等式を用いて簡略化している．

$$
\begin{alignat}{4}
\nabla_{\mathbf{W}_{\ell}}\mathcal{L}&= \left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}\right)^\top&&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{W}_\ell}\right)^\top&&=\boldsymbol{\delta}_\ell \mathbf{z}_\ell^\top &&\left(\in \mathbb{R}^{n_{\ell+1}\times n_{\ell}}\right)\\
\nabla_{\mathbf{b}_{\ell}}\mathcal{L}&= \left(\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}\right)^\top&&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{u}_\ell} \frac{\partial \mathbf{u}_\ell}{\partial \mathbf{b}_\ell}\right)^\top&&=\boldsymbol{\delta}_\ell^\top &&\left(\in \mathbb{R}^{n_{\ell+1}}\right)
\end{alignat}
$$

が成り立つ．実装時に注意したいこととして，Juliaは基本が列ベクトルであるので，$\boldsymbol{\delta}_\ell$ も行ベクトルではなく列ベクトルとして保存および処理をする．さらにバッチ処理も考慮するので，行列を乗じる順番や転置の有無などが数式通りとはならない．

### 損失関数
回帰問題において，代表的に用いられるのが平均二乗誤差 (mean squared error) である．教師信号を $\mathbf{y}^*$ として，

$$
\begin{align}
\mathbf{y} &= \mathbf{z}_{L+1}\\
\mathcal{L}&:=\frac{1}{2}\left\|\mathbf{y}-\mathbf{y}^*\right\|^{2}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{y}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}}=\left(\mathbf{y}-\mathbf{y}^*\right)^\top\\
\delta_L&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{u}_L}\right)^\top=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{u}_L}\right)^\top=\left(\mathbf{y}-\mathbf{y}^*\right) \odot f_L^{\prime}\left(\mathbf{u}_L\right)\\
\end{align}
$$

2クラス分類で用いられるのがバイナリ交差エントロピー (binary cross entropy) である．

多クラス分類課題で用いられるのが，softmaxおよびcross entropy lossである．
softmax関数は $\mathbf{y} = \text{softmax}(\mathbf{z})$ とすると，各成分を以下のように定義される．

$$
\begin{equation}
y_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
\end{equation}
$$

### オプティマイザ
abstract typeとして`Optimizer`タイプを作成する．

勾配降下法は$N$が大きい場合，あるいは1つのサンプルのデータサイズが大きい場合は非効率であるので，ニューラルネットワークの学習においては，データの部分集合であるミニバッチ (mini-bacth) を用いた **確率的勾配降下法** (stochastic gradient descent; SGD) が用いられる．
**確率的勾配降下法(stochastic gradient descent; SGD)** を実装する．

次に**Adam** {cite:p}`Kingma2014-fm` を実装する．

Adamに関する考察．
https://www.sciencedirect.com/science/article/pii/S030326472300285X

重みのL2正則化 (Weight decay) を加える．正則化があることにより，実際のニューロンの活動を人工神経回路で再現できる研究も複数ある．バイアス項にはweight decayをしないため，optimizerの構造体の外からweight decayの値を与えることとする．

### サンプル課題
#### 例1. スパイラルデータセット (分類課題)
スパイラルデータセットは渦巻状をしており，$k$番目$(k=1, \ldots, K)$のクラスに属するサンプルは次の式で表される:

$$
\begin{align}
&\phi_k = \theta+\delta\theta+\frac{2\pi k}{K}\quad (\theta=[0, \pi])\ \\
&\mathbf{x}_k=[r\cos(\phi_k),\ r\sin(\phi_k)]\in \mathbb{R}^2\quad (r=[0, 1])
\end{align}
$$
ただし，$\theta/r=\pi$であり，$\delta\theta \sim \mathcal{N}(0, \sigma^2=0.2^2)$である．

#### 例2. 座標系の変換：Zipser-Andersenモデル
ニューラルネットワークを用いた神経回路網のモデル化の例としてZipser-Andersenモデル {cite:p}`Zipser1988-nc` を取り上げる．モデルの説明の前に，脳における**座標系** (coordinate system, reference frame) について説明する．座標系は身体や他の物体の位置を「定義」し，自己の運動制御や外界の理解をするために必要とされ，脳内では異なる原点を持つ複数の座標系が神経回路により並列して表現されている．座標系は大別して自己の位置を基準とした自己中心座標系 (egocentric coordinates) と外部の物体や環境を基準とした外界中心座標系 (allocentric coordinates) に分けられる．自己中心座標系としては網膜の中心窩を原点とした網膜中心座標系 (retinotopic coordinate)，頭部や身体を原点とした頭部/身体中心座標系(head centered/body centered coordinate) がある．外界中心座標系としては海馬の場所細胞 (place cells) などが代表的である．

Zipser-Andersenモデル {cite:p}`Zipser1988-nc` は頭頂葉の7a野のモデルであり，眼球位置の情報を用いて物体の位置の表現を網膜座標系から頭部中心座標に変換する3層ニューラルネットワークモデルである．隠れ層はPPC(Posterior parietal cortex)の細胞のモデルになっている．網膜座標系から頭部中心座標への変換を体験してみよう．例えば本書を読んでいるとして，頭を動かさずに眼球だけを上に動かしてみる (動かすとこの先の文章は読めなくなるが)．本の位置が網膜像では下側にずれたと思われる．しかし，頭部を動かしていない限り，頭部に対する相対的な位置は変化していない．

##### データセットの生成
物体位置の表現にはGaussian形式とmonotonic形式があるが，簡単のために，Gaussian形式を用いる．

入力は64(網膜座標系での位置)+2(眼球位置信号)=66とする．元のZipser-Andersenモデルにおいては眼球位置信号は活動が単調変化する32ユニット (=8ユニット×2(x, y方向)×2 (傾き正負)) によって符号化されているが，ここでは簡単のために眼球位置信号は$x, y$の2次元とする．

補足としてMonotonic formatによる位置のエンコーディングに触れる．monotonic形式を入力の眼球位置と出力の頭部中心座標で用いるという仮定には，視覚刺激を中心窩で捉えた際，得られる眼球位置信号を頭部中心座標での位置の教師信号として使用できるという利点がある．\citep{Andersen1983-zp} では Parietal visual neurons (PVNs)の活動を調べ，傾き正あるいは負．0度をピークとして減少あるいは上昇の4種類あることを示した．前者は一次関数（とReLU関数）で記述可能である．

視覚刺激は-40度から40度までの範囲であり，10度で離散化する．よって，網膜座標系での位置は$8\times 8$の行列で表現される．位置は2次元のGaussianで表現する．ただし，$1/e$ 幅（ピークから $1/e$ に減弱する幅）は15度である．$1/e$ の代わりに $1/2 $とすれば半値全幅(FWHM)となる．スポットサイズを $w$，ガウス関数を $G(x)$ とすると．$G(x+w/2)=G/e$ より，$\sigma=\frac{\sqrt{2}w}{4}$ と求まる．

#### 例3. MNIST
`MNIST` の代わりに`FashionMNIST` を用いることもできる．MNISTは易しい課題であるため，MNISTを訓練できるからと言って複雑な課題でも機能する保証はない．とは言え，基本的なデータセットであるため，

#### 例4. 自己符号化器
https://www.science.org/doi/10.1126/science.1127647

### 線形多層ニューラルネットワークの学習ダイナミクス

> A. M. Saxe, J. L. McClelland, S. Ganguli. "**A mathematical theory of semantic development in deep neural networks**". *PNAS.* (2019). ([arXiv](https://arxiv.org/abs/1810.10531)). ([PNAS](https://www.pnas.org/content/early/2019/05/16/1820226116))

#### モデルと学習
入力 $\mathbf{x}$ は「もの」の項目(例えばカナリア，犬，サーモン，樫など)，出力 $\mathbf{y}$はそれぞれの項目の性質・特性となっている．例えばカナリア(Canary)は成長し(Grow)，動き(Move)，空を飛べる(Fly)ので，Canaryという入力に対し，ネットワークが出力するのはGrow, Move, Flyとなる．モデルは3層の全結合線形ネットワークである．

$$
\hat{\mathbf{y}}=\mathbf{W}_2 \mathbf{W}_1\mathbf{x} 
$$

ただし非線形な活性化関数が無いことに注意しよう．このようなネットワークを線形ニューラルネットワーク (linear neural network)と呼ぶ．当然， $\mathbf{W}_s=\mathbf{W}_2 \mathbf{W}_1$として， 上のネットワークは

$$
\hat{\mathbf{y}}=\mathbf{W}_s\mathbf{x}
$$

とまとめることができる．このため，線形な活性化関数で深いニューラルネットワークを構築しても意味がなく，それゆえ非線形な活性化関数が必要となる．しかし，**勾配降下法で学習させると**3層と2層のネットワークの学習ダイナミクスはそれぞれ異なるものとなり，得られる解にも違いが生まれる．加えて，深い(3層の)ネットワークである場合のみ，幼児の発達における非線形な現象が説明できる．

3層ネットワークの学習(重みの更新)は誤差逆伝搬から導かれる次の2式により行う．

$$
\begin{aligned} 
\tau \frac{d\mathbf{W}_1}{dt} &=\mathbf{W}_2^\top \left(\mathbf{\Sigma}^{yx} - \mathbf{W}_2 \mathbf{W}_1 \mathbf{\Sigma}^{x}\right)\\
\tau \frac{d\mathbf{W}_2}{dt} &=\left(\mathbf{\Sigma}^{yx} - \mathbf{W}_2 \mathbf{W}_1 \mathbf{\Sigma}^{x}\right) \mathbf{W}_1^\top
\end{aligned}
$$

ただし，$\mathbf{\Sigma}^{x}$ は入力間の関係を表す行列，$\mathbf{\Sigma}^{yx}$ は入出力の関係を表す行列である．

#### 特異値分解(SVD)による学習ダイナミクスの解析
学習ダイナミクスは $\mathbf{\Sigma}^{yx}$ に対する特異値分解(singular value decomposition; SVD)を用いて説明できる．

$$
\begin{equation}
\mathbf{\Sigma}^{yx}=\mathbf{USV}^\top
\end{equation}
$$

行列$\mathbf{S}$ の対角成分の非ゼロ要素が特異値である．次に学習途中の時刻$(t)$における $\hat{\mathbf{\Sigma}}^{yx}(t)=\mathbf{W}_2 (t) \mathbf{W}_1(t) \mathbf{\Sigma}^{x}$ に対してSVDを実行し，特異値 $\mathbf{u}(t)=[a_{\alpha}(t)]$ を得る．この $a_{\alpha}(t)$ だが，3層のネットワークでは大きな特異値から先に学習されるのに対し，2層のネットワークでは全ての特異値が同時に学習される．このダイナミクスだが，**低ランク近似** (low-rank approximation)が生じていて，特異値の大きな要素から学習されていると捉えることができる．学習が進むとランクが大きくなっていく，ということである．低ランク近似の例として，SVDによる画像の圧縮と復元を見てみよう．カメラマンの画像に対し，低ランク近似を行い，ランクを上げていく．するとランクが上がるにつれて，画像が鮮明になる．

3層線形ネットワーク (deep)では大きな特異値から学習が始まっているのが分かる．また，それぞれの特異値の学習においてはシグモイド関数様の急速な学習段階が見られる．一方で2層線形ネットワーク (shallow)では全ての特異値の学習が初めから起こっていることがわかる．パラメータが少ないため，収束はこちらの方が速い．

このモデルの特徴として，知識の混同（例えば「芋虫には骨がある」）の仕組みを提供することがある．発達において，大きい特異値から先に学習されるため，「動く」，「成長する」などの動物の要素が先に獲得される．身の回りの動物のほとんどが「骨を持つ」ので，低ランク近似により，「芋虫にも骨がある」と錯覚してしまうのではないか，という仮説が立てられている．

#### 線形多層ニューラルネットワークにおける勾配降下法による低ランク解の獲得

> Jing, L., Zbontar, J. & LeCun, Y. **Implicit Rank-Minimizing Autoencoder**. *NeurIPS' 20*, 2020. <https://arxiv.org/abs/2010.00679>

([Arora et al., *NeurIPS' 19*. 2019](https://arxiv.org/abs/1905.13655))は深層線形ニューラルネットワークが低ランクの解を導出できることを理論的及び実験的に実証した．([Gunasekar et al., *NeurIPS' 18*. 2018](https://arxiv.org/abs/1806.00468))は，線形畳み込みニューラルネットワークにおいて勾配降下が正則化作用を持つことを示した．

証明は省略するが，([Arora et al., *NeurIPS' 19*. 2019](https://arxiv.org/abs/1905.13655))におけるTheorem 3.を紹介する．まず，$N$層の線形多層ニューラルネットワークを考え，$W_j \in \mathbb{R}^{d_j \times d_{j−1}}$を$j$層の重みとする．$t$を学習のタイムステップとし，$W(t) \in \mathbb{R}^{d \times d^\prime}$を重み行列を全て乗じた行列とする (ただし$d := d_N, d^\prime := d_0$)．つまり$W(t):=\prod_{j=1}^N W_j(t)$である．

ここで$W(t)$を特異値分解し，$W(t) = U(t)S(t)V^\top(t)$と表現する．$S(t)$は対角行列で，その要素を$\sigma_1(t), \ldots , \sigma_{\min\{d, d^\prime\}}(t),$とする．これが$W(t)$の特異値となる．さらに$U(t), V (t)$の列ベクトルをそれぞれ $\mathbf{u}_1(t), \ldots, \mathbf{u}_{\min\{d, d^\prime\}}(t)$, および $\mathbf{v}_1(t), \ldots, \mathbf{v}_{\min\{d,d^\prime \}}(t)$とする．このとき，特異値$ \sigma_r(t)\ (r=1, \ldots, \min\{d,d^\prime \})$の損失関数$\mathcal{L}(W(t))$に対する勾配降下法による変化は

$$
\frac{d \sigma_r(t)}{dt} = - N \cdot \left[\sigma_r(t)\right]^{1 - \frac{1}{N}} \cdot \left\langle \nabla \mathcal{L}(W(t)) , \mathbf{u}_r(t) \mathbf{v}_r^\top(t) \right\rangle
$$

と表される (Arora et al., 2019; Theorem 3)．(1)式で重要なのは$\left[\sigma_r(t)\right]^{1 - \frac{1}{N}}$の項である．これは$N\geq 2$のときに**特異値$\sigma_r(t)\ (\geq 0)$を小さくするような正則化作用が生じる**ことを意味している．一方で，隠れ層が1つのニューラルネットワーク ($N=1$)の場合 (1)式は

$$
\frac{d \sigma_r(t)}{dt} = - \left\langle \nabla \mathcal{L}(W(t)) , \mathbf{u}_r(t) \mathbf{v}_r^\top(t) \right\rangle
$$

となり，正則化作用は消失する．

このように線形多層ニューラルネットワークを勾配降下法で学習させると**陰的正則化(implicit regularization)** により低ランクの解が得られるということが複数の研究により明らかとなっている（線形多層ニューラルネットワークの陰的正則化に関して日本語で書かれた資料としては鈴木大慈先生の[深層学習の数理](https://www.slideshare.net/trinmu/ss-161240890)のスライドp.64, 65がある)．Jingらはこの性質を用い，**Autoencoderに線形層を複数追加**するという簡便な方法で低次元表現を学習する決定論的Autoencoder (**Implicit Rank-Minimizing Autoencoder; IRMAE)** を考案した．