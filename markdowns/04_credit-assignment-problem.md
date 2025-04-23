# 第4章：ニューラルネットワークと貢献度分配問題

\footnote{ニューラルネットワーク（neural network）」という用語は，実際の神経細胞から構成されるネットワークを指すのか，人工的に構築された数理モデルを指すのかが，字面だけでは判別しにくい．両者を区別する必要がある文脈では，人工的なモデルには「ニューラルネットワーク (neural network)」，「人工ニューラルネットワーク (artificial neural network; ANN)」，あるいは「人工神経回路網」といった用語が用いられる．一方，生体の神経回路を指す場合には，「ニューロナルネットワーク (neuronal network)」，「生物的ニューラルネットワーク (biological neural network; BNN)」，あるいは「神経回路網」といった用語を使用し，両者を区別することができる．}

## 貢献度分配問題

学習中に特定のシナプスがどのように選択され、異なる形態の可塑性が生じるのかは依然として不明であり、これはしばしば「クレジット割り当て問題」と呼ばれます。

生体における神経回路には、情報の入出力を中継する神経細胞が存在し、これらの細胞群は神経経路を構成している。

例えば、物体に手を伸ばして到達運動を行う際、一次運動野を経て錐体路を通り、末梢神経へと情報が伝達される（修正必要）。

この過程では、複数の神経細胞が協力して一連の運動を実現する。このような神経細胞群は、人工神経回路においては隠れ層（中間層）と呼ばれる部分に相当する。隠れ層の神経細胞は単に情報を中継しているのではなく、入力された情報を処理し、最終的な出力を改善するために重要な役割を果たしている。

このような神経回路において、特に注目すべきは、神経経路の途中に存在する細胞の活動をどのように調節すれば全体の機能を向上させることができるかという問題である。例えば、到達運動においては、物体に正確に触れること、すなわち到達誤差を最小化することが求められる。このような問題を解決するためには、各神経細胞の貢献度を適切に割り当てる必要があり、この問題を**貢献度分配問題** (credit assignment problem) と呼ぶ。

貢献度分配問題を解くためには、各神経細胞がどれだけ全体の運動結果に寄与しているかを評価し、適切に調整する必要がある。このような最適化の手法として、勾配法が広く用いられる。勾配法では、損失関数 \( L(\Theta) \) を最小化することを目的として、損失関数のパラメータに対する勾配を計算し、最適な更新を行う。具体的には、パラメータ \( \Theta = \{\theta_i\} \) （ここで \( i = 1, 2, ..., L \) は神経細胞のインデックス）に関して、損失関数の変化率である \( \delta\theta_i = \frac{dL}{d\theta_i} \) を求めることが、貢献度分配問題の解決に直結する。このようにして、個々の神経細胞の出力が全体の目標にどれだけ貢献しているかを反映させることができる。

人工神経回路における，この問題の解決策として誤差逆伝播法がある．本章では解決策の1つである誤差逆伝播法について実装し，誤差逆伝播法の何が生理的でないかを整理する．生理学的制約を付与したうえで誤差逆伝播法を近似する場合，どうすれば最適化されるのかを考える．そこから誤差逆伝播法のいくつかの近似法について整理する．

時系列問題に対しては経時的貢献度分配問題 (temporal credit assignment problem)という．

credit assignment problem 

The study of plasticity has always been about gradients

脳は学習を行う．学習を行った結果，ネットワークはよりよい状態となる．ネットワークのよさを表現するのが目的関数である．
機械学習では主に目的関数が明示的 (explicit) に与えられ，明示的に目的関数の値が改善されるように勾配降下法などでパラメータが更新される．
それでは脳には目的関数があり，その勾配を計算しているのだろうか．
これに対して否定的であったとしても，ネットワークがよい状態になるということは，勾配に部分的に従ってパラメータ更新がなされていることとなる．
つまり，脳は陰的 (implicit) に勾配降下を行っていると言える．

https://www.pnas.org/doi/10.1073/pnas.2111821118

local lossの発想

https://arxiv.org/abs/1702.07800

Stork, D. G. (1989). Is backpropagation biologically plausible. In International joint
conference on neural networks (Vol. 2, pp. 241–246).


## ニューラルネットワークと誤差逆伝播法
ニューラルネットワークにおいて，効率よく各重みの勾配を推定することで貢献度割り当て問題を解決する方法が**誤差逆伝播法** (backpropagation) である．本節では入力層，隠れ層，出力層からなる多層ニューラルネットワークを実装し，誤差逆伝播法による勾配推定を用いて学習を行う．

本書では誤差逆伝播法を用いない学習法を実施することも考慮し，数式と対応するような実装を行う．そのため，Deep Learningライブラリ (PyTorch, Flux.jl等) のようにLayerを定義し，それを繋げてモデルを定義するということや計算グラフの構築は行わない．

代表的な活性化関数を紹介する．なお，`backward` における `y` は `forward` での出力に対応する．これは活性化関数を作用させる前の変数 ($x$であり，膜電位に対応する) を保持しておかなくても良いようにするためである．

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

ただし，$\max(a, b)$は, $a, b$のうち，大きい値を返す関数である．また，$\mathbf{1}_{A}(x)$ は指示関数 (indicator function)であり，$x\in A$ ならば $\mathbf{1}_A(x)=1$ であり，それ以外の場合は $\mathbf{1}_A(x)=0$ となる関数である．ReLU関数は $x=0$ で折れ曲がるが，その他では線形であるため，**区分線形関数** (piecewise linear function) の一種であると言える．

これらの活性化関数を構造体 `ActivationFunction`を用いて実装する．

struct `MLP`を用意し，**重みの初期化** (weight initialization) を行う同名の関数`MLP`を用意する．重みの初期化に関しては，各層の出力および勾配の分散が一定となるような初期化をすることで学習が進行することが知られている．出力は活性化関数に依存するため，初期化についても活性化関数に応じて変更することが推奨され，sigmoid関数やtanh関数を用いる場合はXavierの初期化 \citep{Glorot2010-iu}，ReLU関数を用いる場合はHeの初期化 \citep{He2015-fs} が用いられる．入力ユニット数を $n_{\textrm{in}}$, 出力ユニット数を $n_{\textrm{out}}$ とすると，Xavierの初期化では重み $w$ の平均が0, 分散が $\frac{2}{n_{\textrm{in}}+n_{\textrm{out}}}$ となるように一様分布 $U\left(-\sqrt{\frac{6}{n_{\textrm{in}}+n_{\textrm{out}}}}, \sqrt{\frac{6}{n_{\textrm{in}}+n_{\textrm{out}}}}\right)$ や正規分布 $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{\textrm{in}}+n_{\textrm{out}}}}\right)$ 等から重みをサンプリングする．Heの初期化ではReLUを用いる場合，重み $w$ の平均が0, 分散が$\frac{2}{n_{\textrm{in}}}$ あるいは $\frac{2}{n_{\textrm{out}}}$ となるようにし，前者の分散を使用する場合は一様分布 $U\left(-\sqrt{\frac{6}{n_{\textrm{in}}}}, \sqrt{\frac{6}{n_{\textrm{in}}}}\right)$ や正規分布 $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{\textrm{in}}}}\right)$ 等から重みをサンプリングする．

### 順伝播 (forward propagation)
$f(\cdot)$を活性化関数とする．順伝播(feedforward propagation)は以下のようになる．$(\ell=1,\ldots,L)$

$$
\begin{align}
\text{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{a}_\ell=\mathbf{W}_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{a}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{z}_{L+1}
\end{align}
$$

### 逆伝播 (backward propagation)
ニューラルネットワークの学習 (learning) あるいは訓練 (training) とは，目的関数 (objective function) あるいは損失関数 (loss function) と呼ばれる評価指標を可能な限り小さく (場合によっては大きく) するようなパラメータ集合 $\Theta = \{W_\ell, b_\ell\}_{\ell=1}^{L}$ を求める過程のことである．学習においてパラメータを最適化するアルゴリズムを**オプティマイザ** (optimizer) という．オプティマイザは多数提案されており，代表的なものを後ほど紹介する．まず，最も単純なオプティマイザである **勾配降下法** (gradient descent; GD) を紹介する．勾配降下法では全データを用いてパラメータ $\theta \in \Theta$ の更新量 $\Delta \theta$ を 

$$
\Delta \theta = -\eta \frac{\partial \mathcal{L}_{\textrm{GD}}}{\partial \theta} = -\frac{\eta}{N} \sum_{i=1}^N \frac{\partial \mathcal{L}^{(i)}}{\partial \theta}
$$

として計算する (パラメータは$\theta\leftarrow \theta + \Delta \theta$により更新される)．ただし，$\mathcal{L}_{\textrm{GD}}:=\frac{1}{N}\sum_{i=1}^N \mathcal{L}^{(i)}$ であり，$\mathcal{L}^{(i)}$は$i$ 番目のサンプルに対する目的関数であり，$N$ は全データのサンプル数を意味する．$\eta$ は学習率 (learning rate) である．オプティマイザは一般的に勾配 $\dfrac{\partial \mathcal{L}}{\partial \theta}$ の計算を必要とする．この計算を効率よく行う手法が**誤差逆伝播法** (backpropagation) である．誤差逆伝播法は連鎖律 (chain rule; 合成関数の微分の関係式) を用いて導くことができる．$\mathbf{a}_\ell=W_\ell \mathbf{z}_\ell +\mathbf{b}_\ell$ および $\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{a}_\ell\right)$ であることを踏まえると，

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}}\\
\delta_L&:=\frac{\partial \mathcal{L}}{\partial \mathbf{a}_L}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{a}_L}\\
\mathbf{\delta}_\ell&:=\frac{\partial \mathcal{L}}{\partial \mathbf{a}_{\ell}}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{a}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{a}_{\ell+1}}\frac{\partial \mathbf{a}_{\ell+1}}{\partial \mathbf{z}_{\ell+1}}\right)\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{a}_{\ell}}\\
&={\mathbf{W}_{\ell+1}}^\top \delta_{\ell+1} \odot f_\ell^{\prime}\left(\mathbf{a}_{\ell}\right)\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{W}_\ell}=\delta_\ell \mathbf{z}_\ell^\top\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{b}_\ell}=\delta_\ell
\end{align}
$$

が成り立つ．バッチ処理を考慮すると，行列を乗ずる順番が変わる．以下では$z=f(a), g(z)=f'(a)$として膜電位を使わず，発火率情報のみを使うようにしている．このようにできない関数もあるが，今回はこのように書き下せる活性化関数のみを扱う．

### 損失関数
回帰問題において，代表的に用いられるのが平均二乗誤差 (mean squared error) である．

$$
\begin{align}
\hat{\mathbf{y}} &= \mathbf{z}_{L+1}\\
\mathcal{L}&:=\frac{1}{2}\left\|\hat{\mathbf{y}}-\mathbf{y}\right\|^{2}\\
\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}}=\hat{\mathbf{y}}-\mathbf{y}\\
\delta_L&=\frac{\partial \mathcal{L}}{\partial \mathbf{a}_L}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{a}_L}=\left(\hat{\mathbf{y}}-\mathbf{y}\right) \odot f_L^{\prime}\left(\mathbf{a}_L\right)\\
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

行列$\mathbf{S}$ の対角成分の非ゼロ要素が特異値である．次に学習途中の時刻$(t)$における $\hat{\mathbf{\Sigma}}^{yx}(t)=\mathbf{W}_2 (t) \mathbf{W}_1(t) \mathbf{\Sigma}^{x}$ に対してSVDを実行し，特異値 $\mathbf{A}(t)=[a_{\alpha}(t)]$ を得る．この $a_{\alpha}(t)$ だが，3層のネットワークでは大きな特異値から先に学習されるのに対し，2層のネットワークでは全ての特異値が同時に学習される．このダイナミクスだが，**低ランク近似** (low-rank approximation)が生じていて，特異値の大きな要素から学習されていると捉えることができる．学習が進むとランクが大きくなっていく，ということである．低ランク近似の例として，SVDによる画像の圧縮と復元を見てみよう．カメラマンの画像に対し，低ランク近似を行い，ランクを上げていく．するとランクが上がるにつれて，画像が鮮明になる．

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

## 非対称な逆向き投射による誤差伝播
Feedback alignment, DFA, CP

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
\text{隠れ層 : }&\mathbf{a}_\ell=W_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{a}_\ell\right)\\
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
\delta_L&:=\frac{\partial \mathcal{L}}{\partial \mathbf{a}_L}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{L+1}} \frac{\partial \mathbf{z}_{L+1}}{\partial \mathbf{a}_L}\\
\mathbf{\delta}_\ell&:=\frac{\partial \mathcal{L}}{\partial \mathbf{a}_{\ell}}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\ell+1}} \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{a}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{a}_{\ell+1}}\frac{\partial \mathbf{a}_{\ell+1}}{\partial \mathbf{z}_{\ell+1}}\right)\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{a}_{\ell}}\\
&={\mathbf{W}_{\ell+1}}^\top \delta_{\ell+1} \odot f_\ell^{\prime}\left(\mathbf{a}_{\ell}\right)\\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{W}_\ell}=\delta_\ell \mathbf{z}_\ell^\top\\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{b}_\ell}=\delta_\ell
\end{align}
$$

## 摂動を用いた学習則
本節では摂動法 (permutation) による勾配推定について説明する．摂動法に含まれる手法は複数あるが，総じて次のような手法を指す．まず，あるモデル（ネットワーク）を用意し，その目的関数を $\mathcal{L}$ とする．次にモデルのパラメータや活動にランダムな微小変化（摂動）$\mathbf{v}$ を加え，摂動を受ける前後の目的関数の変化量 $\delta \mathcal{L}$ を取得する．この $\delta \mathcal{L}$ や $\mathbf{v}$ およびモデルの活動等を用いてパラメータを更新するのが摂動法である．

### ノード摂動法と重み摂動法
代表的なニューラルネットワークの摂動法は**ノード摂動法** (Node perturbation; NP) と**重み摂動法** (weight perturbation; WP) である．ノード摂動法は各ノード（ニューロン）の活動に摂動を加える手法であり，重み摂動法は各パラメータ（シナプス結合等）に摂動を加える手法である．両者は統一的に解釈することが可能である．

まず，以下のように順伝播を行う $L$ 層のニューラルネットワークを定義する $(\ell=1,\ldots,L)$. $\mathbf{z}_{\ell}\in \mathbb{R}^{n_\ell}$とすると

$$
\begin{align}
\text{入力層 : }&\mathbf{z}_1=\mathbf{x}\\
\text{隠れ層 : }&\mathbf{a}_\ell=\mathbf{W}_\ell \mathbf{z}_\ell +\mathbf{b}_\ell\\
&\mathbf{z}_{\ell+1}=f_\ell\left(\mathbf{a}_\ell\right)\\
\text{出力層 : }&\hat{\mathbf{y}}=\mathbf{z}_{L+1}
\end{align}
$$

ただし，$\mathbf{W}_\ell \in \mathbb{R}^{n_{\ell+1}\times n_{\ell}}, \mathbf{b}_\ell \in \mathbb{R}^{n_{\ell+1}}$ である．
ここでは単純なMLPを扱うが，RNNでも可能である．損失は $\mathcal{L}(\mathbf{z}_{L+1}; \mathbf{x})$ とする．それぞれの手法において，以下のようにネットワークを摂動する．

$$
\begin{align}
\text{重み摂動法:}\quad &\tilde{\mathbf{z}}_{\ell+1}=f_\ell\left((\mathbf{W}_\ell+\sigma \mathbf{V}_\ell) \tilde{\mathbf{z}}_\ell +\mathbf{b}_\ell +\sigma \mathbf{v}_\ell\right)\\
\text{ノード摂動法:}\quad &\tilde{\mathbf{z}}_{\ell+1}=f_\ell\left(\mathbf{W}_\ell \tilde{\mathbf{z}}_\ell +\mathbf{b}_\ell+\sigma \mathbf{v}_\ell \right)
\end{align}
$$

ただし，$\mathbf{V}_\ell \in \mathbb{R}^{n_{\ell+1}\times n_{\ell}}, \mathbf{v}_\ell \in \mathbb{R}^{n_{\ell+1}}$ であり，各要素は $\mathcal{N}(0, 1)$ より独立にサンプリングされる \footnote{摂動は正規分布以外の分布，例えば $\{-1, 1\}$ (1か-1かを等確率で取る分布) からサンプリングすることも可能である．}．目的関数の変化量を

$$
\begin{equation}
\delta \mathcal{L}=\mathcal{L}(\tilde{\mathbf{z}}_{L+1}; \mathbf{x})-\mathcal{L}(\mathbf{z}_{L+1}; \mathbf{x})
\end{equation}
$$

とする．SGDでパラメータを行う場合，

$$
\begin{align}
\text{重み摂動法:}\quad &\Delta \mathbf{W}_\ell^{\mathrm{WP}}=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{V}_\ell, &\Delta \mathbf{b}_\ell^{\mathrm{WP}}=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell\\
\text{ノード摂動法:}\quad &\Delta \mathbf{W}_\ell^{\mathrm{NP}}=- \eta  \frac{\delta \mathcal{L}}{\sigma} \mathbf{v}_\ell \mathbf{z}_{\ell}^\top, &\Delta \mathbf{b}_\ell^{\mathrm{NP}} =- \eta  \frac{\delta \mathcal{L}}{\sigma} \mathbf{v}_\ell
\end{align}
$$

でパラメータを更新する．

#### 不偏推定量であることの証明
各手法の更新則が勾配の不偏推定量 (unbiased estimator) であることを示す．まず方向微分 (directional derivative) を導入する．関数 $f$ について点 $\mathbf{u}$ における方向 $\mathbf{v}$ の方向微分は

$$
\begin{equation}
\nabla_\mathbf{v}f(\mathbf{u}):= \lim_{h\to 0} \frac{f(\mathbf{u}+h\mathbf{v}) - f(\mathbf{u})}{h}
\end{equation}
$$

として定義される．また $f$ が点 $\mathbf{u}$ において微分可能なら

$$
\begin{equation}
\nabla_\mathbf{v}f(\mathbf{u})=\nabla f(\mathbf{u})\cdot \mathbf{v}\left(=\frac{\partial f(\mathbf{u})}{\partial \mathbf{u}}\cdot \mathbf{v}\right)
\end{equation}
$$

が成り立つ．ここで，$\nabla f(\mathbf{u})\cdot \mathbf{v}$ を Jacobian-vector product (JVP) と呼び，$f(\mathbf{u})\in \mathbb{R}$ の場合，$\nabla f(\mathbf{u})\cdot \mathbf{v}\in \mathbb{R}$ となる．このJVPを有限差分 (finite difference) を用いて近似計算すると\footnote{JVPは順方向自動微分 (Forward-mode automatic differentiation) により計算でき，有限差分法よりも数値的に安定する (順方向自動微分はPythonライブラリのJAX等に実装されている)．Forward Gradientは順方向自動微分を採用して重み摂動法をより安定させた手法である．}，

$$
\begin{equation}
\nabla f(\mathbf{u})\cdot \mathbf{v} \approx \frac{f(\mathbf{u}+\epsilon \mathbf{v}) - f(\mathbf{u})}{\epsilon}
\end{equation}
$$

となる (ただし，$0 < \epsilon \ll 1$)．

まず，重み摂動法について考える．モデルのパラメータを $\boldsymbol{\theta} \in \mathbb{R}^P$ とする．これは $\mathbf{W}_\ell$ および $\mathbf{b}_\ell$ をまとめたベクトルであり，$P$ はパラメータ空間の次元である．$\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_P)\ (\in \mathbb{R}^P)$ とすると，$\sigma\to 0$ の場合，

$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\cdot \mathbf{v} = \frac{\mathcal{L}(\boldsymbol{\theta}+\sigma \mathbf{v}) - \mathcal{L}(\boldsymbol{\theta})}{\sigma}=\frac{\delta \mathcal{L}}{\sigma}
\end{equation}
$$

となるので，

$$
\begin{align}
\mathbb{E}\left[\frac{\delta \mathcal{L}}{\sigma}\mathbf{v}\right] &=
\mathbb{E}\left[\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\cdot \mathbf{v}\right)\mathbf{v}\right]\\
&=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \mathbb{E}[\mathbf{v} \mathbf{v}^\top]=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\quad \left(\because \mathbb{E}[\mathbf{v} \mathbf{v}^\top]=\mathbf{I}_P\right)
\end{align}
$$

が成立する．SGDでパラメータ更新する場合は

$$
\begin{equation}
\mathbb{E}[\Delta \mathbf{W}_\ell]=-\eta \dfrac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell},\quad \mathbb{E}[\Delta \mathbf{b}_\ell]=-\eta \dfrac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}
\end{equation}
$$

であればいいので，$(\boldsymbol{\theta}, \mathbf{v}) \to (\mathbf{W}_\ell, \mathbf{V}_\ell), (\mathbf{b}_\ell, \mathbf{v}_\ell)$ と置き換えて

$$
\begin{equation}
\Delta \mathbf{W}_\ell^{\mathrm{WP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{V}_\ell,\quad \Delta \mathbf{b}_\ell^{\mathrm{WP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell
\end{equation}
$$

となる．ノード摂動法はパラメータのうちバイアス項のみを摂動する重み摂動法であると解釈できるため，$\Delta \mathbf{b}_\ell^{\mathrm{NP}}:=\Delta \mathbf{b}_\ell^{\mathrm{WP}}$ とすることができる．ここで

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_\ell}&=\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell} \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{W}_\ell}\\
&=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{a}_\ell}\frac{\partial \mathbf{a}_\ell}{\partial \mathbf{b}_\ell}\right) \mathbf{z}_\ell^\top\quad \left(\because \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{b}_\ell}=\mathbf{1}\right)\\
&=\frac{\partial \mathcal{L}}{\partial \mathbf{b}_\ell}\mathbf{z}_\ell^\top
\end{align}
$$

が成り立つので，ノード摂動法の更新則は

$$
\begin{equation}
\Delta \mathbf{W}_\ell^{\mathrm{NP}}:=\Delta \mathbf{b}_\ell^{\mathrm{NP}}\mathbf{z}_\ell^\top=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell\mathbf{z}_\ell^\top,\quad \Delta \mathbf{b}_\ell^{\mathrm{NP}}:=-\eta \frac{\delta \mathcal{L}}{\sigma}\mathbf{v}_\ell
\end{equation}
$$

と設定できる．


Chaotic neural dynamics facilitate probabilistic computations through sampling

Effective Learning with Node Perturbation in Multi-Layer Neural Networks (fig1は図の参考になる．)
On the stability and scalability of node perturbation learning
Node perturbation learning without noiseless baseline


重み摂動法 (Weight perturbation; WP)
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021006
Weight Perturbation Learning Performs Similarly or Better than Node Perturbation on Broad Classes of Temporally Extended Tasks

A. Dembo and T. Kailath, Model-Free Distributed Learning, IEEE Trans. Neural Networks 1, 58 (1990).

G. Cauwenberghs, A Fast Stochastic Error-Descent Algorithm for Supervised Learning and Optimization, in Advances in Neural Information Processing Systems (Morgan Kaufmann, Burlington, 1993), Vol. 5, pp. 244–251