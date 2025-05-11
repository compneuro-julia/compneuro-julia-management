## 神経細胞の発火率モデル
神経細胞は複雑な構造と機能を有する特殊な細胞であるが，その基本的な働きを理解するためには，ある程度抽象化された単純なモデルの導入が有用である．本章から第5章までは，この目的のもとに形式ニューロンや発火率モデルといった抽象的な数理モデルを用いることとし，チャネル動態などを含む詳細な生物物理モデルについては第6章で扱う\footnote{神経細胞のモデル化においては，詳細なモデルから抽象モデルを導出する記述順も考えられるが，本書では，コードのまとまりを考慮して，先に単純なモデルから入り，徐々に生物物理学的に忠実なモデルへと発展させる構成とする．}．本章で取り上げる**発火率モデル**（firing rate model）は，神経細胞の発火活動を平均的な頻度，すなわち**発火率**（firing rate）として記述する枠組みであり，この連続的な発火率により情報を表現する形式を**発火率による符号化**（rate coding）と呼ぶ．

### 離散時間モデル
発火率モデルでは，シナプス前細胞の活動の重みづけ和に基づき，シナプス後細胞の出力が決定される．まず，時間を離散的に扱い，時間発展を考慮しない場合のモデルを導入しよう．$n$ 個のシナプス前細胞の活動をベクトル $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top \in \mathbb{R}^n$，シナプス重みを $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top \in \mathbb{R}^n$，バイアス項を $b \in \mathbb{R}$，シナプス後細胞の出力を $y \in \mathbb{R}$ とすると，離散時間発火率モデルは以下のように定式化される：

$$
\begin{equation}
y = f(\mathbf{w}^\top \mathbf{x} + b) = f\left(\sum_{i=1}^n w_i x_i + b\right)
\end{equation}
$$

ここで，$f(\cdot)$ は入出力関係を表す関数であり，**活性化関数**（activation function）または**伝達関数**（transfer function）と呼ばれる．生理学的観点からこの式を解釈すると，$\mathbf{x}$ はシナプス前細胞から送られる発火率（に比例する量），$\mathbf{w}$ は各シナプス結合の強度を反映した重みであり，その内積 $\mathbf{w}^\top \mathbf{x}$ はシナプス後細胞に流入する総電流に相当する\footnote{2つの神経細胞間は1つのシナプス結合しか繋がっていないわけではなく，冗長な結合が存在する．ここではそうした複数のシナプス結合によるシナプス後細胞への影響の総和を取ってシナプス重みとしている．}．バイアス項 $b$ は，発火閾値（神経細胞の興奮性などの電気的特性）や定常的な興奮性入力などを含む項として解釈される．出力 $y$ は，神経細胞の平均的な発火率（に比例する量）とみなすことができ，活性化関数 $f(\cdot)$ はこの電流入力に応じた発火頻度の変化を表す関数，すなわち**周波数–電流曲線** (F-I曲線, frequency-current curve) に対応する．このF-I曲線の具体的形状と導出については，第6章で詳しく扱う．

活性化関数には線形関数と非線形関数の両方が用いられる．活性化関数を恒等写像 $f(x) = x$ とした場合，このモデルは線形回帰と同型になり，こうしたモデルを**線形ニューロンモデル** (linear neuron model) と呼ぶ．非線形な活性化関数は、線形関数に比べて実際の神経活動の電気的性質をより適切に反映し、その特性に応じて様々な関数が用途別に用いられる。まず、**Heavisideの階段関数** (Heaviside step function) あるいは単にHeaviside関数 $H(\cdot)$ は、入力が0以上であれば1を出力し、それ以外は0を出力する不連続な関数であり、次式で定義される：

$$
\begin{equation}
f(x) = H(x):=
\begin{cases}
1 & (x \geq 0) \\
0 & (x < 0)
\end{cases}
\end{equation}
$$

Heaviside関数は、閾値を境に出力が離散的に変化するという **全か無かの法則** (all-or-none principle) を表現するための最も基本的な関数として位置づけられる。なお，$H(0)$ の値には複数の定義が存在し、主に $0$, $\frac{1}{2}$, $1$ のいずれかを取ることがあるが、本書では $H(0) = 1$ を採用する。次に、**符号関数**（sign function）$\mathrm{sgn}(\cdot)$は、入力の正負に応じて $+1$ または $-1$ を出力し、ゼロの場合には出力0を与える：

$$
\begin{equation}
f(x) = \mathrm{sgn}(x):=
\begin{cases}
1 & (x > 0) \\
0 & (x = 0) \\
-1 & (x < 0)
\end{cases}
\end{equation}
$$

符号関数を用いれば、次の細胞に与える影響が興奮性であれば正、抑制性であれば負として表現できる。これは、Daleの法則を無視するか、あるいは他の神経細胞を介した間接的な効果として解釈する場合に限られるが、神経活動における興奮と抑制という対照的な作用を簡潔に記述する手法となる。

Heaviside関数と符号関数は、いずれも $x=0$ に不連続点をもち、通常の解析においては微分不可能であるため、理論的に扱いにくい場合がある。これらの関数を連続的かつ滑らかな非線形関数で近似する関数として、**シグモイド関数** (sigmoid function, logistic function) および**双曲線正接関数** (tanh function) がある。以降，後者の関数に関しては，tanh関数と呼称する．

まず、シグモイド関数 $\mathrm{sigmoid}(\cdot)$ はS字型の形状を持ち、実数値の入力を $[0, 1]$ の範囲に滑らかに写像する関数であり、次の式で定義される：

$$
\begin{equation}
f(x) = \mathrm{sigmoid}(x):= \frac{1}{1 + e^{-x}}
\end{equation}
$$

一方、tanh関数 $\tanh(\cdot)$ も類似したS字型の形状を持つが、その出力は $[-1, 1]$ の範囲にわたり、より対称的な性質を持つ。定義は以下の通りである：

$$
\begin{equation}
f(x) = \tanh(x) := \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}}
\end{equation}
$$

また、両者の間には次のような明確な関係がある：

$$
\begin{equation}
\tanh(x) = 2\cdot \mathrm{sigmoid}(2x) - 1
\end{equation}
$$

このように、tanh関数はシグモイド関数に線形変換を施すことで得ることができる。さらに、シグモイド関数およびtanh関数に**逆温度**（inverse temperature）$\beta\ (>0)$ を導入することで、関数の遷移の鋭さ（sharpness）を調整することが可能である。逆温度付きのシグモイド関数およびtanh関数は、以下のように定義される：

$$
\begin{align}
\mathrm{sigmoid}_\beta(x; \beta)&:= \frac{1}{1 + e^{-\beta x}}\\
\tanh_\beta(x; \beta)&:= \frac{e^{\beta x} - e^{-\beta x}}{e^{\beta x} + e^{-\beta x}}
\end{align}
$$

ここで、$\beta \to \infty$ の極限を取ると、$\mathrm{sigmoid}_\beta(x; \beta) \to H(x)$、および $\tanh_\beta(x; \beta) \to \mathrm{sgn}(x)$ となる．すなわち，シグモイド関数およびtanh関数は，それぞれHeaviside関数と符号関数に近づく。なお、これらの温度付き関数は、関数の定義そのものを新たに与える必要はなく、元の関数の引数を単に $x \to \beta x$ と置換することで導入することができる。

非線形だが飽和しない関数として、**ランプ関数** (ramp function)がある \citep{householder1941theory}。この関数は、後に**ReLU関数** (rectified linear unit function, 正規化線形関数) と呼ばれるようになり \citep{nair2010rectified}、現在ではこの名称の方が広く一般に定着している。本書では、機械学習の事項も扱うため，統一的に「ReLU関数」という名称を用いる。ReLU関数 $\textrm{ReLU}(\cdot)$ は、入力が負のときに0を出力し、正のときはそのままの値を出力する関数であり，

$$
\begin{equation}
f(x) = \textrm{ReLU}(\cdot):=\max(0, x)
\end{equation}
$$

と表される．ただし，$\max(a, b)$は, $a, b$のうち，大きい値を返す関数である．ReLU関数はHeaviside関数や符号関数などのように不連続点はないが，$x=0$ において微分不可能である．ReLU関数は非線形関数であるが，区間ごと ($x\geq 0$ および $x<0$) に見ると線形であるため，**区分線形関数** (piecewise linear function) に含まれる．

シグモイド関数やtanh関数は、神経細胞が強い入力に対して発火率を飽和させるという生理的特性を捉えているが、その出力範囲（dynamic range）は限られており、極端な入力に対しては出力がほぼ一定となる。このとき出力の分布は0または1（あるいは $\pm 1$）付近に偏るため、エントロピー、すなわち出力の不確実性や多様性が低下する。情報理論的に見れば、出力のエントロピーが低いということは、活性化関数が伝達できる情報量が減少していることを意味し、結果として表現力が制限される。このように、飽和を起こす関数では入力に応じた情報の分解能が失われやすく、信号伝達の効率という観点から不利である。こうした問題を避けるため、出力が飽和せず、広い入力範囲にわたってより多様な出力を保持できるReLU関数が、深層学習を中心に広く用いられている。なお，機械学習でReLU関数が使用されるのには勾配消失問題に対処できるという性質もあるが，これに関しては第4章で触れる．

#### 形式ニューロンとパーセプトロン
この離散時間モデルは，McCullochとPittsによって1943年に提案された**形式ニューロン**（formal neuron, McCulloch–Pitts neuron）に起源を持つ \citep{mcculloch1943logical}．形式ニューロンでは，活性化関数としてHeaviside関数が用いられる．このモデルは出力が0か1のいずれかであり，発火率ではなくスパイクの発火の有無を二値的に表現するものである．したがって，形式ニューロンは「入力の重みづけ和がある閾値 $\theta \ (=-b)$ を超えるかどうか」によって出力を決定する**閾値判定器**として機能する．

このような形式ニューロンに学習機構を導入したモデルが，1958年にRosenblattによって提案された**パーセプトロン**（perceptron）である \citep{rosenblatt1958perceptron}．ただし，Rosenblattは形式ニューロンをただ使用するのではなく，形式ニューロンも含めた神経回路網のモデルを提案した．例えばRosenblattによって単純パーセプトロン (simple perceptron, Mark I perceptron)と呼ばれたモデルは，3つのユニット群（感覚ユニット，連想ユニット，応答ユニット）から構成されていた．感覚ユニットと連想ユニットはランダムなシナプス重みで結合\footnote{入力信号のランダム重みによる投射は第8章で触れるリザバーコンピューティングと同様の形態である．}しており，連想ユニットと応答ユニット間には双方向の結合が存在していた．これに対して，後に提案された簡略化されたモデルでは，連想ユニットと双方向の結合を排除し，感覚ユニットと応答ユニットを直接結合する形となっており，これが現在一般的に用いられている**現代的パーセプトロン**（modern perceptron）あるいは**単純パーセプトロン** (simple perceptron) である．したがって，本節で紹介した離散時間の発火率モデルは，広義にはこの単純化されたパーセプトロンに対応する．なお，パーセプトロンの学習則に関しては次々節で詳解を行う．

##### コラム：活性化関数にシグモイド関数を用いる場合の補足
活性化関数としてシグモイド関数を用いた場合，このモデルはロジスティック回帰と同型になる \footnote{パーセプトロンとロジスティック回帰は同年（1958年）に提案された．}．この場合の出力は「スパイク発生の確率」と「発火率」の双方の解釈が可能である．すなわち，出力が $[0, 1]$ の範囲に正規化されているため，「ある入力に対して神経細胞が発火する確率」として解釈することも，「ある入力に対する平均的な発火頻度を正規化した値」として解釈することもできる．

F-I曲線の形状としては，シグモイド関数のような飽和関数（saturated function）が用いられることが多いが，実際の神経細胞では多くの場合完全な飽和には至らず，部分的な飽和挙動を示す\footnote{ただし，シグモイド関数に渡す入力を適切にスケーリングすることで飽和を防ぐことは可能である．}．そのような特性を表現する関数として，以下のようなNaka–Rushton関数が用いられることもある\citep{naka1966s,sclar1990coding,wilson1999spikes}：

$$
\begin{equation}
s(x) = \frac{Mx^a}{\sigma^a + x^a} \cdot \Theta(x)
\end{equation}
$$

ここで，$M$ は最大応答，$\sigma$ は感度定数，$a$ は正の指数，$\Theta(x)$ は正の電流入力に対してのみ応答する制限を導入するための関数である．この関数は，入力に対して初期は急峻に応答し，その後徐々に応答が飽和する非線形性を持つ点で，生理的F-I曲線により近い特性を示す．

例えば，$M=100, \sigma=25, a=2.4$ の場合は次のようになる．

#### 多出力形式への拡張
発火率モデルは1つのシナプス後細胞を対象としたものであったが，容易に多出力形式へと拡張可能である．複数のシナプス後細胞が同一のシナプス前細胞群から投射を受けるとし，$m$ 個のシナプス後細胞を $\mathbf{y} \in \mathbb{R}^m$，重み行列を $\mathbf{W} \in \mathbb{R}^{m \times n}$，定常項 (bias) を $\mathbf{b} \in \mathbb{R}^m$ とすれば，多出力形式のモデルは次のように表される：

$$
\begin{equation}
\mathbf{y} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
\end{equation}
$$

この形式では，入力ベクトル $\mathbf{x}$ から出力ベクトル $\mathbf{y}$ への変換が一組の線形変換 $\mathbf{W}\mathbf{x} + \mathbf{b}$ と非線形変換 $f(\cdot)$ からなる操作で構成されており，機械学習においてはこの一連の変換過程を**層**（layer）と呼ぶ．さらに層を区分化し，線形変換部分を**線形層**（linear layer）あるいは**全結合層**（fully connected layer），非線形変換 $f(\cdot)$ を**活性化層**（activation layer）と呼ぶこともある．このような層を複数連結し，ある層の出力が次の層の入力となるようにしたモデルが**多層パーセプトロン**（multilayer perceptron; MLP）である．MLPは**ニューラルネットワーク** (neural network; NN) の基本的な構造であり，第4章で詳解する．

#### 再帰型結合の追加
ここまで神経細胞間に順方向結合 (feedforward connection) しかない，すなわち電気的活動（あるいは情報）が順伝播するモデルを取り扱った．次に，時間発展とシナプス後細胞群の再帰的結合 (recurrent connection) も考慮したモデルを考える．時刻 $t$ の活動を $\mathbf{x}_t \in \mathbb{R}^n, \mathbf{y}_t \in \mathbb{R}^m$ とし，順方向結合重みを $\mathbf{W} \in \mathbb{R}^{m\times n}$，再帰的結合重みを $\mathbf{M} \in \mathbb{R}^{m \times m}$ とすると\footnote{ここで重みの時間発展は無視しているが，オンライン学習を行う場合は考慮が必要である．}，シナプス後細胞群の離散時間における再帰的発火率モデルは次のように表される：

$$
\begin{equation}
\mathbf{y}_{t+1} = f(\mathbf{W} \mathbf{x}_{t} + \mathbf{M}\mathbf{y}_t+ \mathbf{b})
\end{equation}
$$

このような再帰的結合を含むネットワークを**再帰型ニューラルネットワーク** (recurrent neural network; RNN) と呼ぶ．
なお，$\mathbf{M}$ の対角成分が0以外の場合は，神経細胞の軸索終末がそれ自身の樹状突起と結合している状態を意味する．こうしたシナプス結合をオータプス (autapse) と呼ぶ．

### 連続時間モデル
次に，**連続時間における発火率モデル**（continuous-time firing rate model）について説明する．このモデルは，神経細胞の出力が入力に対して緩やかに時間変化する様子を記述するものであり，出力が入力に対してローパスフィルタ的に応答する構造を持つ．連続時間モデルにおいて，時刻 $t$ における出力 $\mathbf{y}(t)$ の時間変化は次のように表される：

$$
\begin{equation}
\tau \frac{\mathrm{d}\mathbf{y}(t)}{\mathrm{d}t} = -\mathbf{y}(t) + f(\mathbf{W} \mathbf{x}(t) + \mathbf{M} \mathbf{y}(t)+\mathbf{b})
\end{equation}
$$

ここで，$\tau$ は時定数であり，神経細胞の応答の時間スケールを決定するパラメータである．このモデルにおいて第一項の $-\mathbf{y}(t)$ は，入力が存在しない場合に神経活動が自然に減衰する性質を記述している．また $\mathbf{x}(t)$ が一定であり，$\frac{\mathrm{d}\mathbf{y}(t)}{\mathrm{d}t}=0$ を満たす **平衡点** (equilibrium point, fixed point) $\mathbf{y}^*$ が存在するとき，平衡点は離散時間モデルの更新式と同様の形式 $\mathbf{y}^* = f(\mathbf{W}\mathbf{x} + \mathbf{M} \mathbf{y}^* + \mathbf{b})$ を満たす．

コンピュータ上でシミュレーションする上では連続時間モデルを離散化する必要がある．単純な離散化手法として，Euler近似を用いると，次のように記述できる：

$$
\begin{align}
\mathbf{y}_{t+1} &= \mathbf{y}_t + \frac{\Delta t}{\tau} \left[ -\mathbf{y}_t + f(\mathbf{W} \mathbf{x}_t + \mathbf{M} \mathbf{y}_t + \mathbf{b}) \right]\\
&= (1-\alpha)\cdot \mathbf{y}_t + \alpha \cdot f(\mathbf{W} \mathbf{x}_t + \mathbf{M} \mathbf{y}_t + \mathbf{b})
\end{align}
$$

ただし，$\alpha := \frac{\Delta t}{\tau}$ とした．また，出力 $\mathbf{y}(t)$ を内部状態 $\mathbf{u}(t)$ を通じて間接的に記述する形式もある：

$$
\begin{align}
\tau \frac{\mathrm{d}\mathbf{u}(t)}{\mathrm{d}t} &= -\mathbf{u}(t) + \mathbf{W} \mathbf{x}(t) + \mathbf{M} \mathbf{y}(t) + \mathbf{b} \\
\mathbf{y}(t) &= f(\mathbf{u}(t))
\end{align}
$$

この2段階の構造では，$\mathbf{u}(t)$ を膜電位，$\mathbf{y}(t)$ を発火率と解釈することができ，より生理学的に妥当なモデルとなる．

#### 多細胞形式のWilson–Cowanモデル
前節で導入した連続時間発火率モデルは，WilsonとCowanによって提案された，縮約化された神経回路網の数理モデル（**Wilson–Cowanモデル**）に基づいている \citep{wilson1972excitatory, wilson1973mathematical, wilson2021evolution, chow2020before}．このモデルは，個々の神経細胞の発火活動を扱うのではなく，興奮性あるいは抑制性の神経細胞群における平均的な発火活動の時間変化を連続時間の微分方程式として記述するものである．そのため，Wilson–Cowanモデルは神経集団の力学を対象とする**集団モデル**（population model）の一種であり，特に大脳皮質などにおけるマクロな神経活動の時空間的構造を解析する目的で広く用いられている．元来のWilson–Cowanモデルは2つの変数からなる連立微分方程式で構成され，それぞれが興奮性および抑制性の神経集団の平均発火率を表している．本節では，このモデルにおける1変数を1個の神経細胞群の代表値ではなく1個の神経細胞の活動と解釈し，並列化することで多細胞系へと拡張したモデルを導入する\footnote{元のWilson-Cowanモデルと同様に解釈し，多数の神経細胞群間での相互作用のモデルと解釈することも可能である．}．さらに，この拡張された形式が前節で述べた一般的な連続時間発火率モデルと一致する構造を持つことを明示し，両者の関係を明らかにする．

以下では興奮性 (excitatory) あるいは抑制性 (inhibitory) の細胞群が関わる変数にそれぞれ添え字 $\mathrm{E}$ あるいは $\mathrm{I}$ をつける．また表記を簡略化するため，添え字 $\alpha, \beta$ が $\mathrm{E}$ あるいは $\mathrm{I}$ を表すとする．まず，各細胞群の活動を $\mathbf{y}_\alpha(t) \in \mathbb{R}^{n_\alpha}$ とする．ここで $n_\mathrm{E}=n_\mathrm{I}=1$ とすれば元の形のWilson–Cowanモデルとなる．時定数を $\tau_\alpha$，細胞群への入力を $\mathbf{x}_\mathrm{\alpha}(t) \in \mathbb{R}^{n_\alpha}$，細胞群間での結合行列を $\mathbf{W}_{\alpha \beta}\in \mathbb{R}^{n_\alpha\times n_\beta}$とする．
ただし，$\mathbf{y}_\alpha, \mathbf{W}_{\alpha \beta}$ の要素はすべて非負である．この場合，時間的に粗視化した多細胞形式のWilson–Cowan方程式は次のように定式化される：

$$
\begin{aligned}
\tau_\mathrm{E} \frac{\mathrm{d}\mathbf{y}_\mathrm{E}(t)}{\mathrm{d}t} &= -\mathbf{y}_\mathrm{E}(t) + f_\mathrm{E} \left[ \mathbf{W}_\mathrm{EE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{EI} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{E}(t) \right] \\
\tau_\mathrm{I} \frac{\mathrm{d}\mathbf{y}_\mathrm{I}(t)}{\mathrm{d}t} &= -\mathbf{y}_\mathrm{I}(t) + f_\mathrm{I} \left[ \mathbf{W}_\mathrm{IE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{II} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{I}(t) \right]
\end{aligned}
$$

このモデルは，それぞれの神経集団が自己結合および他集団からの入力を受け取り，非線形な活性化関数を通じて平均発火率を変化させることを記述している．ここで，活性化関数 $f_\mathrm{E}, f_\mathrm{I}$ は各集団に応じた非線形性を表し，通常はシグモイド関数のような単調増加かつ飽和的な関数が用いられる\footnote{Wilson–Cowanモデルのより原型に近い形式では，活性化関数の前に $(1 - r_\alpha \mathbf{y}_\alpha(t))$ を乗じる．この項は，神経細胞の発火率が生理学的に上限をもつという事実，すなわち飽和的性質を数理モデルに明示的に組み込むことを意図して導入されたものである．ただし，$r_\alpha=0$ と設定した場合でも，モデルの定性的挙動には大きな差が見られないことが知られている \citep{wilson2021evolution}．したがって，本書では記述の簡明さを優先し，この飽和項は導入せずにWilson–Cowanモデルを扱うこととする．}．

この連立微分方程式は，ベクトルとブロック行列を用いて単一の微分方程式に統合することができる．まず，興奮性・抑制性集団の発火率をまとめたベクトルとして $\mathbf{y}(t) := \begin{bmatrix} \mathbf{y}_\mathrm{E}(t) \\ \mathbf{y}_\mathrm{I}(t) \end{bmatrix}$ を定義し，外部入力 $\mathbf{x}(t)$ も同様に結合されたベクトル $\mathbf{x}(t) := \begin{bmatrix} \mathbf{x}_\mathrm{E}(t) \\ \mathbf{x}_\mathrm{I}(t) \end{bmatrix}$ とする．各結合を表す行列をブロック行列としてまとめることで，再帰的結合行列 $\mathbf{M}$ は

$$
\begin{equation}
\mathbf{M} := \begin{bmatrix}
\mathbf{W}_\mathrm{EE} & -\mathbf{W}_\mathrm{EI} \\
\mathbf{W}_\mathrm{IE} & -\mathbf{W}_\mathrm{II}
\end{bmatrix}
\end{equation}
$$

と定義できる．さらに，時定数を要素ごとに異なる対角行列 $\boldsymbol{\tau}$ として

$$
\begin{equation}
\boldsymbol{\tau} := \begin{bmatrix}
\tau_\mathrm{E} \mathbf{I} & \mathbf{0} \\
\mathbf{0} & \tau_\mathrm{I} \mathbf{I}
\end{bmatrix}
\end{equation}
$$

と表す．さらに活性化関数 $f(\cdot)$ が $f_\mathrm{E}$ や $f_\mathrm{I}$ を要素ごとに適用するベクトル値関数とすると，Wilson-Cowan方程式は次のように統合された形式にまとめることができる：

$$
\begin{equation}
\boldsymbol{\tau} \frac{\mathrm{d}\mathbf{y}(t)}{\mathrm{d}t} = -\mathbf{y}(t) + f(\mathbf{M} \mathbf{y}(t) + \mathbf{x}(t))
\end{equation}
$$

この形式にまとめることで，Wilson–Cowanモデルは，前節で紹介した一般的な連続時間発火率モデルの枠組みの中に位置づけることができる．なお，外部入力の構造をより詳細に反映したい場合には，$\mathbf{x}(t) \to \mathbf{W} \mathbf{x}(t) + \mathbf{b}$ と置換し，入力重みと定常項（バイアス）を明示的に導入することも可能である．

#### 抑制安定化ネットワークと安定性解析
Wilson–Cowanモデル等でモデル化される興奮性および抑制性神経細胞群が相互作用する神経回路網は，各シナプス結合強度のバランスが取れていなければ活動は不安定（発散あるいは静止）となる\footnote{シナプス結合強度のバランスを保ったモデルを学習させる手法として\citep{soldado2022paradoxical}などがある．}．抑制性神経集団により神経回路網全体の活動が安定化されている神経回路網（あるいはその状態）を**抑制安定化ネットワーク** (inhibition-stabilized network; ISN) と呼ぶ \citep{sadeh2021inhibitory}．

ISNでは，興奮性細胞間の結合が強いために本来は活動が発散してしまうところを，抑制性細胞の抑制があることでネットワークが安定化されている．このような状態で生じる現象が**逆説的効果** (paradoxical effect) であり \citep{tsodyks1997paradoxical}，これは抑制性細胞に興奮性刺激を加えると，逆に抑制性細胞群の活動が減少するという現象である．逆説的効果は抑制性細胞の活動が高まると興奮性細胞への抑制が強まり，結果として興奮性細胞からの抑制性細胞への入力が減少するために生じる．逆の過程として，抑制性細胞に抑制性刺激を加えると逆に活動が増加するという現象も逆説的効果と呼ばれる．このような逆説的効果が生理的に存在することを示した実験研究としては \citep{ozeki2009inhibitory,kato2017network,sanzeni2020inhibition} があり，ISNという枠組みが生体脳においても成立しうることを支持している．

Wilson–CowanモデルはISN状態を含むため，ISNの数理的性質を理論的に解析する際の基礎モデルとして広く用いられている．あるモデルがISNであるかどうかを調べるには，まず**固定点解析**（fixed point analysis）\footnote{安定性解析まで含めて固定点解析と呼ぶ場合もある．} により系の平衡点（固定点）を求める．次に，その平衡点が安定かどうかは，線形安定性解析（linear stability analysis）によって判定される \citep{strogatz2024nonlinear}．線形安定性解析は，具体的には，非線形モデルを平衡点まわりで線形化し，ヤコビ行列（Jacobian matrix）を導出したうえで，その固有値の実部の符号を調べれば，安定性の有無を判定できる．以下では，興奮性・抑制性細胞が統合された形式でのWilson-Cowan方程式を例として，安定性解析を行う．なお，外部入力 $\mathbf{x}$ は一定であるとし，全ての細胞の活性化関数 $f(\cdot)$ が同一であるとする．まず，平衡点 $\mathbf{y}^*$ に小さな摂動$\delta \mathbf{y}(t)$を加え，$\mathbf{y}(t) = \mathbf{y}^* + \delta \mathbf{y}(t)$ として元の微分方程式に代入する：

$$
\begin{equation}
\tau \frac{\mathrm{d}}{\mathrm{d}t} (\mathbf{y}^* + \delta \mathbf{y}) = -(\mathbf{y}^* + \delta \mathbf{y}) + f(\mathbf{M} (\mathbf{y}^* + \delta \mathbf{y}) + \mathbf{x})
\end{equation}
$$

右辺第2項をTaylor展開すると，

$$
\begin{equation}
f(\mathbf{M} (\mathbf{y}^* + \delta \mathbf{y}) + \mathbf{x})
=f(\mathbf{M} \mathbf{y}^* + \mathbf{x}) + \mathbf{D}_f \mathbf{M}\delta \mathbf{y} + \mathcal{O}(\|\delta \mathbf{y}\|^2)
\end{equation}
$$

となる．ただし，$\mathbf{D}_f := \mathrm{diag}\left(f'(\mathbf{M} \mathbf{y}^* + \mathbf{x})\right)$ は固定点における微分値を並べた対角行列である（$\mathrm{diag}(\cdot)$ はベクトルの各要素を対角要素に持つ対角行列を作る関数）．また，$\mathcal{O}(\cdot)$ はLandauの略記であり，$\mathcal{O}(\|\delta \mathbf{y}\|^2)$ は $\delta \mathbf{y}$ の2次以上の項を意味する．
ここで $\frac{\mathrm{d}\mathbf{y}^*}{\mathrm{d}t}=0$ かつ $-\mathbf{y}^*+f(\mathbf{M} \mathbf{y}^* + \mathbf{x})=0$ が成り立つことも踏まえ，$\mathcal{O}(\|\delta \mathbf{y}\|^2)$ を無視して整理すると，平衡点周囲の線形系が得られる：

$$
\begin{align}
\tau \frac{\mathrm{d}\delta \mathbf{y}}{\mathrm{d}t} &= -\delta \mathbf{y} + \mathbf{D}_f \mathbf{M} \delta \mathbf{y}\\
&=\left(-\mathbf{I} + \mathbf{D}_f \mathbf{M} \right) \delta \mathbf{y} = \mathcal{J} \delta \mathbf{y}
\end{align}
$$

ここで，$\mathcal{J}:=-\mathbf{I} + \mathbf{D}_f \mathbf{M}$ を平衡点 $\mathbf{y}^*$ におけるヤコビ行列という．平衡点が漸近的に安定 (asymptotically stable) であるためには，ヤコビ行列 $\mathcal{J}$ の全ての固有値 $\lambda$ の実部が負，すなわち $\mathrm{Re}(\lambda) < 0$ を満たせばよい．この場合，ネットワークは安定となる．

なお，機能的なRNNの内部機構を明らかにする手法としても，固定点解析および安定性解析は有用である \citep{sussillo2013opening}．Sussilloらの解析に基づいて2つの例を紹介する．1つ目の例として，パルス入力の符号を記憶するフリップフロップ（flip-flop）課題を考える．この課題では，入力ごとに「+1」または「−1」のパルスが与えられ，RNNはその最新の入力値を出力として保持する．フリップフロップ課題を実行するRNNを固定点解析すると，各記憶状態が安定固定点として実装されており，状態遷移は鞍点を通じて実現されることが示された．このように，RNNが「記憶の遷移」をどのように表現しているかを，固定点空間の構造として記述することができる．2つ目の例として，正弦波のような周期的な動的パターンを生成する課題を考える．この課題では，RNNの出力は時間とともに変化し続けるため，状態空間の軌道は固定点から大きく離れる場合が多い．しかしながら，各周波数に対して対応する静的入力を与えた条件下でRNNの固定点を求め，その周囲の線形化を行うと，得られる線形系は振動的な不安定性（虚部をもつ複素固有値）を示す．その虚部の大きさは，目標とする正弦波の角周波数と一致しており，線形近似によってRNNの出力振動の周期性を的確に説明できることが示されている．従って，このような動的挙動であっても，固定点周辺の線形ダイナミクスが振動パターンの生成機構を支配していると解釈できる．なお，こうした機能的なRNNを訓練する方法に関しては主に第5章で詳解する．

#### 神経集団モデルと神経場モデル
Wilson–Cowanモデルと密接に関連し，より大域的・集団的な神経活動を記述する枠組みとして，**神経集団モデル**（neural mass model, neural population model）および**神経場モデル**（neural field model）があり，これらのモデルに関して簡単に触れておく．いずれも個々のニューロンの詳細な活動ではなく，神経細胞集団の平均的な膜電位や発火率の時間変化を対象とし，脳波などのマクロな神経活動を記述するための理論的枠組みを提供する．

神経集団モデルでは，皮質のマイクロカラムや局所回路といった小規模な神経集団を1ユニットとしてモデル化し，Wilson–Cowanモデルと同様に，平均発火率や膜電位のダイナミクスを扱う．神経集団モデルの例としては局所神経回路をモデル化したJansen-Ritモデル \citep{jansen1995electroencephalogram, david2003neural} や，てんかん活動の動態を記述するWendlingモデル \citep{wendling2002epileptic} などがある．

一方，神経場モデル（neural field model）では，神経活動を空間的に連続な関数として記述し，広範囲における神経活動の時空間的なダイナミクスを扱う \citep{coombes2014tutorial, cook2022neural}．神経場モデルはWilsonおよびCowan \citep{wilson1973mathematical}, Nunez \citep{nunez1974brain}, 甘利 \citep{amari1975homogeneous, amari1977dynamics} らの研究に基づいており，ここでは甘利による定式化（**甘利モデル**, Amari model）を簡単に説明する．甘利モデルでは，まず神経場の定義域 $\Omega$（一次元の皮質断面や二次元の皮質平面など）を考える．$\Omega$ における神経活動は以下のような積分–微分方程式によって与えられる：

$$
\begin{equation}
\tau \frac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w(x, x') f(u(x', t))\,\mathrm{d}x' + I(x,t)
\end{equation}
$$

ここで，$x, x' \in \Omega$ は神経場における位置を表す．$u(x,t)$ は位置 $x$ における時刻 $t$ の発火率，$w(x,x')$ は位置 $x'$ から $x$ への結合重み，$f(\cdot)$ は活性化関数，$I(x,t)$ は外部入力を表す．神経場モデルは，皮質進行波（cortical travelling waves）\sitep{muller2018cortical} 等の現象を理論的に説明する手段となる．