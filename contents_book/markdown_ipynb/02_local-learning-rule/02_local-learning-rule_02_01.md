## 神経細胞の発火率モデル
神経細胞は複雑な構造と機能を有する特殊な細胞であるが，その基本的な働きを理解するためには，ある程度抽象化された単純なモデルの導入が有用である．本章から第5章までは，この目的のもとに形式ニューロンや発火率モデルといった抽象的な数理モデルを用いることとし，チャネル動態などを含む詳細な生物物理モデルについては第6章で扱う\footnote{神経細胞のモデル化においては，詳細なモデルから抽象モデルを導出する記述順も考えられるが，本書では，コードのまとまりを考慮して，先に単純なモデルから入り，徐々に生物物理学的に忠実なモデルへと発展させる構成とする．}．本章で取り上げる**発火率モデル**（firing rate model）は，神経細胞の発火活動を平均的な頻度，すなわち**発火率**（firing rate）として記述する枠組みであり，この連続的な発火率により情報を表現する形式を**発火率による符号化**（rate coding）と呼ぶ．

### 静的離散時間モデル
発火率モデルでは，シナプス前細胞の活動の重みづけ和に基づき，シナプス後細胞の出力が決定される．まず，時間を離散的に扱い，神経細胞の内部状態を考慮しない静的なモデルを導入しよう．$n$ 個のシナプス前細胞の活動をベクトル $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top \in \mathbb{R}^n$，シナプス重みを $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top \in \mathbb{R}^n$，定常項（バイアス項）を $b \in \mathbb{R}$，シナプス後細胞の出力を $y \in \mathbb{R}$ とすると，静的離散時間発火率モデルは以下のように定式化される：

$$
\begin{equation}
y = f(\mathbf{w}^\top \mathbf{x} + b) = f\left(\sum_{i=1}^n w_i x_i + b\right)
\end{equation}
$$

ここで，$f(\cdot)$ は入出力関係を表す関数であり，**活性化関数**（activation function）または**伝達関数**（transfer function）と呼ばれる．

この静的離散時間モデルは，McCullochとPittsによって1943年に提案された**形式ニューロン**（formal neuron, McCulloch–Pitts neuron）に起源を持つ \citep{mcculloch1943logical}．形式ニューロンでは，活性化関数としてHeaviside関数が用いられる．このモデルは出力が0か1のいずれかであり，発火率ではなくスパイクの発火の有無を二値的に表現するものである．したがって，形式ニューロンは「入力の重みづけ和がある閾値 $\theta \ (=-b)$ を超えるかどうか」によって出力を決定する**閾値判定器**として機能する．

このような形式ニューロンに学習機構を導入したモデルが，1958年にRosenblattによって提案された**パーセプトロン**（perceptron）である \citep{rosenblatt1958perceptron}．ただし，Rosenblattは形式ニューロンをただ使用するのではなく，形式ニューロンも含めた神経回路網のモデルを提案した．例えばRosenblattによって単純パーセプトロン (simple perceptron, Mark I perceptron)と呼ばれたモデルは，3つのユニット群（感覚ユニット，連想ユニット，応答ユニット）から構成されていた．感覚ユニットと連想ユニットはランダムなシナプス重みで結合\footnote{入力信号のランダム重みによる投射は第8章で触れるリザバーコンピューティングと同様の形態である．}しており，連想ユニットと応答ユニット間には双方向の結合が存在していた．これに対して，後に提案された簡略化されたモデルでは，連想ユニットと双方向の結合を排除し，感覚ユニットと応答ユニットを直接結合する形となっており，これが現在一般的に用いられている**現代的パーセプトロン**（modern perceptron）あるいは**単純パーセプトロン** (simple perceptron) である．したがって，本節で紹介した離散時間の発火率モデルは，広義にはこの単純化されたパーセプトロンに対応する．なお，パーセプトロンの学習則に関しては次々項で詳解を行う．

#### 多出力形式への拡張
前項での静的発火率モデルは1つのシナプス後細胞を対象としたものであったが，容易に多出力形式へと拡張可能である．複数のシナプス後細胞が同一のシナプス前細胞群から投射を受けるとし，$m$ 個のシナプス後細胞を $\mathbf{y} \in \mathbb{R}^m$，重み行列を $\mathbf{W} \in \mathbb{R}^{m \times n}$，定常項 (bias) を $\mathbf{b} \in \mathbb{R}^m$ とすれば，多出力形式のモデルは次のように表される：

$$
\begin{equation}
\mathbf{y} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
\end{equation}
$$

この形式では，入力ベクトル $\mathbf{x}$ から出力ベクトル $\mathbf{y}$ への変換が一組の線形変換 $\mathbf{W}\mathbf{x} + \mathbf{b}$ と非線形変換 $f(\cdot)$ からなる操作で構成されており，機械学習においてはこの一連の変換過程を**層**（layer）と呼ぶ．さらに層を区分化し，線形変換部分を**線形層**（linear layer）あるいは**全結合層**（fully connected layer），非線形変換 $f(\cdot)$ を**活性化層**（activation layer）と呼ぶこともある．このような層を複数連結し，ある層の出力が次の層の入力となるようにしたモデルが**多層パーセプトロン**（multilayer perceptron; MLP）である．MLPは**ニューラルネットワーク** (neural network; NN) の基本的な構造であり，第4章で詳解する．

#### 活性化関数
生理学的観点からこの式を解釈すると，$\mathbf{x}$ はシナプス前細胞から送られる発火率（に比例する量），$\mathbf{w}$ は各シナプス結合の強度を反映した重みであり，その内積 $\mathbf{w}^\top \mathbf{x}$ はシナプス後細胞に流入する総電流に相当する\footnote{2つの神経細胞間は1つのシナプス結合しか繋がっていないわけではなく，冗長な結合が存在する．ここではそうした複数のシナプス結合によるシナプス後細胞への影響の総和を取ってシナプス重みとしている．}．バイアス項 $b$ は，発火閾値（神経細胞の興奮性などの電気的特性）や定常的な興奮性入力などを含む項として解釈される．出力 $y$ は，神経細胞の平均的な発火率（に比例する量）とみなすことができ，活性化関数 $f(\cdot)$ はこの電流入力に応じた発火頻度の変化を表す関数，すなわち**周波数–電流曲線** (F-I曲線, frequency-current curve) に対応する．このF-I曲線の具体的形状と導出については，第6章で詳しく扱う．

本書では、活性化関数にベクトルを入力する場合、原則として各要素ごと (element-wise, point-wise) に計算が行われるものとする。ただし、Softmax関数のようにベクトル全体を参照して出力を決定する例外もある。

活性化関数には線形関数と非線形関数の両方が用いられる．活性化関数を恒等写像 $f(x) = x$ とした場合，このモデルは線形回帰と同型になり，こうしたモデルを**線形ニューロンモデル** (linear neuron model) と呼ぶ．非線形な活性化関数は、線形関数に比べて実際の神経活動の電気的性質をより適切に反映し、その特性に応じて様々な関数が用途別に用いられる。

##### Heavisideの階段関数・符号関数
まず、**Heavisideの階段関数** (Heaviside step function) あるいは単にHeaviside関数 $H(\cdot)$ は、入力が0以上であれば1を出力し、それ以外は0を出力する不連続な関数であり、次式で定義される：

$$
\begin{equation}
f(x) = H(x):=
\begin{cases}
1 & (x \geq 0) \\
0 & (x < 0)
\end{cases}
\end{equation}
$$

Heaviside関数は、閾値を境に出力が離散的に変化するという **全か無かの法則** (all-or-none principle) を表現するための最も基本的な関数として位置づけられる。なお，$H(0)$ の値には複数の定義が存在し、主に $0$, $\frac{1}{2}$, $1$ のいずれかを取ることがあるが、ここでは $H(0) = 1$ を採用する。次に、**符号関数**（sign function）$\mathrm{sgn}(\cdot)$は、入力の正負に応じて $+1$ または $-1$ を出力し、ゼロの場合には出力0を与える：

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

##### シグモイド関数・tanh関数
Heaviside関数と符号関数は、いずれも $x=0$ に不連続点をもち、通常の解析においては微分不可能であるため、理論的に扱いにくい場合がある。これらの関数を連続的かつ滑らかな非線形関数で近似する関数として、**シグモイド関数** (sigmoid function, logistic function) および**tanh関数** (双曲線正接関数) がある。

まず、シグモイド関数 $\mathrm{sigmoid}(\cdot)$ はS字型の形状を持ち、実数値の入力を $[0, 1]$ の範囲に滑らかに写像する関数であり、次の式で定義される：

$$
\begin{equation}
f(x) = \mathrm{sigmoid}(x):= \frac{1}{1 + e^{-x}}
\end{equation}
$$

活性化関数としてシグモイド関数を用いた場合，このモデルはロジスティック回帰と同型になる \footnote{パーセプトロンとロジスティック回帰は同年（1958年）に提案された．}．この場合の出力は「スパイク発生の確率」と「発火率」の双方の解釈が可能である．すなわち，出力が $[0, 1]$ の範囲に正規化されているため，「ある入力に対して神経細胞が発火する確率」として解釈することも，「ある入力に対する平均的な発火頻度を正規化した値」として解釈することもできる．

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

ここで、$\beta \to \infty$ の極限を取ると、$\mathrm{sigmoid}_\beta(x; \beta) \to H(x)$、および $\tanh_\beta(x; \beta) \to \mathrm{sgn}(x)$ となる．すなわち，シグモイド関数およびtanh関数は，それぞれHeaviside関数と符号関数に近づく。

$\beta \to \infty$

具体的には、$x > 0$ では $\mathrm{sigmoid}_\beta(x)\to \frac{1}{1+0}=1$ 
に、$x < 0$ では $\mathrm{sigmoid}_\beta(x)\to \frac{1}{1+\infty}=0$ に、$x = 0$ では $\mathrm{sigmoid}_\beta(x) \to \frac{1}{1+1} = \frac{1}{2}$ に近づくため、これは $H(0) = \frac{1}{2}$ とするHeaviside関数に一致する。

同様に、$\beta \to \infty$ のときに符号関数 $\mathrm{sgn}(x)$ に収束する。すなわち、正の入力では 1、負の入力では -1、ゼロでは 0 に対応する。

なお、これらの逆温度付き関数は、関数の定義そのものを新たに与える必要はなく、元の関数の引数を単に $x \to \beta x$ と置換することで導入することができる。

##### ReLU関数・ソフトプラス関数
シグモイド関数やtanh関数は、神経細胞が強い入力に対して発火率を飽和させるという生理的特性を捉えているが、その出力範囲（dynamic range, ダイナミックレンジ）は限られており、極端な入力に対しては出力がほぼ一定となる。このとき出力の分布は0または1（あるいは $\pm 1$）付近に偏るため、エントロピー、すなわち出力の不確実性や多様性が低下する。情報理論的に見れば、出力のエントロピーが低いということは、活性化関数が伝達できる情報量が減少していることを意味し、結果として表現力が制限される。このように、飽和を起こす関数では入力に応じた情報の分解能が失われやすく、信号伝達の効率という観点から不利である。こうした問題を避けるため、非線形だが出力が飽和しない関数として、**ランプ関数** (ramp function) が用いられる \footnote{なお，機械学習にランプ関数（ReLU関数）が導入された理由はダイナミックレンジの問題だけでなく，勾配消失問題に対処できるという性質もあるが，これに関しては第4章で触れる．}．ランプ関数はHouseholder により神経細胞のモデルに導入され \citep{householder1941theory}，後に**ReLU関数** (rectified linear unit function, 正規化線形関数) と呼ばれるようになり \citep{nair2010rectified}、現在では後者の名称の方が広く一般に定着している。本書では、機械学習の事項も扱うため，統一的に「ReLU関数」という名称を用いる。ReLU関数 $\textrm{ReLU}(\cdot)$ は、入力が負のときに0を出力し、正のときはそのままの値を出力する関数であり，

$$
\begin{equation}
f(x) = \textrm{ReLU}(x):=\max(0, x)=
\begin{cases}
x & (x > 0) \\
0 & (x \leq 0)
\end{cases}
\end{equation}
$$

と表される．ただし，$\max(a, b)$ は, $a$ と $b$ のうち，大きい値を返す関数である．ReLU関数はHeaviside関数や符号関数などのように不連続点はないが，$x=0$ において微分不可能である．ReLU関数は非線形関数であるが，区間ごと ($x\geq 0$ および $x<0$) に見ると線形であるため，**区分線形関数** (piecewise linear function) に含まれる．

ReLU関数を滑らかに近似する関数がソフトプラス関数 (Softplus function) $\textrm{Softplus}(\cdot)$ であり，

$$
\begin{equation}
f(x) = \textrm{Softplus}(x):=\log(1+e^x)
\end{equation}
$$

と表される．シグモイド関数などと同様に逆温度で関数をスケーリングでき，逆温度付きソフトプラス関数は次のように表される：

$$
\begin{equation}
\textrm{Softplus}_\beta(x; \beta):=\frac{1}{\beta}\log(1+e^{\beta x})
\end{equation}
$$

$\beta \to \infty$ とする場合，$x>0$ に対しては $\mathrm{Softplus}_\beta(x)\to \frac{1}{\beta} \log(e^{\beta x}) = x$ となり，$x<0$ では $\mathrm{Softplus}_\beta(x)\to \frac{1}{\beta} \log(1+0) = 0$ となる．また，$x=0$ では $\mathrm{Softplus}_\beta(x)=\frac{1}{\beta}\log(1+1)\to 0$ に漸近するため，逆温度付きソフトプラス関数は $\beta \to \infty$ でReLU関数 $\max(0, x)$ に収束する。

##### Naka–Rushton関数
F-I曲線の形状としては，シグモイド関数のような飽和関数（saturated function）が用いられることが多いが，実際の神経細胞では多くの場合完全な飽和には至らず，部分的な飽和挙動を示す\footnote{ただし，シグモイド関数に渡す入力を適切にスケーリングすることで飽和を防ぐことは可能である．}．そのような特性を表現する関数として，以下のようなNaka–Rushton関数 $\textrm{NR}(\cdot)$ が用いられることもある\citep{naka1966s,sclar1990coding,wilson1999spikes}：

$$
\begin{equation}
f(x) = \textrm{NR}(x; a, s, m):=\frac{m\cdot x^a}{s^a + x^a} \cdot H(x)
=\begin{cases}
\frac{m\cdot x^a}{s^a + x^a} & (x > 0) \\
0 & (x \leq 0)
\end{cases}
\end{equation}
$$

ここで，$m\;(>0)$ は最大応答，$s\;(>0)$ は感度定数，$a\;(>0)$ は指数，$H(x)$ はHeaviside関数である．この関数は，入力に対して初期は急峻に応答し，その後徐々に応答が飽和する非線形性を持つ点で，生理的F-I曲線により近い特性を示す．