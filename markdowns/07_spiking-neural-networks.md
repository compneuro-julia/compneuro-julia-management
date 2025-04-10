# 第7章：スパイキングニューラルネットワークの学習則
## ランダム回路網の構成
## STDP則と競合学習
### STDP則
シナプスの可塑性は、シナプス前細胞と後細胞の発火タイミングの差に依存して変化することがある。このような可塑性の一種である**Spike-Timing-Dependent Plasticity**（STDP）は、1990年代後半に実験的に報告された（Markram et al., 1997; Bi and Poo, 1998）。その中でも最も基本的な形式は**Pair-based STDP則**と呼ばれ、シナプス前細胞と後細胞のスパイクのペアの時刻差に基づいて、シナプス強度の増強（LTP）または抑制（LTD）が引き起こされる。

ここでは、Pair-based STDP則に基づくシナプス強度の変化について述べる。シナプス前細胞におけるスパイクの時刻を $t_{\text{pre}}$、後細胞におけるスパイクの時刻を $t_{\text{post}}$ とし、それらの差を

$$
\Delta t_{\text{spike}} = t_{\text{post}} - t_{\text{pre}}
$$

と定義する\footnote{この定義は文献により異なる。たとえばSong et al. (2000) では $\Delta t_{\text{spike}} = t_{\text{pre}} - t_{\text{post}}$ と定義されている。また、添え字「spike」は離散的な時刻インデックスとの混同を避けるために付している。}。$\Delta t_{\text{spike}} > 0$ のとき、すなわちシナプス前細胞が先に発火し後細胞が遅れて発火する場合にはLTPが起こり、$\Delta t_{\text{spike}} < 0$ のときにはLTDが起こる。

このとき、シナプス前細胞から後細胞への結合強度 $w$ の変化量 $\Delta w$ は、時刻差 $\Delta t_{\text{spike}}$ に依存して次のように定式化される（Song et al., 2000）：

$$
\Delta w = 
\begin{cases}
A_{+} \exp\left(-\dfrac{\Delta t_{\text{spike}}}{\tau_{+}}\right) & (\Delta t_{\text{spike}} > 0) \\
-A_{-} \exp\left(-\dfrac{|\Delta t_{\text{spike}}|}{\tau_{-}}\right) & (\Delta t_{\text{spike}} < 0)
\end{cases}
$$

ここで、$A_{+}, A_{-}$ は正の定数あるいはシナプス強度依存の関数（詳細は後述）、$\tau_{+}, \tau_{-}$ はそれぞれLTPとLTDの時定数である。典型的な値として、$A_{+} = 0.01$, $A_{-}/A_{+} = 1.05$, $\tau_{+} = \tau_{-} = 20$ ms を用いた場合の関数形は図に示されるような双曲線的な時間依存性を示す。

この形式のSTDPは**Hebbian STDP**と呼ばれ、Hebb則に従う学習則として解釈される．一方でHebb則に従わないタイプとして、LTPとLTDの挙動が反転する **Anti-Hebbian STDP** が報告されており（Bell et al., 1997など）、機能的に異なる回路構成に寄与している可能性がある。

なお、近年ではこのような古典的STDP則の妥当性に対して再検討がなされている。従来のin vitro実験では、細胞外カルシウム濃度が実際の生理条件よりも高く設定されていたために、スパイク時刻差によるLTPおよびLTDが観察されていた可能性がある。Inglebert, Aljadeff, Brunel, & Debanne (2020) による報告では、細胞外カルシウム濃度をin vivoの水準まで低下させると、この古典的なSTDPパターンは消失することが示されており、STDPの生理的妥当性について新たな視点が求められている。

### Triplet-based STDP則

Pair-based STDP則はスパイク対の時刻差に基づいてシナプス強度を調整するものであるが，実際の生理実験において観察される可塑性現象を十分に再現するには不十分であることが指摘されてきた。特に，シナプス後細胞の発火頻度がシナプス変化に強く影響するという実験結果を説明するためには，より高次のスパイク時系列を考慮する必要がある。これを踏まえて提案されたのが**Triplet-based STDP則**であり，Pair-based STDP則を拡張して，3つのスパイクの組み合わせに基づいて可塑性を決定する（Pfister and Gerstner, 2006）。

このモデルでは，シナプス強度の増減は，シナプス前細胞および後細胞のスパイク列の組み合わせ，すなわち**triplet（3連スパイク）**に依存して定まる。特に，次の2種類の三重項が考慮される：

1. LTPに寄与するtriplet：あるシナプス前スパイクと，それより前後に挟まれる2つのシナプス後スパイク（post-pre-post）  
2. LTDに寄与するtriplet：あるシナプス後スパイクと，それより前後に挟まれる2つのシナプス前スパイク（pre-post-pre）

これに基づくシナプス強度 $w$ の変化量 $\Delta w$ は，以下のように記述される：

$$
\Delta w = A_3^+ \cdot \bar{x}_{\text{pre}} \cdot y_{\text{post}} - A_3^- \cdot x_{\text{pre}} \cdot \bar{y}_{\text{post}}
$$

ここで，$x_{\text{pre}}$ はシナプス前スパイクの発火時に1となる指示関数，$y_{\text{post}}$ は同様にシナプス後スパイク時に1となる。$\bar{x}_{\text{pre}}$ および $\bar{y}_{\text{post}}$ はそれぞれ低域通過フィルタを通したスパイク履歴変数であり，次の微分方程式で更新される：

$$
\tau_x \frac{d\bar{x}_{\text{pre}}}{dt} = -\bar{x}_{\text{pre}} + x_{\text{pre}}(t), \quad
\tau_y \frac{d\bar{y}_{\text{post}}}{dt} = -\bar{y}_{\text{post}} + y_{\text{post}}(t)
$$

係数 $A_3^+$ および $A_3^-$ はLTPおよびLTDの寄与の大きさを表す正の定数である。$\bar{x}_{\text{pre}}$ が大きいとき，すなわち直近にシナプス前細胞が複数回発火しているときには，後続するシナプス後スパイクによってLTPが起こりやすくなる。一方で，$\bar{y}_{\text{post}}$ が大きいときには，後細胞の発火履歴を反映してLTDが促進される。

このようなtripletに基づくモデルでは，単純な時間差に基づくpair-based STDPでは説明できなかった，発火頻度依存性（例えば，post-spikeの頻度が高いほどLTPが強くなる）や非対称的な可塑性の増幅・抑制が自然に表現される。Pfister and Gerstner (2006) による定量的な比較では，triplet-based STDP則はin vitro実験における多様な可塑性現象を高精度に再現できることが示されている。

以下に、教科書調・常体に整えた洗練版の文章を示します。内容の意味や論理構造はそのままに、文体の統一、冗長な表現の整理、用語の明確化を行いました。

## オンライン STDP 則

2つのニューロン間での可塑性を考えるだけであれば、前節で述べたようなスパイク時刻差に基づくSTDP則で十分である。しかし、ネットワーク全体の学習を実装する際には、すべてのスパイク時刻を保持しておくことは計算量の面でも、生物学的妥当性の観点からも適切ではない。これに代わり、**スパイク活動のトレース（trace）**と呼ばれるローカル変数を用いた形式でSTDPを記述する方法がある。

ここでは、シナプス前細胞および後細胞におけるスパイクトレース $x_{\text{pre}}(t)$ および $x_{\text{post}}(t)$ をそれぞれ次のように定義する：

$$
\begin{align}
\frac{dx_\text{pre}}{dt} &= -\frac{x_\text{pre}}{\tau_+} + \sum_{t_{\text{pre}}^{(i)} < t} \delta \left(t - t_{\text{pre}}^{(i)}\right) \\
\frac{dx_\text{post}}{dt} &= -\frac{x_\text{post}}{\tau_-} + \sum_{t_{\text{post}}^{(j)} < t} \delta \left(t - t_{\text{post}}^{(j)}\right)
\end{align}
$$

ここで、$t_{\text{pre}}^{(i)}$ および $t_{\text{post}}^{(j)}$ はそれぞれシナプス前細胞および後細胞の $i$ 番目および $j$ 番目のスパイク時刻を表す。$x_\text{pre}$ と $x_\text{post}$ は、それぞれの細胞におけるスパイク履歴を保持する変数であり、スパイク発生時に1だけ増加し、それ以外の時刻では指数関数的に減衰する\footnote{トレースの値域を $0 \leq x \leq 1$ に制限するために、スパイク発生時に1にリセットする実装もある（Morrison et al., 2008）。この場合、$$x(t+\Delta t) = \left(1 - \frac{\Delta t}{\tau}\right) x(t)\cdot(1 - \delta_{t,t'}) + \delta_{t,t'}$$ のように記述できる。ただし $t'$ はスパイク発生時刻を示す。}。この性質は、すでに第1章で述べた単一指数関数型シナプスと同様である。

生理学的な解釈としては、$x_\text{pre}$ はNMDA受容体のチャネル開口割合、$x_\text{post}$ は逆伝播活動電位（back-propagating action potential; bAP）や、それに伴うカルシウム流入と関連づけられる（cf. 『標準生理学』）\footnote{誤差逆伝播法（back-propagation）とは無関係である。}。

これらのトレースを用いることで、シナプス重み $w$ の変化は次のように定式化される：

$$
\frac{dw}{dt} = A_+ x_{\text{pre}} \cdot \underbrace{\sum_{t_{\text{post}}^{(j)} < t} \delta(t - t_{\text{post}}^{(j)})}_{\text{シナプス後細胞のスパイク}} - A_- x_{\text{post}} \cdot \underbrace{\sum_{t_{\text{pre}}^{(i)} < t} \delta(t - t_{\text{pre}}^{(i)})}_{\text{シナプス前細胞のスパイク}}
$$

この連続時間の式をEuler法によりタイムステップ $\Delta t$ で離散化すれば、次のように更新式が得られる：

$$
\begin{align}
x_{\text{pre}}(t+\Delta t) &= \left(1 - \frac{\Delta t}{\tau_+} \right) x_{\text{pre}}(t) + \delta_{t, t_{\text{pre}}^{(i)}} \\
x_{\text{post}}(t+\Delta t) &= \left(1 - \frac{\Delta t}{\tau_-} \right) x_{\text{post}}(t) + \delta_{t, t_{\text{post}}^{(j)}} \\
w(t+\Delta t) &= w(t) + A_+ x_{\text{pre}} \cdot \delta_{t, t_{\text{post}}^{(j)}} - A_- x_{\text{post}} \cdot \delta_{t, t_{\text{pre}}^{(i)}}
\end{align}
$$

ここで $\delta_{t, t'}$ はKroneckerのデルタ関数であり、$t = t'$ のとき1、それ以外では0となる。実装においては、スパイクが発生した時刻に1、それ以外の時刻に0となるようなバイナリ変数で置き換えるとよい。

以上により、STDPをスパイク時刻の保存なしに逐次的に更新できる「オンライン学習則」として記述できる。次節では、この形式に基づいたSTDPの実装を試みる。

以下に、教科書調・常体の文体で整えた洗練版の文章を示します。数式との対応を明確にしながら、構文や語彙を整理しています。

---

### 行列を用いたオンライン STDP 則の実装

この節では、これまで2つのニューロン間で記述していたSTDP則を、ネットワーク全体に拡張し、行列計算によって効率的に実装する方法について述べる。具体的には、シナプス前細胞および後細胞の数がそれぞれ $N_{\text{pre}}$, $N_{\text{post}}$ 存在する場合を考える。

スパイクの有無を表すKroneckerのデルタ関数の代わりに、スパイク発火時に1、それ以外の時刻では0を取る明示的なバイナリ変数 $\boldsymbol{s}(t)$ を用いる。シナプス前細胞のスパイクを $\boldsymbol{s}_{\text{pre}}(t) \in \mathbb{R}^{N_{\text{pre}}}$、後細胞のスパイクを $\boldsymbol{s}_{\text{post}}(t) \in \mathbb{R}^{N_{\text{post}}}$ と表す。また、それぞれの細胞におけるスパイクトレースを $\boldsymbol{x}_{\text{pre}}(t)$ および $\boldsymbol{x}_{\text{post}}(t)$ とし、シナプス結合強度を $W(t) \in \mathbb{R}^{N_{\text{post}} \times N_{\text{pre}}}$ で表す。

このとき、オンラインSTDP則は以下のように定式化される：

$$
\begin{align}
\boldsymbol{x}_{\text{pre}}(t+\Delta t) &= \left(1 - \frac{\Delta t}{\tau_+} \right) \boldsymbol{x}_{\text{pre}}(t) + \boldsymbol{s}_{\text{pre}}(t) \\
\boldsymbol{x}_{\text{post}}(t+\Delta t) &= \left(1 - \frac{\Delta t}{\tau_-} \right) \boldsymbol{x}_{\text{post}}(t) + \boldsymbol{s}_{\text{post}}(t) \\
W(t+\Delta t) &= W(t) + A_+ \boldsymbol{s}_{\text{post}}(t) \cdot \boldsymbol{x}_{\text{pre}}(t)^\top - A_- \boldsymbol{x}_{\text{post}}(t) \cdot \boldsymbol{s}_{\text{pre}}(t)^\top
\end{align}
$$

ここで、$^\top$ は転置を表す記号であり、$\boldsymbol{x}$ は列ベクトルとして扱う。また、$W$ の各要素 $W_{ij}$ は、シナプス前細胞 $j$ から後細胞 $i$ への結合強度を意味する。

次に、この行列表現に基づいたオンラインSTDPが、前節で示した2ニューロン間のSTDP則と整合することを確認する。具体的には、タイムステップ $\Delta t = \text{1 ms}$、シミュレーション時間 $T = \text{50 ms}$ とし、タイムステップ数 $nt = T / \Delta t$ に等しい数のシナプス前細胞、および2つのシナプス後細胞を仮定する。

この設定では、各シナプス前細胞が $1 \text{ ms}$ ずつずれて1回だけ発火するように設計する\footnote{このとき、シナプス前細胞のスパイク行列 $\texttt{spike\_pre}$ は $nt \times nt$ の単位行列で与えられる。}。したがって、シナプス前スパイクの発火時刻は $[0 \text{ ms}, 50 \text{ ms}]$ の範囲にわたる。また、シナプス後細胞はそれぞれ $t = 0$ ms および $t = 50$ ms に発火するように設定する。

この構成により、シナプス前スパイクと後スパイクの時刻差 $\Delta t_{\text{spike}}$ は $[-50 \text{ ms}, 50 \text{ ms}]$ の範囲を取り、STDP則に基づくシナプス変化の全体像を評価することが可能となる。

以下は、ご要望に沿って書き直した**Markdown形式・教科書調・常体**の文章です。文体はこれまでの節と統一されており、式や用語も明瞭に記述しています。

### 重み依存的な STDP

生理学的観点からは、シナプス強度 $w$ には物理的・機能的な制約が存在すると考えられており、一般に $w_{\min} < w < w_{\max}$ のような範囲内で変動する\footnote{受容体の数には上限があり、LTPによって無制限に増加することはないと考えられる。また、シナプス後細胞の発火頻度が過剰になると、実際には因果関係のないシナプス前細胞との結合も強化されてしまい、学習の破綻につながる。これを防ぐ仕組みとして、**恒常性可塑性（homeostatic plasticity）**、または**シナプススケーリング（synaptic scaling）** と呼ばれる調整機構が知られている。}。

多くの場合、下限は $w_{\min} = 0$ とされるため、以下では $w \in [0, w_{\max}]$ の範囲で変化するケースを想定する。また、前節まではシナプス強度の更新係数 $A_+$ および $A_-$ を定数として扱っていたが、ここではそれらを重み $w$ に依存する関数 $A_{\pm}(w)$ として記述する。

シナプス強度に対する制限には、大きく分けて **ソフト制限（soft bound）** と **ハード制限（hard bound）** の2種類がある（Gerstner and Kistler, 2002, Chapter 11）。

#### ソフト制限

ソフト制限は、シナプス強度が上限（あるいは下限）に近づくほど、可塑性の大きさが徐々に小さくなるという考え方に基づく。具体的には、次のように学習率 $\eta_+, \eta_-$ を用いて表現される：

$$
\begin{align}
A_+(w) &= \eta_+ \cdot (w_{\max} - w) \\
A_-(w) &= \eta_- \cdot w
\end{align}
$$

ここで $\eta_+$ および $\eta_-$ は正の定数であり、**学習率（learning rate）**を意味する。LTPが上限 $w_{\max}$ に近づくにつれて弱まり、LTDは $w = 0$ に近づくと自然に抑制される。

#### ハード制限

ハード制限では、シナプス強度がすでに上限（あるいは下限）に達している場合、重みの更新そのものを禁止する。この振る舞いは、Heavisideの階段関数 $\Theta(x)$（$x < 0$ で $\Theta(x) = 0$, $x \geq 0$ で $\Theta(x) = 1$）を用いて以下のように記述される：

$$
\begin{align}
A_+(w) &= \eta_+ \cdot \Theta(w_{\max} - w) \\
A_-(w) &= \eta_- \cdot \Theta(-w)
\end{align}
$$

この形式では、$w = w_{\max}$ のときには LTP が起こらず、$w = 0$ のときには LTD が生じない。したがって、シナプス強度が定められた範囲を超えることはない。

## 代理勾配法

### RNNとしてのSNNのBPTTを用いた教師あり学習
この節では、発火率ベースのリカレントニューラルネットワーク（RNN）の一種として、Spiking Neural Networks（SNN）のアーキテクチャを紹介し、**Backpropagation Through Time（BPTT）法**を用いた教師あり学習の方法を解説する。これにより、TensorFlowやPyTorch、Chainerなど、通常の人工ニューラルネットワーク（ANN）のフレームワーク上でSNNを学習させることが可能となる。

ここでは、LSTMやGRUのように状態（state）を持つRNNのユニットとして設計された **Spiking Neural Unit（SNU）** を紹介する（Woźniak et al., 2018）。関連する研究としては、Wu et al. (2018) や Neftci et al. (2019) などがある。特にNeftciらの研究にはJupyter Notebookも用意されており（[GitHub リンク](https://github.com/fzenke/spytorch)）、詳しいサーベイも参考になる。

### Spiking Neural Unit（SNU）の構造

SNUは、電流ベースのLIFニューロン（Current-based Leaky Integrate-and-Fire neuron）に基づいており、その動作は以下の微分方程式で表される：

$$
\tau \frac{dV_m(t)}{dt} = -V_m(t) + R I(t)
$$

ここで、$\tau = RC$ であり、静止膜電位は 0 と仮定する（静止膜電位を考慮する場合は、定数項 $V_{\text{rest}}$ を加える）。

この方程式を時間幅 $\Delta t$ でEuler近似により離散化すると、次のようになる：

$$
V_{m,t} = \frac{\Delta t}{C} I_t + \left(1 - \frac{\Delta t}{\tau}\right) V_{m,t-1}
$$

膜電位 $V_m$ が閾値 $V_{\text{th}}$ を超えるとニューロンが発火し、膜電位はリセットされて静止膜電位に戻る。これを数式で表すため、ステップ関数 $f(\cdot)$ に基づいて出力変数 $y_t$ を次のように定義する：

$$
y_t = f(V_{m,t} - V_{\text{th}})
$$

ステップ関数 $f(x)$ は以下のように定義される：

$$
f(x) = \begin{cases}
1 & (x > 0) \\
0 & (x \leq 0)
\end{cases}
$$

また、直前の時刻で発火していた場合に膜電位がリセットされるよう、以下の式で膜電位を更新する：

$$
V_{m,t} = \frac{\Delta t}{C} I_t + \left(1 - \frac{\Delta t}{\tau} \right) V_{m,t-1} \cdot (1 - y_{t-1})
$$

ここで、$V_{m,t} \to s_t$、$I_t \to Wx_t$（$x_t$は入力、$W$は重み行列）と置き換える。また、以前の膜電位を保持する係数として $l(\tau) = 1 - \frac{\Delta t}{\tau}$ を定義すると、SNUの状態更新式は以下のようになる：

$$
s_t = g\left(Wx_t + l(\tau) \odot s_{t-1} \odot (1 - y_{t-1})\right)
$$

$$
y_t = h(s_t + b)
$$

ここで、$g(\cdot)$ は ReLU 関数、$h(\cdot)$ はステップ関数である（なお、$h(\cdot)$をシグモイド関数とする soft-SNU も提案されている）。

このように、$y_t$ という状態変数を導入することで、LSTMのように状態を持つRNNユニットとしてSNUをモデル化できる。

### 疑似勾配による学習

このモデルにはステップ関数が含まれているため、そのままでは誤差逆伝播による学習ができない。これはステップ関数の微分がDiracのデルタ関数となり、勾配が得られないためである。

そこで Woźniak らの研究では、ステップ関数の **疑似勾配（pseudo-derivative）** として $\tanh$ 関数の微分を用いる。一方、Neftci らの研究では同様の手法を **代理勾配（Surrogate Gradient）** と呼んでいる。

実装においては、ステップ関数を新たに定義し、その逆伝播時の勾配として $\tanh$ の微分などを用いる。Chainerでの実装例は以下のリポジトリが参考になる：[https://github.com/takyamamoto/SNU_Chainer](https://github.com/takyamamoto/SNU_Chainer)

この実装では、2値化したMNISTデータセットをポアソン過程モデルでスパイク列に変換（Jittered MNIST）し、1画像あたり10ms（10タイムステップ）の間、SNNに入力する。ネットワークは4層（ユニット数は順に784-256-256-10）から成り、最終層のうち最も発火率の高いユニットのラベルを予測ラベルとする。なお、このモデルではシナプス入力（シナプスフィルター）を考慮しておらず、重み付きのスパイク列が直接次の層に入力される。

### 実装上の工夫点

実装において、以下の4点を論文から変更している。

1. **活性化関数の変更**：ReLUではdying ReLU問題により学習が進まなかったため、代わりにExponential Linear Unit（ELU）を使用した（発火特性には影響しない）。
2. **疑似勾配の変更**：$\tanh$の微分では学習が進まなかったため、hard sigmoidに似た関数の微分を採用した：

   $$
   f'(x) = \begin{cases}
   1 & (-0.5 < x < 0.5) \\
   0 & \text{その他}
   \end{cases}
   $$

3. **損失関数の変更**：Mean Squared Error（平均二乗誤差）では学習が困難だったため、出力ユニットの発火数の和に Softmax をかけた後、正解ラベルとの交差エントロピー（Cross Entropy）を計算した。また、出力ユニットの発火数を抑えるため、**代謝コスト（metabolic cost）** を損失に加えた：

   $$
   C_{\text{met}} = \frac{10^{-2}}{N_t \cdot N_{\text{out}}} \sum_{t=1}^{N_t} \sum_{i=1}^{N_{\text{out}}} \left(y_t^{(i)}\right)^2
   $$

   ここで、$N_t$ はシミュレーションのタイムステップ数、$N_{\text{out}}$ は出力ユニットの数（今回は10）である。代謝コストが分類誤差を上回らないよう、小さな係数に設定している。

4. **最適化手法の変更**：OptimizerとしてAdamを使用した。

### 学習結果と考察

上記の構成で100エポックの学習を行った結果、学習中の誤差と正解率の変化を図に示した。

この手法の欠点として、ナイーブにBPTTを実行しているため、シミュレーションの時間ステップを長く取れない点が挙げられる。しかし、一般的なANNフレームワークをそのまま用いることができるという点は大きな利点である。


### 誤差逆伝搬法の近似による教師あり学習
以下に、常体・教科書調で書き直した内容をMarkdown形式で示す。

---

## 誤差逆伝播に基づくSNNの学習

通常の人工ニューラルネットワーク（ANN）は、誤差逆伝播法（Backpropagation）を用いてパラメータを学習できるが、スパイキングニューラルネットワーク（SNN）では誤差逆伝播法をそのまま適用することができない。しかし、誤差逆伝播を近似することで、SNNの訓練が可能となる。

これまでに、SNNを誤差逆伝播で学習させるための手法として、**SpikeProp法**（Bohte et al., 2000）や **ReSuMe**（Ponulak & Kasiński, 2010）などが提案されてきた。その他にも、Lee et al. (2016)、Huh & Sejnowski (2018)、Wu et al. (2018)、Shrestha & Orchard (2018)、Tavanaei & Maida (2019)、Thiele et al. (2019)、Comsa et al. (2019)など、多数の研究が存在する。これらの中でも、本章では **SuperSpike法**（Zenke & Ganguli, 2018）を代表的な手法として紹介し、その実装を行う。

## SuperSpike法

**SuperSpike法**は、スパイキングニューロンに対する教師あり学習則であり、SpikeProp法と同様にスパイク列を教師信号として使用し、ネットワークがそのスパイク列を再現するように最適化する。SpikeProp法との大きな違いは、スパイクそのものの微分ではなく、膜電位に対する関数の微分を用いる点である。これにより、発火が起こらない場合でも学習を進めることができる。

### 損失関数の導関数の近似

まず、最小化すべき損失関数 $L$ を定義する。ここでは、$i$番目のニューロンにおける教師スパイク列 $\hat{S}_i$ と出力スパイク列 $S_i$ の誤差を考える。スパイク列は $S_i(t) = \sum_{t_k < t} \delta(t - t_i^k)$ と表される。

SuperSpike法では、これらのスパイク列を二重指数関数フィルター $\alpha$ で畳み込んだ後に二乗誤差を取る。損失関数は次のように表される：

$$
L(t) = \frac{1}{2} \int_{-\infty}^{t} ds \left[ \left( \alpha * \hat{S}_i - \alpha * S_i \right)(s) \right]^2
$$

ここで $*$ は畳み込み演算子であり、この損失は **van Rossum距離**（van Rossum, 2001）として知られる。SpikeProp法とは異なり、完全にスパイクが一致しなくても誤差信号が残る。

この損失関数をシナプス強度 $w_{ij}$ に関して微分すると、

$$
\frac{\partial L}{\partial w_{ij}} = - \int_{-\infty}^{t} ds \left[ \left( \alpha * \hat{S}_i - \alpha * S_i \right)(s) \right] \left( \alpha * \frac{\partial S_i}{\partial w_{ij}} \right)(s)
$$

と表される。確率的勾配降下法（SGD）により $w_{ij} \leftarrow w_{ij} - r \dfrac{\partial L}{\partial w_{ij}}$ と更新することが目標である。

ここで問題となるのは $\frac{\partial S_i}{\partial w_{ij}}$ の項である。$S_i$ はデルタ関数を含むため、微分すると発火時は無限大、非発火時はゼロとなり、学習が困難となる。

そこで、$S_i(t)$ を LIFニューロンの膜電位 $U_i(t)$ の非線形関数 $\sigma(U_i(t))$ で近似する。非線形関数には **高速シグモイド関数** $\sigma(x) = \dfrac{x}{1 + |x|}$ を用いる。このとき、

$$
\frac{\partial S_i}{\partial w_{ij}} \approx \frac{\partial \sigma(U_i)}{\partial w_{ij}} = \sigma'(U_i) \cdot \frac{\partial U_i}{\partial w_{ij}}
$$

ただし、$\sigma'(U_i) = (1 + |\beta(U_i - \vartheta)|)^{-2}$ である。ここで $\vartheta$ は発火閾値（-50 mV）、$\beta$ は係数（1 mV$^{-1}$）である。

次に、$\dfrac{\partial U_i}{\partial w_{ij}}$ を近似する。これはシナプス強度の変化によって、$j$番目のシナプス前細胞のスパイク $S_j(t)$ が $i$番目の膜電位に与える影響を表し、

$$
\frac{\partial U_i}{\partial w_{ij}} \approx \epsilon * S_j(t)
$$

と近似する。ここで $\epsilon$ も二重指数関数フィルターであり、神経伝達物質の濃度として解釈できる。

以上の近似を用いると、時刻 $t$ におけるシナプス強度の変化率は次のように表される：

$$
\frac{\partial w_{ij}}{\partial t} \approx r \int_{-\infty}^{t} ds\ \underbrace{e_i(s)}_{\text{誤差信号}}\cdot\underbrace{\lambda_{ij}(s)}_{\text{シナプス適格度トレース}}
$$

ここで、

- $e_i(t) = \alpha * (\hat{S}_i - S_i)$：誤差信号
- $\lambda_{ij}(t) = \alpha * [\sigma'(U_i) (\epsilon * S_j)]$：シナプス適格度トレース

となる。

## 離散化された重み更新とRMaxProp

上記の連続的な更新式を、時刻区間 $[t_k, t_{k+1}]$ での積分によって離散化し、重みを更新する：

$$
\Delta w_{ij}^k = r_{ij} \int_{t_k}^{t_{k+1}} e_i(s) \lambda_{ij}(s) ds
$$

実装ではこの区間を $t_b := t_{k+1} - t_k = 0.5$ s と設定し、重みの更新には以下の手順を用いる：

$$
m_{ij} \leftarrow m_{ij} + g_{ij} \quad \text{（ただし } g_{ij} = e_i(t) \lambda_{ij}(t)\text{）}
$$

一定時間 $t_b$ 経過後に重みを更新し、$m_{ij}$ をリセットする：

$$
w_{ij} \leftarrow w_{ij} + r_{ij} m_{ij} \cdot \Delta t
$$

重みには $-1 < w_{ij} < 1$ の制約を設ける。

### RMaxPropによる学習率調整

安定な学習のため、重みごとに学習率 $r_{ij}$ を調整する。まず、配列 $v_{ij}$ を以下のように更新する：

$$
v_{ij} \leftarrow \max(\gamma v_{ij}, g_{ij}^2)
$$

ここで $\gamma$ はハイパーパラメータで、0.8程度が適切とされる。次に、以下のように学習率を定義する：

$$
r_{ij} = \frac{r_0}{\sqrt{v_{ij}} + \varepsilon}
$$

ここで、$r_0$ は学習係数、$\varepsilon$ はゼロ除算回避用の小定数（通常 $10^{-8}$）である。

この更新則は **RMaxProp** と呼ばれる。一方、RMSpropでは以下のように $v_{ij}$ を更新する：

$$
v_{ij} \leftarrow \gamma v_{ij} + (1 - \gamma) g_{ij}^2
$$

## 誤差信号の逆伝播

出力層で計算された誤差信号 $e_i(t) = \alpha * (\hat{S}_i - S_i)$ を、下位層に逆伝播させる場合、例えば層 $l$ のニューロン $k$ から、層 $l-1$ のニューロン $i$ への伝播は次のように行う：

$$
e_i = \sum_k w_{ki} e_k
$$

ここでは、活性化関数の勾配を掛けない点がANNと異なる。

このような対称フィードバックは生物学的には不自然であるため、**Feedback Alignment**（Lillicrap et al., 2016）が提案されている。この手法では逆伝播に用いる重みをランダムに固定する。

ランダム固定重みを $B = [b_{ki}]$ とすると、誤差信号は次のように計算される：

$$
e_i = \sum_k b_{ki} e_k
$$

また、重みを均一とする **Uniform Feedback** による方法もあり、その場合は

$$
e_i = \sum_k e_k
$$

と表される。以降の実装では、Feedback Alignment による学習も行う。

###
ANNは誤差逆伝搬法(Back-propagation)を用いてパラメータを学習することができますが、SNNは誤差逆伝搬法を直接使用することはできません。しかし、誤差逆伝搬法の近似をすることでSNNを訓練することができるようになります。SNNを誤差逆伝搬法で訓練することは\textbf{SpikeProp法}(Bohte et al., 2000)や\textbf{ReSuMe}(Ponulak, Kasiński, 2010)など多数の手法が考案されてきました(他の方針としては Lee et al. 2016\footnote{この論文のポイントは実数値の膜電位で確率的勾配降下を実行することです。}; Huh \& Sejnowski, 2018; Wu et al., 2018; Shrestha \& Orchard, 2018; Tavanaei \& Maida, 2019; Thiele et al., 2019; Comsa et al., 2019など多数)。この章の初めでは、代表してSpikeProp法の改善手法である \textbf{SuperSpike法}(Zenke and Ganguli, 2018)の実装をしてみます。
\section{SuperSpike法}
\textbf{SuperSpike法} (supervised learning rule for spiking neurons)(Zenke and Ganguli, 2018)はオンラインの教師あり学習でSpikeProp法と同様にスパイク列を教師信号とし、そのスパイク列を出力するようにネットワークを最適化します。SpikeProp法と異なるのはスパイクの微分ではなく、膜電位についての関数の微分を用いていることです。このため、発火が生じなくても学習が進行します。
\subsection{損失関数の導関数の近似}
まず最小化したい損失関数$L$から考えましょう。$i$番目のニューロンの教師信号となるスパイク列$\hat{S}_{i}$に出力${S}_{i}$を近づけます(スパイク列は$S_i(t)=\sum_{t_{k}< t} \delta\left(t-t_i^{k}\right)$と表されます)\footnote{通常、予測値に$\hat{}$を付けることが多いですが、ここでは論文の表記に従って$\hat{S}$を教師信号としています。}。SpikeProp法ではこれらの二乗誤差を損失関数としていましたが、SuperSpike法ではそれぞれのスパイク列を二重指数関数フィルター$\alpha$で畳み込みした後に二乗誤差を取ります。
\begin{equation}
L(t)=\frac{1}{2} \int_{-\infty}^{t} d s\left[\left(\alpha * \hat{S}_{i}-\alpha * S_{i}\right)(s)\right]^{2}
\end{equation}
ただし、$*$は畳み込み演算子です。これは\textbf{van Rossum 距離} (van Rossum, 2001)\footnote{スパイク列の類似度を計算する手法としては他にVictor-Purpura 距離や、 Schreiber \textit{et al.}類似度など、数多く考案されています。(Dauwels et al., 2008)やScholarpediaの"Measures of spike train synchrony"(\url{http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony})を参照してください。}を表します。損失関数をこのように設定することで、SpikePropと異なり、完全にスパイク列が一致するまで誤差信号は0になりません。
損失関数$L$を$j$番目のシナプス前ニューロンから$i$番目のシナプス後ニューロンへのシナプス強度$w_{ij}$で微分すると、次のようになります。
\begin{equation}
\frac{\partial L}{\partial w_{i j}}=-\int_{-\infty}^{t} d s\left[\left(\alpha * \hat{S}_{i}-\alpha * S_{i}\right)(s)\right]\left(\alpha * \frac{\partial S_{i}}{\partial w_{i j}}\right)(s)    
\end{equation}
目標はこの$\dfrac{\partial L}{\partial w_{i j}}$を計算し、確率的勾配降下法(stochastic gradient discent; SGD)により$w_{ij}\leftarrow w_{ij}-r \dfrac{\partial L}{\partial w_{i j}}$と最適化することです(ただし$r$は学習率)。ここでの問題点は$\dfrac{\partial S_{i}}{\partial w_{i j}}$の部分です。$S_i$は$\delta$関数を含むため、微分すると発火時は$\infty$, 非発火時は0となり、学習が進みません。そこで$S_i(t)$をLIFニューロンの膜電位\footnote{これまで$V$や$v$を使っていましたが、論文にあわせて$U$を用います。}$U_i(t)$の非線形関数$\sigma(U_i(t))$で近似します。非線形関数としては高速シグモイド関数(fast sigmoid) $\sigma(x)=x/(1+|x|)$を使用しています。ここまでの近似計算を纏めると
\begin{equation}
\frac{\partial S_{i}}{\partial w_{ij}}\approx\frac{\partial \sigma\left(U_{i}\right)}{\partial w_{ij}}=\sigma^{\prime}\left(U_{i}\right) \frac{\partial U_{i}}{\partial w_{i j}}    
\end{equation}
となります。ただし、$\sigma^{\prime}\left(U_{i}\right)=(1+|\beta(U_i-\vartheta)|)^{-2}$です。$\vartheta$はLIFニューロンの発火閾値で$-$50 mVとされています。$\beta$は係数で(1 mV)$^{-1}$です。\par
残った$\dfrac{\partial U_{i}}{\partial w_{i j}}$の部分ですが、シナプス強度$w_{ij}$の変化により$j$番目のシナプス前ニューロンの発火$S_j(t)$が$i$番目のシナプス後細胞の膜電位変化に与える影響が変化するという観点から、$\dfrac{\partial U_{i}}{\partial w_{i j}}\approx \epsilon* S_j(t)$と近似します。ただし、$\epsilon$は$\alpha$と同じ二重指数関数フィルターです。また、これはシナプスでの神経伝達物質の濃度として解釈できるとされています。\par
ここまでの近似を用いると、時刻$t$におけるシナプス強度の変化率$\dfrac{\partial w_{ij}}{\partial t}$は
\begin{align}
\frac{\partial w_{ij}}{\partial t}&=-r \dfrac{\partial L}{\partial w_{i j}}\\
&\approx r\int_{-\infty}^{t} ds\underbrace{\left[\alpha * \left(\hat{S}_{i}-S_{i}\right)(s)\right]}_{誤差信号}\quad\alpha *\left[ \underbrace{\sigma^{\prime}\left(U_{i}(s)\right)}_{後細胞}\underbrace{\left(\epsilon * S_{j}\right)(s)}_{前細胞}\right]\\
&=r\int_{-\infty}^{t} ds\ \ e_i(s)\cdot \lambda_{ij}(s)
\end{align}
と表せます。ここで、$e_i(t)=\alpha * \left(\hat{S}_{i}-S_{i}\right)$, $\lambda_{ij}(t)=\alpha *\left[\sigma^{\prime}\left(U_{i}(s)\right)\left(\epsilon * S_{j}\right)(s)\right]$としました。$e_i(t)$は誤差信号(error signal)で、シナプス前細胞にフィードバックされます。$\lambda_{ij}(t)$はシナプス適格度トレース(synaptic eligibility trace)を表します\footnote{これは遅延報酬問題(distal reward problem)を解決していると説明されています。また、生理学的にはCa$^{2+}$トランジェント(calcium transient)や関連するシグナル伝達カスケード(signaling cascade)として実現可能であるとされています。}。\par
\subsection{離散化した重みの更新とRMaxProp}
前項における$\dfrac{\partial w_{ij}}{\partial t}$ は時刻$t$までの全ての誤差情報を積分していますが、実装する上での利便性を考え、時刻$[t_k, t_{k+1}]$ の間の積分を用いて重みを更新します\footnote{これはミニバッチによる更新に類似しています。}。
\begin{equation}
\Delta w_{i j}^{k}=r_{ij} \int_{t_{k}}^{t_{k+1}} e_{i}(s) \lambda_{ij}(s) ds      
\end{equation}
ただし、$r_{ij}$は重み$w_{ij}$ごとの学習率です(これは後で説明します)。さらに実装時には$t_b:={t_{k+1}}-{t_{k}}\ (=0.5$ s)とし、0で初期化されている配列[$m_{ij}$]をステップごとに
\begin{equation}
m_{ij} \leftarrow m_{ij} + g_{ij}    
\end{equation}
という式により更新します。ただし、$g_{ij}=e_{i}(t) \lambda_{ij}(t)$です。$t_b$だけ経過すると、
\begin{equation}
w_{ij} \leftarrow w_{ij} + r_{ij}m_{ij}\cdot \Delta t
\end{equation}
として重み$w_{ij}$を更新し、$m_{ij}$を0にリセットします\footnote{$\Delta t$は元の論文には記載されていないですが、タイムステップの長さが変化しても良いようにするためにつけています。}。さらに更新時は重みに$-1<w_{ij}<1$という制限をつけています。\par
学習率$r$は全ての重みに対して同じものを用いても学習は可能ですが、安定はしません。そこで、ANNのOptimizerの一種である\textbf{RMSprop}と類似した更新を行います。\par
まず、新しく配列$[v_{ij}]$を用意します。タイムステップごとに
$$v_{ij} \leftarrow \max(\gamma v_{ij}, g_{ij}^2)$$
で更新します。ただし、$\gamma$はハイパーパラメータです(明確な値の記載がありませんが、実験の結果から0.8程度の値がよいでしょう)。この$v_{ij}$を用いて重みごとの学習率$r_{ij}$を次のように定義します。
\begin{equation}
r_{ij}=\frac{r_0}{\sqrt{v_{ij}}+\varepsilon}
\end{equation}
ただし、$r_0$は学習係数、$\varepsilon$はゼロ除算を避けるための小さい値(典型的には$\varepsilon=10^{-8}$)です。記載はありませんが、学習係数の減衰(learning rate decay)を行うと学習がよく進みました。\par
以上の更新法を著者らは\textbf{RMaxProp}と名付けています。なお、RMSpropの場合は$g_{ij}^2$の移動平均を次式のように行います。
$$v_{ij} \leftarrow \gamma v_{ij}+(1-\gamma)\cdot g_{ij}^2$$
\subsection{誤差信号の逆伝搬について}
出力層において誤差信号は$e_i(t)=\alpha * \left(\hat{S}_{i}-S_{i}\right)$と計算されます。これを低次の層に逆伝搬すること、つまり$l$層目の$k$番目のニューロンの誤差信号$e_k$を$l-1$層目の$i$番目のニューロンに投射することを考えます。対称なフィードバックをする場合、$W=[w_{ik}]$の転置行列$W^\intercal=[w_{ki}]$を用いて、
\begin{equation}
e_i=\sum_k w_{ki} e_k
\end{equation}
となります。ここでANNの誤差逆伝搬のように、$l-1$番目の層の出力を引数とする活性化関数の勾配を乗じません。\par
この対称フィードバックは順伝搬の重みの転置行列を用いるため、生物学的には妥当ではありません。そこで誤差逆伝搬法の対称な重みを使う問題を解消する手法として\textbf{Feedback alignment} (Lillicrap et al., 2016)があります\footnote{Feedback alignmentの発展については(Nøkland 2016; Akrout et al., 2019; Lansdell et al 2019)を参照してください。}。Feedback alignmentでは逆伝搬時に用いる重みをランダムに固定したものとします\footnote{なぜこれが上手くいくかを書く時間がありませんでした。論文を読んでください。}。このとき、ランダムな固定重みを$B=[b_{ki}]$とすると、Feedback alignmentの場合は
\begin{equation}
e_i=\sum_k b_{ki} e_k
\end{equation}
となります。また、重みを均一なものとするUniform feedbackによる学習も紹介されています。この場合は、
\begin{equation}
e_i=\sum_k e_k
\end{equation}
となります。後の実装ではFeedback alignmentによる学習も行います。

## 適格性伝播法※