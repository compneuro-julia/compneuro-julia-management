## STDP則と競合学習
### STDP則
シナプスの可塑性は、シナプス前細胞と後細胞の発火タイミングの差に依存して変化することがある。このような可塑性の一種である**Spike-Timing-Dependent Plasticity**（STDP）は、1990年代後半に実験的に報告された（Markram et al., 1997; Bi and Poo, 1998）。その中でも最も基本的な形式は**Pair-based STDP則**と呼ばれ、シナプス前細胞と後細胞のスパイクのペアの時刻差に基づいて、シナプス強度の増強（LTP）または抑制（LTD）が引き起こされる。

https://www.sciencedirect.com/science/article/pii/S0896627312007039

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