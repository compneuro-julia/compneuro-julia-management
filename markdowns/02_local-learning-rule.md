# 第2章：発火率モデルと局所学習則
## 神経細胞の概要
#### 脳における細胞の種類と数
脳は膨大な数の細胞によって構成されており、主に**神経細胞（ニューロン）**と**グリア細胞**の二種類に分類される。ヒトの脳には約860億個の神経細胞が存在し[引用]、

大脳
小脳

グリア細胞も同数またはそれ以上の数が含まれるとされる。これらの細胞は非常に高密度に詰まっており、1 mm³あたりでは平均して5万〜10万個の神経細胞が存在する[引用]。

文献確認

グリア細胞は、神経細胞の代謝支援、修復、髄鞘形成、シナプスの調節など、神経系の正常な機能を維持するために不可欠な働きを担っている。

ミクログリア，アストロサイト，オリゴデンドロサイト，シュワン細胞

#### 神経細胞の形態
神経細胞の形態は他の細胞と大きく異なり、**細胞体** (soma, cell body)，**樹状突起**(dendrite)、**軸索** (axon) という三つの主要構造からなる。細胞体には細胞核があり、タンパク質合成やエネルギー代謝など基本的な細胞機能が行われる。樹状突起は木の枝のように複雑に分岐した構造で、他の神経細胞からの入力（シナプス入力）を受け取る部位である。軸索は通常1本の細長い突起であり、細胞体で統合された情報を他の神経細胞へと送る電気信号を伝導する。軸索の起始部には、細胞体との接合部である**軸索小丘**（あるいは軸索起始円錐; axon hillock）が存在し、それより遠位の領域は**軸索初節**（axon initial segment, AIS） と呼ばれる。AISは、電気信号の発生、すなわち**活動電位**（詳細は後述）の出発点として極めて重要な役割を果たす[引用]。この部位には電位依存性ナトリウムチャネルが高密度に存在し、膜電位が閾値を超えると活動電位がここで生成される。AISは、ニューロンが入力を受けて出力すべきかを判断する**意思決定点**とも呼ばれる。

シナプス前終末

トリガー帯 (trigger zone)

https://www.pnas.org/doi/10.1073/pnas.1215125110
https://www.pnas.org/doi/10.1073/pnas.1720493115

#### 神経細胞の電気的活動
神経細胞は、主に**電気的活動**によって情報を処理・伝達する。この活動は、細胞膜を挟んだ**イオンの移動**に基づいており、特に**イオンチャネル**と**イオントランスポータ**の働きが重要である。神経細胞の膜は静止時に内側が負に帯電しており、この状態は主に**ナトリウム・カリウムポンプ**（Na⁺/K⁺ ATPase）によって維持される。外部からの入力によって膜電位が上昇し、ある**閾値**（ただし一定ではない）を超えると、AISに存在する電位依存性ナトリウムチャネルが開き、ナトリウムイオンの流入によって膜が急激に脱分極する。この過程で生じる電位変化が**活動電位** (action potential) あるいは **スパイク** (spike) と呼ばれる信号であり、軸索を伝導して末端まで到達する。活動電位が発生することを**発火** (firing) とも呼ぶ．スパイクの後には、一時的に再発火が困難となる**不応期**（refractory period）が存在し、これにより信号の一方向性が保たれ、連続的なスパイクの発生頻度が制御される。

スパイクは最終的に**シナプス**（synapse）に到達し、次の細胞に情報を伝える。この伝達には大きく分けて二種類のシナプスがある。**化学シナプス**では、スパイクの到達により**シナプス小胞**が開口し、内部に蓄えられた**神経伝達物質**（neurotransmitter）が細胞間隙に放出される。この物質は次の細胞の**受容体**（receptor）に結合し、膜電位を変化させる。膜電位が脱分極方向に変化する場合は**興奮性シナプス後電位** (excitatory postsynaptic potential; EPSP)、過分極方向であれば**抑制性シナプス後電位**（ inhibitory postsynaptic potential; IPSP）と呼ばれる。一方、**電気シナプス**では**ギャップ結合**を通じてイオン電流が直接隣の細胞に流れ、より高速で同期的な通信が可能である。

シナプス前細胞
シナプス後細胞
スパイン
ブトン

IPSPとEPSPの補足

#### 神経細胞の種類
神経細胞はその形態や機能、伝達物質の種類により多くのサブタイプ（subtype）に分類されるが、最も基本的な区別は**興奮性ニューロン**（excitatory neuron）と**抑制性ニューロン**（inhibitory neuron）である。興奮性ニューロンは主にグルタミン酸（glutamate）を放出し、標的細胞を脱分極に導いて興奮させる。抑制性ニューロンは主にGABA（γ-アミノ酪酸, gamma-aminobutyric acid）あるいはグリシン (glycine) を放出し、標的の膜電位を過分極させて抑制する。皮質においては、神経細胞の約80%が興奮性、約20%が抑制性とされる [引用]。

特に大脳皮質や海馬において、興奮性ニューロンの代表的な形態として知られるのが**錐体細胞（pyramidal neuron）**である。錐体細胞は三角形に近い細胞体を持ち、1本の長い太い**尖端樹状突起**（apical dendrite）と複数の**基底樹状突起**（basal dendrites）を持つのが特徴である。これにより空間的に広く分布した入力を統合でき、かつ軸索はしばしば長距離にわたって他の皮質領域や皮質下構造に投射する。これらの細胞は大脳皮質では第5層や第3層に多く存在し、皮質内外の広範な情報伝達を担う。皮質回路において中心的な情報出力の担い手として、認知・運動・記憶などの高次機能に不可欠である。

神経細胞の伝達物質の一貫性に関しては、**Daleの法則**（Dale’s principle）が古くから知られている[引用]。この法則は、「一つの神経細胞はその全ての出力部位で同一の神経伝達物質を放出する」という原則である。たとえば錐体細胞はどのシナプスでもグルタミン酸を放出し、同様に抑制性ニューロンであればGABAを一貫して使用する[引用]。現在では、補助的な神経ペプチドや共放出物質の存在が知られており、Daleの法則は厳密には修正されているものの、「主たる伝達物質の一貫性」という点では今も有効な原理とされている[引用]。

このように、神経細胞はその構造、電気的性質、機能的分類において精緻な多様性と秩序を持ち、脳回路全体の動的バランスと情報処理を支えている。とりわけ、興奮性ニューロンと抑制性ニューロンの適切な協調は、神経活動の安定化と時間的精度の制御において決定的な役割を果たしている。

#### グリア細胞の種類
脳は膨大な数の細胞によって構成されており、主に**神経細胞（ニューロン）**と**グリア細胞（glial cells）**の二種類に分類される。ヒトの脳には約860億個の神経細胞が存在し、グリア細胞も同数またはそれ以上の数が含まれるとされる。これらの細胞は非常に高密度に詰まっており、1 mm³あたりでは平均して5万〜10万個の神経細胞が存在する。

神経細胞が情報の受容・統合・出力といった処理の中心を担うのに対し、**グリア細胞**はその活動を支持・調節し、神経系全体の恒常性と可塑性を維持する役割を果たす。グリア細胞には複数の種類があり、それぞれ異なる機能をもつ。

まず**アストロサイト（astrocyte）**は、中枢神経系において最も豊富なグリア細胞であり、星状の形態を持つ。アストロサイトは血管と神経細胞の間を仲介し、**血液脳関門（blood-brain barrier）**の形成、**イオン濃度の調節**、**神経伝達物質の再取り込み**、さらには**シナプスの形成と除去の調整**に関与する。神経回路の機能に対して能動的に影響を与える点で、単なる支持細胞という枠を超えた存在である。

次に、**オリゴデンドロサイト（oligodendrocyte）**は中枢神経系において**ミエリン鞘（myelin sheath）**を形成する細胞である。1個のオリゴデンドロサイトは複数の軸索に分岐を伸ばし、それぞれにミエリンを巻き付ける。ミエリンは絶縁体として機能し、**跳躍伝導**を可能にすることでスパイクの伝導速度を著しく高める。

これに対し、**シュワン細胞（Schwann cell）**は**末梢神経系（peripheral nervous system）**に存在し、オリゴデンドロサイトと類似の役割を果たす。ただし、シュワン細胞は**1つの細胞が1つの軸索の1セグメントのみにミエリンを形成する**という点でオリゴデンドロサイトとは異なる。また、シュワン細胞は神経損傷後の**再生過程の促進**にも関与する。

最後に、**ミクログリア（microglia）**は中枢神経系内に存在する**免疫担当細胞**であり、発生学的には他のグリア細胞とは起源が異なる（造血系由来）。ミクログリアは脳内の**異物の貪食（ファゴサイトーシス）**や**アポトーシス細胞の除去**を担い、また炎症性サイトカインの分泌を通じて**神経炎症応答**を調節する。最近では、ミクログリアが**シナプスの刈り込み（synaptic pruning）**にも関与することが報告されており、神経回路の発達と可塑性にも寄与していると考えられている。

このように、グリア細胞はかつて単なる「糊（glia）」として捉えられていたが、現在では神経細胞と並んで神経系の恒常性維持・可塑性制御・防御反応において不可欠な役割を担う能動的な細胞群であると認識されている。

---

こうしたイオンチャネルの働き等を考慮した神経細胞の生物物理モデルに関しては，後の第6章で説明を行う．本章から第5章までは単純化した発火率モデルを使用する．

錐体細胞の場合は，apical tuft, basal dendrite
房状分枝（tuft）

AIS

神経細胞はある程度区画化されているが，

神経細胞をモデル化する上では，詳細なモデルから抽象化されたモデルを説明する方が理解は深まるが，本書では発火率モデルの次にスパイキングモデルの説明を行う都合上，抽象化された単純なモデルから複雑な生物物理モデルの説明へと移行する．

発火率モデル (firing rate model) と呼ぶ．発火率モデルは

発火率モデル
離散時間
連続時間
を説明．形式ニューロンとperceptronは省略するか．


MuCulloch-Pittsの形式ニューロン (1943)

離散時間

$$
\mathbf{y}=f(\mathbf{Wx})
$$

連続時間
$$
\tau\frac{d\mathbf{y}(t)}{dt} = -\mathbf{y}(t) + f(\mathbf{Wx}(t))
$$

Wilson Cowan

Amari Model
https://link.springer.com/referenceworkentry/10.1007/978-1-4614-7320-6_51-2

FIRING RATE MODELS AS ASSOCIATIVE MEMORY: EXCITATORY-INHIBITORY BALANCE FOR ROBUST RETRIEVAL

Evolution of the Wilson–Cowan equations
https://link.springer.com/article/10.1007/s00422-021-00912-7

飽和関数 (saturated function) 

Naka–Rushton関数

$$
s(x) = 
\frac{Mx^2}{\sigma^2 + x^2} \cdot \Theta (x)
$$

部分飽和関数（partially saturated function）

$\Theta (x)$ はHeaviside step function

The brain wave equation: a model for the EEG
https://www.sciencedirect.com/science/article/pii/0025556474900200

Tutorial on Neural Field Theory

https://compneuro.neuromatch.io/tutorials/W2D4_DynamicNetworks/student/W2D4_Tutorial2.html

Before and beyond the Wilson-Cowan equations


## 形式ニューロン
人工ニューロンの理論的基盤は，生物の神経細胞の単純化に基づいており，その最も基本的な形式は**形式ニューロン**（formal neuron）と呼ばれる．これは，複数の入力信号を受け取り，それらを重み付きで加算し，ある閾値を超えたときに出力を発するという単純な演算規則に従うモデルである．形式ニューロンにおける出力 $y$ は，入力ベクトル $\mathbf{x} = (x_1, x_2, \dots, x_n)$ に対して，対応する重みベクトル $\mathbf{w} = (w_1, w_2, \dots, w_n)$ を用いて次のように定義される：

$$
y = \phi\left( \sum_{i=1}^n w_i x_i + b \right) = \phi(\mathbf{w}^\top \mathbf{x} + b)
$$

ここで，$b$ はバイアス項であり，$\phi(\cdot)$ は活性化関数（activation function）を表す．この活性化関数には階段関数やシグモイド関数，ReLU関数などが用いられるが，形式ニューロンの原型では一般に**ステップ関数（Heaviside関数）**が想定されており，これは以下のように定義される：

$$
\phi(u) = 
\begin{cases}
1 & \text{if } u \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

このようにして形式ニューロンは，入力の線形結合がある閾値を超えるかどうかによって出力を決定する**閾値判定器**として機能する．


## パーセプトロン
形式ニューロンに学習機構を導入したものが**パーセプトロン（perceptron）**である．1958年にローゼンブラット（Rosenblatt）によって提案されたこのモデルは，出力が目標値と一致しない場合に重みを修正する単純な学習則を備えており，2クラス分類問題において線形分離可能なパターンを識別することができる．パーセプトロンの出力は形式ニューロンと同様に

$$
y = \phi(\mathbf{w}^\top \mathbf{x} + b)
$$

と表されるが，学習の過程でパラメータ $\mathbf{w}, b$ は次のように更新される：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta (t - y) \mathbf{x}, \quad b \leftarrow b + \eta (t - y)
$$

ここで，$t$ は目標出力（target），$y$ は現在の出力，$\eta$ は学習率（learning rate）を表す．この更新式は，出力が誤っていた場合にその方向に重みを修正するという単純なルールに基づいており，反復的に適用することで分類精度を高めていく．ただし，パーセプトロンは**線形分離可能でないデータ**に対しては学習が収束せず，分類が失敗するという制限を持つ．

一方で，実際の神経細胞の活動は0か1のような離散的な出力ではなく，**一定時間あたりの発火頻度（spike rate）**として観察されることが多い．この現象を抽象化したものが**発火率モデル（rate-based model）**である．このモデルでは，ニューロンの出力はスパイクの発生の有無ではなく，時間的平均を取った連続値（例えば Hz 単位の発火頻度）として表現される．数式的には，出力は活性化関数を通じた連続値として次のように定義される：

$$
r = \phi(\mathbf{w}^\top \mathbf{x} + b)
$$

ここで $r$ はニューロンの発火率（rate），$\phi$ は連続的かつ微分可能な関数であり，しばしばシグモイド関数

$$
\phi(u) = \frac{1}{1 + \exp(-u)}
$$

や整流線形単位（ReLU関数）

$$
\phi(u) = \max(0, u)
$$

などが用いられる．このように発火率モデルは，数理的にはパーセプトロンと類似の構造を持ちながらも，出力を連続的に扱うことで，生物学的ニューロンの性質や誤差逆伝播法による学習の実装に適した形式となっている．

したがって，形式ニューロンはニューロンの抽象的な演算原理を示す基礎であり，パーセプトロンはその学習機構を導入した離散分類モデル，発火率モデルは生物学的な現象をより忠実に反映した連続出力モデルと位置づけることができる．これらは人工ニューラルネットワークの設計と理解における基本的構成要素であり，それぞれの仮定と構造を理解することは，後続の深層学習や神経科学的モデルを学ぶための土台となる．

## ニューロンの発火率モデルとパーセプトロン


生物学的な神経回路に着想を得た計算モデルのひとつに、**ニューロンの発火率モデル**がある。このモデルでは、個々のニューロンが入力刺激に対してどの程度活性化されるか（＝発火するか）を連続値で表現し、情報の処理や伝達を数理的に記述する。

### ニューロンの発火率モデル

生物学的ニューロンは、樹状突起を通じて他のニューロンから電気的信号（シナプス電位）を受け取り、その総和が一定の閾値を超えた場合に軸索を通じてスパイク（活動電位）を発生させる。このスパイクの頻度（単位時間あたりの発火回数）は、入力の強度に依存して変化することが知られており、この関係を**発火率モデル** (*firing rate model*) と呼ぶ。

数学的には、入力信号の線形結合を $z = \sum_{j=1}^p w_j x_j + w_0$ とし、発火率（ニューロンの出力）を何らかの**活性化関数** $f(z)$ を通じて表現する：

$$
y = f(z) = f\left(\sum_{j=1}^p w_j x_j + w_0\right)
$$

このように、ニューロンは入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して加重和を計算し、それに非線形な関数を適用することで、出力 $y$ を生成する。代表的な活性化関数としては、**シグモイド関数**や**ReLU（Rectified Linear Unit）関数**がある。特にシグモイド関数は、出力が $[0,1]$ の範囲に収まるため、発火率（スパイク頻度）の確率的な解釈と親和性が高い。

シグモイドは正規化された発火率関数と解釈できる．

この発火率モデルは、後述するロジスティック回帰やニューラルネットワークの基本構成要素として広く用いられている。

### パーセプトロン

このようなニューロンの抽象化を、分類問題に適用するために提案されたのが**パーセプトロン** (*perceptron*) である。パーセプトロンは、入力と重みの線形結合に対して符号関数（ステップ関数）を適用することにより、2クラス分類を実現する最も基本的な形式の**人工ニューロンモデル**である。

#### モデルの構造

入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して、重みベクトル $\mathbf{w} \in \mathbb{R}^{p+1}$（バイアス項 $w_0$ を含む）を用いて線形結合 $z = \mathbf{w}^\top \mathbf{x}'$ を計算する。ただし、$\mathbf{x}' := [1, x_1, x_2, \dots, x_p]^\top$ としてバイアス項を組み込んだ拡張ベクトルを用いる。

次に、活性化関数として**符号関数**を適用する：

$$
\hat{y} = \text{sign}(z) = \begin{cases}
+1 & (z \geq 0) \\
-1 & (z < 0)
\end{cases}
$$

これにより、出力 $\hat{y} \in \{-1, +1\}$ が得られ、2クラス分類を実現する。

#### 学習アルゴリズム：パーセプトロン則

パーセプトロンは**教師あり学習**に基づいて重み $\mathbf{w}$ を更新する。各ステップにおいて、予測と正解が一致していれば何も行わず、誤分類されたときにのみ以下のように重みを更新する：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
$$

ここで、$\eta > 0$ は学習率、$y^{(i)}$ は正解ラベルである。この更新則により、パーセプトロンは誤分類を修正する方向に重みを調整する。

もし訓練データが**線形分離可能**であるならば、このアルゴリズムは有限回の更新で必ず収束する（**パーセプトロン収束定理**）。ただし、データが線形分離不可能な場合は、収束せずに振動を続けることがある。

### ニューロンモデルとの関係

パーセプトロンは、ニューロンの発火率モデルを単純化した形式として位置づけられる。ニューロンの出力が連続値であるのに対し、パーセプトロンでは出力が離散値（2値）である点が主な違いである。また、ニューロンモデルではなめらかな活性化関数が使われるのに対し、パーセプトロンでは不連続な符号関数が用いられる。

このように、**パーセプトロンはニューロンの発火を単純な2値信号として抽象化した分類器**であり、人工知能の初期における重要なモデルのひとつである。さらに、現代の多層ニューラルネットワーク（ディープラーニング）においても、ニューロンの発火率モデルの構造は基本単位として継承されている。

### 1層パーセプトロン

**パーセプトロン**は、1958年に Rosenblatt によって提案された、最も基本的な形式の線形分類器である。ロジスティック回帰と同様に線形結合を用いるが、確率ではなく**符号関数によって離散的な出力**を行う点が異なる。

#### モデルの定義

入力 $\mathbf{x} \in \mathbb{R}^p$ に対し、次のように線形結合を計算する：

$$
z = \mathbf{w}^\top \mathbf{x}'
$$

そして、活性化関数として符号関数 $\text{sign}(z)$ を適用することで、2クラス分類を行う：

$$
\hat{y} = \begin{cases}
+1 & \text{if } z \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$$

このように、パーセプトロンは出力を $\{-1, +1\}$ のいずれかに決定的に分類する。

#### パーセプトロン学習則

パーセプトロンは教師あり学習アルゴリズムに基づいて重みを更新する。予測が正しい場合は更新を行わず、予測が誤っていた場合にのみ次のようにパラメータを更新する：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
$$

ここで $\eta > 0$ は学習率である。更新は誤分類されたデータ点に対してのみ行われ、分類境界が調整される。データが線形分離可能であれば、パーセプトロン学習則は有限回の更新で収束することが知られている（**パーセプトロン収束定理**）。

分類問題
, perceptron
<https://www.cs.utexas.edu/~gdurrett/courses/fa2022/perc-lr-connections.pdf>

<https://en.wikipedia.org/wiki/Perceptron>

<https://arxiv.org/abs/2012.03642>


perceptronは0/1 or -1/1のどちらか

UNDERSTANDING STRAIGHT-THROUGH ESTIMATOR IN TRAINING ACTIVATION QUANTIZED NEURAL NETS

Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Hinton (2012) in his lecture 15b

G. Hinton. Neural networks for machine learning, 2012.
<https://www.cs.toronto.edu/~hinton/coursera_lectures.html>

delta rule


Here σ denotes the (point-wise) activation function, $W \in R^{m\times n}$
is the weight-matrix and $b \in R^n$
is
the bias-vector. The vector $x \in R^m$ and the vector $y \in R^n$ denote the input, respectively the output

$$
\begin{equation}
y=\sigma(W^\top x + b)
\end{equation}
$$

$$
\begin{align}
& \text { Initialize } W^0, b^0 \text {; } \\
& \text { for } k=1,2, \ldots \text { do } \\
& \qquad \begin{array}{|l}
\text { for } i=1, \ldots, s \text { do } \\
e_i=y_i-\sigma\left(\left(W^k\right)^{\top} x_i+b^k\right) \\
W^{k+1}=W^k+e_i x_i^{\top} \\
b^{k+1}=b^k+e_i
\end{array} \\
& \text { end }
\end{align}
$$

これは単純ではあるが，この微分不可能な関数による学習則は，現代的に**Straight-Through Estimator** (STE) と呼ばれる概念と同一である．STEの考えはスパイキングニューラルネットワークの学習や，ニューラルネットワークの量子化へと発展する．ここでは深く触れず，第7章で改めて紹介を行う．

## Hebb則と主成分分析
### Hebb則
神経回路はどのようにして自己組織化するのだろうか．1940年代にカナダの心理学者Donald O. Hebbにより著書"The Organization of Behavior"{cite:p}`Hebb1949-iv` で提案された学習則は「細胞Aが反復的または持続的に細胞Bの発火に関与すると，細胞Aが細胞Bを発火させる効率が向上するような成長過程または代謝変化が一方または両方の細胞に起こる」というものであった．すなわち，発火に時間的相関のある細胞間のシナプス結合を強化するという学習則である．これを**Hebbの学習則** (Hebbian learning rule) あるいは**Hebb則** (Hebb's rule) という．Hebb則は（Hebb自身ではなく）Shatzにより"cells that fire together wire together"（共に活動する細胞は共に結合する）と韻を踏みながら短く言い換えられている {cite:p}`Shatz1992-he`．

数式でHebb則を表してみよう．$n$個のシナプス前細胞と$m$個の後細胞の発火率をそれぞれ$\mathbf{x}\in \mathbb{R}^n, \mathbf{y}\in \mathbb{R}^m$ とする．前細胞と後細胞間のシナプス結合強度を表す行列を$\mathbf{W}\in \mathbb{R}^{m\times n}$とし，$\mathbf{y}=\mathbf{W}\mathbf{x}$が成り立つとする．このようなモデルを線形ニューロンモデル (Linear neuron model) という．このとき，Hebb則は

$$
\begin{equation}
\tau\frac{d\mathbf{W}}{dt}=\phi(\mathbf{y})\varphi(\mathbf{x})^\top
\end{equation}
$$

として表される．ただし，$\tau$は時定数であり，$\eta:=1/\tau$ は**学習率** (learning rate) と呼ばれる学習の速さを決定するパラメータとなる．$\varphi(\cdot)$および$\phi(\cdot)$は，それぞれシナプス前細胞および後細胞の活動量に応じて重みの変化量を決定する関数である．$\varphi(\cdot), \phi(\cdot)$ が恒等関数に設定される場合，Hebb則は $\tau\dfrac{d\mathbf{W}}{dt}=\mathbf{y}\mathbf{x}^\top=(\text{post})\cdot (\text{pre})^\top$ と簡潔に表現される．


#### Hebb則の生理的機序とLTP・LTDの実験的発見
LTPの実験的発見 {cite:p}`Bliss1973-vj` {cite:p}`Dudek1992-nz`

このHebb則の神経生理学的な基盤を裏付けるものとして，1973年にBlissとLømoによってウサギの海馬において**長期増強**（Long-Term Potentiation, LTP）が発見された．彼らの実験では，海馬のシェイファー側枝からCA1錐体細胞への経路に高頻度の電気刺激を加えることで，その後のシナプス応答が長時間にわたって増強される現象が観察された．この持続的なシナプス強度の増加は，まさにHebb則に対応する生理的現象と見なされ，Hebbian plasticityの実体と考えられるようになった．LTPはグルタミン酸作動性シナプスで観察されることが多く，特にNMDA受容体が関与することで知られている．この受容体は膜電位依存的にMg²⁺ブロックが外れることにより，カルシウムイオン（Ca²⁺）の流入を許し，それが下流のシグナル伝達を活性化してシナプス後部のAMPA受容体の増加や活性化を引き起こす．

一方，1980年代には**長期抑圧**（Long-Term Depression, LTD）という現象も発見された．これは，シナプス前ニューロンとシナプス後ニューロンが低頻度で同時活動した場合に，シナプスの伝達効率が長期にわたって減少する現象である．LTDもまた海馬や小脳などの領域で観察されており，この減弱はHebb則の反対の効果を示すものとして位置づけられる．特に，小脳における登上線維と平行線維の同時活動により引き起こされるLTDは，運動学習のモデルとして重要視されている．LTPと同様に，LTDにおいてもCa²⁺シグナリングが重要な役割を果たすが，その振幅や時間的プロファイルが異なっていることが，シナプス強化と抑圧の分岐をもたらすと考えられている．

これらの発見を通じて，Hebb則は単なる理論的仮説にとどまらず，シナプス可塑性という具体的な細胞メカニズムを通して，神経回路における学習と記憶の基盤であることが明らかにされた．

#### 神経ダイナミクスからのHebb則の導出
Hebb則は数学的に導出されたものではないが，特定の目的関数を神経活動及び重みを変化させて最適化するようなネットワークを構築すれば自然に出現する．このようなネットワークを**エネルギーベースモデル** (energy-based models) といい，次章で扱う．エネルギーベースモデルでは，先にエネルギー関数 (あるいはコスト関数) $\mathcal{E}$ を定義し，その目的関数を最小化するような神経活動 $\mathbf{z}$ および重み行列 $\mathbf{W}$ のダイナミクスをそれぞれ,

$$
\begin{equation}
\frac{d \mathbf{z}}{dt}\propto-\left(\frac{\partial \mathcal{E}}{\partial \mathbf{z}}\right)^\top,\quad\frac{d \mathbf{W}}{dt}\propto-\left(\frac{\partial \mathcal{E}}{\partial \mathbf{W}}\right)^\top
\end{equation}
$$

として導出する．この手順の逆を行う，すなわち先に神経細胞の活動ダイナミクスを定義し，神経活動で積分することで神経回路のエネルギー関数$\mathcal{E}$を導出し，さらに $\mathcal{E}$ を重み行列で微分することでHebb則が導出できる {cite:p}`Isomura2020-sn`．Hebb則の導出を連続時間線形ニューロンモデル $\dfrac{d\mathbf{y}}{dt}=\mathbf{W}\mathbf{x}$ を例にして考えよう（簡単のため $\tau=1$ とした）．ここで$\dfrac{\partial\mathcal{E}}{\partial\mathbf{y}}:=-\left(\dfrac{d\mathbf{y}}{dt}\right)^\top$となるようなエネルギー関数 $\mathcal{E}(\mathbf{x}, \mathbf{y}, \mathbf{W})$を仮定すると，

$$
\begin{equation}
\mathcal{E}(\mathbf{x}, \mathbf{y}, \mathbf{W})=-\int \mathbf{W}\mathbf{x}\ d\mathbf{y}=-\mathbf{y}^\top \mathbf{W}\mathbf{x} \in \mathbb{R}
\end{equation}
$$

となる．これをさらに$\mathbf{W}$で微分すると，

$$
\begin{equation}
\dfrac{\partial\mathcal{E}}{\partial\mathbf{W}}=-\mathbf{y}\mathbf{x}^\top\Rightarrow
\frac{d\mathbf{W}}{dt}=-\dfrac{\partial\mathcal{E}}{\partial\mathbf{W}}=\mathbf{y}\mathbf{x}^\top
\end{equation}
$$

となり，Hebb則が導出できる．

### Hebb則の安定化
#### BCM則
Hebb則には問題点があり，シナプス結合強度が際限なく増大するか，0に近づくこととなってしまう．これを数式で確認しておこう．前細胞と後細胞がそれぞれ1つの場合を考える．2細胞間の結合強度を$w\ (>0)$ とし，$y=wx$が成り立つとすると，Hebb則は$\dfrac{dw}{dt}=\eta yx=\eta x^2w$となる．この場合，$\eta x^2>1$ なら $\lim_{t\to\infty} w= \infty$, $\eta x^2<1$ なら $\lim_{t\to\infty} w= 0$ となる．当然，生理的にシナプス結合強度が無限大となることはあり得ないが，不安定なほど大きくなってしまう可能性があることに違いはない．このため，Hebb則を安定化させるための修正が必要とされた．

Cooper, Liberman, Ojaらにより頭文字をとって**CLO則** (CLO rule) が提案された {cite:p}`Cooper1979-wz`．その後，Bienenstock, Cooper, Munroらにより提案された学習則は同様に頭文字をとって**BCM則** (BCM rule) と呼ばれている{cite:p}`Bienenstock1982-km` {cite:p}`Cooper2012-ec`．

$\mathbf{x}\in \mathbb{R}^d, \mathbf{w}\in \mathbb{R}^d, y\in \mathbb{R}$とし，単一の出力$y = \mathbf{w}^\top \mathbf{x}=\mathbf{x}^\top \mathbf{w}$を持つ線形ニューロンを仮定する．重みの更新則は次のようにする．

$$
\begin{equation}
\frac{d\mathbf{w}}{dt} = \eta_w \mathbf{x} \phi(y, \theta_m)
\end{equation}
$$

ここで関数$\phi$は$\phi(y, \theta_m)=y(y-\theta_m)$などとする．また$\theta_m:=\mathbb{E}[y^2]$は閾値を決定するパラメータ，**修正閾値** (modification threshold) であり，

$$
\begin{equation}
\frac{d\theta_m}{dt} = \eta_{\theta} \left(y^2-\theta_m\right)
\end{equation}
$$

として更新される．

#### Oja則
Hebb則を安定化させる別のアプローチとして，結合強度を正規化するという手法が考えられる．BCM則と同様に$\mathbf{x}\in \mathbb{R}^d, \mathbf{w}\in \mathbb{R}^d, y\in \mathbb{R}$とし，単一の出力$y = \mathbf{w}^\top \mathbf{x}=\mathbf{x}^\top \mathbf{w}$を持つ線形ニューロンを仮定する．$\eta$を学習率とすると，$\mathbf{w}\leftarrow\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とすれば正規化できる．ここで，$f(\eta):=\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とし，$\eta=0$においてTaylor展開を行うと，

$$
\begin{align}
f(\eta)&\approx f(0) + \eta \left.\frac{df(\eta^*)}{d\eta^*}\right|_{\eta^*=0} + \mathcal{O}(\eta^2)\\
&=\frac{\mathbf{w}}{\|\mathbf{w}\|} + \eta \left(\frac{\mathbf{x}y}{\|\mathbf{w}\|}-\frac{y^2\mathbf{w}}{\|\mathbf{w}\|^3}\right)+ \mathcal{O}(\eta^2)
\end{align}
$$

ここで$\|\mathbf{w}\|=1$として，1次近似すれば$f(\eta)\approx \mathbf{w} + \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)$となる．重みの変化が連続的であるとすると，

$$
\begin{equation}
\frac{d\mathbf{w}}{dt} = \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)
\end{equation}
$$

として重みの更新則が得られる．これを**Oja則 (Oja's rule)** と呼ぶ {cite:p}`Oja1982-yd`．こうして得られた学習則において$\|\mathbf{w}\|\to 1$となることを確認しよう．

$$
\begin{equation}
\frac{d\|\mathbf{w}\|^2}{dt}=2\mathbf{w}^\top\frac{d\mathbf{w}}{dt}= 2\eta y^2\left(1-\|\mathbf{w}\|^2\right)
\end{equation}
$$

より，$\dfrac{d\|\mathbf{w}\|^2}{dt}=0$のとき，$\|\mathbf{w}\|= 1$となる．

#### 恒常的可塑性
Oja則は更新時の即時的な正規化から導出されたものであるが，恒常的可塑性 (synaptic scaling)により安定化しているという説がある{cite:p}`Turrigiano2008-lm`{cite:p}`Yee2017-fb`．しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない{cite:p}`Zenke2017-el`

ToDo:恒常的可塑性の詳細

Johansen, Joshua P., Lorenzo Diaz-Mataix, Hiroki Hamanaka, Takaaki Ozawa, Edgar Ycu, Jenny Koivumaa, Ashwani Kumar, et al. 2014. “Hebbian and Neuromodulatory Mechanisms Interact to Trigger Associative Memory Formation.” Proceedings of the National Academy of Sciences 111 (51): E5584–92.

### Hebb則と主成分分析
Oja則を用いることで**主成分分析** (Principal component analysis; PCA) という処理をニューラルネットワークにおいて実現できる．

#### 主成分分析
主成分分析 (PCA) は，高次元のデータに内在する低次元の構造を抽出するための線形次元削減法である．この手法は，分散が最大となる方向にデータを射影することにより，元の情報をなるべく保ちながら次元を削減する．

まず，$n$ 個のサンプル $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$ が $d$ 次元の実ベクトル空間 $\mathbb{R}^d$ に属するとし，これらを列ベクトルとしてまとめたデータ行列を $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ とする．PCA では以下の手順を踏む．

1. **平均の除去**  
   各特徴量について平均を 0 にするため，データを中心化する：
   $$
   \bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i, \quad \tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}.
   $$
   中心化されたデータ行列を $\tilde{\mathbf{X}}$ とおく．

2. **共分散行列の構築**  
   中心化後のデータから共分散行列 $\mathbf{C}$ を求める：
   $$
   \mathbf{C} = \frac{1}{n} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \in \mathbb{R}^{d \times d}.
   $$

3. **固有値分解**  
   共分散行列に対して固有値分解を行い，固有ベクトル $\{\mathbf{w}_1, \dots, \mathbf{w}_d\}$ と対応する固有値 $\{\lambda_1, \dots, \lambda_d\}$ を求める．固有値は分散量に対応し，$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ の順に並べる．固有ベクトルは以下を満たす：
   $$
   \mathbf{C} \mathbf{w}_k = \lambda_k \mathbf{w}_k, \quad k=1,\dots,d.
   $$

4. **次元削減と主成分の構成**  
   上位 $m < d$ 個の固有ベクトル $\mathbf{W}_m = [\mathbf{w}_1, \dots, \mathbf{w}_m]$ を用いて，元のデータを $m$ 次元に射影する：
   $$
   \mathbf{z}_i = \mathbf{W}_m^\top \tilde{\mathbf{x}}_i \in \mathbb{R}^m.
   $$
   これにより得られる $\mathbf{z}_i$ は主成分と呼ばれる．

PCA の目的は，情報損失（再構成誤差）を最小限に抑えながら，できるだけ少ない次元でデータを表現することである．この観点から，PCA は次の最適化問題の解とみなすこともできる：

$$
\max_{\mathbf{W}_m \in \mathbb{R}^{d \times m}} \operatorname{Tr}(\mathbf{W}_m^\top \mathbf{C} \mathbf{W}_m), \quad \text{s.t. } \mathbf{W}_m^\top \mathbf{W}_m = \mathbf{I}_m,
$$

ここで $\operatorname{Tr}(\cdot)$ はトレース演算，$\mathbf{I}_m$ は $m$ 次の単位行列である．この最適化問題の解は，共分散行列 $\mathbf{C}$ の上位 $m$ 個の固有ベクトルからなる直交行列 $\mathbf{W}_m$ である．

PCA はデータの冗長性を取り除くと同時に，ノイズの低減や可視化の手法としても広く応用される．また，線形変換であるため，計算効率も高いという特徴がある．

svdを用いて実装をする．

#### Oja則によるPCAの実行
主成分分析はOja則を応用することで神経回路上に実装できる．重みの変化量の期待値を取る．

$$
\begin{align}
\frac{d\mathbf{w}}{dt} &= \eta \left(\mathbf{x}y - y^2 \mathbf{w}\right)=\eta \left(\mathbf{x}\mathbf{x}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{x}\mathbf{x}^\top \mathbf{w}\right] \mathbf{w}\right)\\
\mathbb{E}\left[\frac{d\mathbf{w}}{dt}\right] &= \eta \left(\mathbf{C} \mathbf{w} - \left[\mathbf{w}^\top \mathbf{C} \mathbf{w}\right] \mathbf{w}\right)
\end{align}
$$

$\mathbf{C}:=\mathbb{E}[\mathbf{x}\mathbf{x}^\top]\in \mathbb{R}^{d\times d}$とする．$\mathbf{x}$の平均が0の場合，$\mathbf{C}$は分散共分散行列である．$\mathbb{E}\left[\dfrac{d\mathbf{w}}{dt}\right]=0$となる$\mathbf{w}$が収束する固定点(fixed point)では次の式が成り立つ．

$$
\begin{equation}
\mathbf{C}\mathbf{w} = \lambda \mathbf{w}
\end{equation}
$$

これは固有値問題であり，$\lambda:=\mathbf{w}^\top \mathbf{C} \mathbf{w}$は固有値，$\mathbf{w}$は固有ベクトル(eigen vector)になる．

ここでサンプルサイズを$n$とし，$\mathbf{X} \in \mathbb{R}^{d\times n}, \mathbf{y}=\mathbf{X}^\top\mathbf{w} \in \mathbb{R}^n$とする．標本平均で近似して$\mathbf{C}\simeq \mathbf{X}\mathbf{X}^\top$とする．この場合，

$$
\begin{align}
\mathbb{E}\left[\frac{d\mathbf{w}}{dt}\right] &\simeq \eta \left(\mathbf{X}\mathbf{X}^\top \mathbf{w} - \left[\mathbf{w}^\top \mathbf{X}\mathbf{X}^\top \mathbf{w}\right] \mathbf{w}\right)\\
&=\eta \left(\mathbf{X}\mathbf{y} - \left[\mathbf{y}^\top\mathbf{y}\right] \mathbf{w}\right)
\end{align}
$$

となる．

後のためにOja則においてネットワークが$q$個の複数出力を持つ場合を考えよう．重み行列を$\mathbf{W} \in \mathbb{R}^{q\times d}$, 出力を$\mathbf{y}=\mathbf{W}\mathbf{x} \in \mathbb{R}^{q}, \mathbf{Y}=\mathbf{W}\mathbf{X} \in \mathbb{R}^{q\times n}$とする．この場合の更新則は

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left(\mathbf{y}\mathbf{x}^\top - \mathrm{Diag}\left[\mathbf{y}\mathbf{y}^\top\right] \mathbf{W}\right)
\end{equation}
$$

となる．ただし，$\mathrm{Diag}(\cdot)$は行列の対角成分からなる対角行列を生み出す作用素である．

#### Sanger則
Oja則に複数の出力を持たせた場合であっても，出力が直交しないため，PCAの第1主成分しか求めることができない．**Sanger則** (Sanger's rule)，あるいは**一般化Hebb則** (generalized Hebbian algorithm; GHA) は，Oja則に**Gram–Schmidtの正規直交化法** (Gram–Schmidt orthonormalization) を組み合わせた学習則であり，次式で表される．

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left[\mathbf{y}\mathbf{x}^\top - \mathrm{LT}\left(\mathbf{y}\mathbf{y}^\top\right) \mathbf{W}\right]
\end{equation}
$$

$\mathrm{LT}(\cdot)$は行列の対角成分より上側の要素を0にした下三角行列(lower triangular matrix)を作り出す作用素である．Sanger則を用いればPCAの第2主成分以降も求めることができる．

### 非線形Hebb学習
出力$\mathbf{y}$に非線形関数$g(\cdot)$を適用し，$\mathbf{y}\to g(\mathbf{y})$として置き換えることで非線形Hebb学習となる{cite:p}`Oja1997-hr`{cite:p}`Brito2016-mx`. 関数`HebbianPCA`の`func`引数に非線形関数を渡すことで実現できる．

ToDo: 詳細

#### 非負主成分分析によるグリッドパターンの創発
内側嗅内皮質(MEC)にある**グリッド細胞** (grid cells) は六角形格子状の発火パターンにより自己位置等を符号化するのに貢献している．この発火パターンを生み出すモデルは多数あるが，**場所細胞** (place cells) の発火パターンを**非負主成分分析** (nonnegative principal component analysis) で次元削減するとグリッド細胞のパターンが生まれるというモデルがある {cite:p}`Dordek2016-ff`．非線形Hebb学習を用いてこのモデルを実装しよう．なお，同様のことは**非負値行列因子分解** (nonnegative matrix factorization; NMF) でも可能である．

##### 場所細胞の発火パターン
まず，訓練データとなる場所細胞の発火パターンを人工的に作成する．場所細胞の発火パターンはガウス差分フィルタ (difference of Gaussians; DoG) で近似する．DoGは大きさの異なる2つのガウス関数の差分を取った関数であり，画像に適応すればband-passフィルタとして機能する．また，DoGは網膜神経節細胞等の受容野のON中心OFF周辺型受容野のモデルとしても用いられる．受容野中央では活動が大きく，その周辺では活動が抑制される，という特性を持つ．2次元のガウス関数とDoG関数を実装する．

Place cellの受容野をDoGに設定したが，これが無いと格子状の受容野は出現しない．path integrationをRNNで実行する場合も同様．一方で，DoGは場所細胞の受容野としては不適切である．

No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit 
<https://openreview.net/forum?id=mxi1xKzNFrb>

ToDo: 他のgrid cellsのモデルについて

## 独立成分分析
独立成分分析（Independent Component Analysis; ICA）は，観測された多次元信号が，統計的に独立な複数の潜在変数（独立成分）の線形混合であると仮定し，元の独立成分を復元することを目的とする手法である．ICAは特に，脳波や自然画像などに見られる信号分離問題に有効である．Blind source separation.

ICAでは，観測ベクトル $\mathbf{x} \in \mathbb{R}^n$ が独立な潜在変数ベクトル $\mathbf{s} \in \mathbb{R}^n$ の線形混合であると仮定する．すなわち，

$$
\mathbf{x} = \mathbf{A} \mathbf{s}
$$

と表される．ここで，$\mathbf{A}$ は未知の正則行列であり，これを分離行列 $\mathbf{W}$ によって推定することを目指す．独立成分 $\mathbf{s}$ の推定は，

$$
\mathbf{y} = \mathbf{W} \mathbf{x}
$$

と表されるように行い，得られた $\mathbf{y}$ の各成分が統計的に独立となるように $\mathbf{W}$ を求める．

ICAを実現するための代表的な原理の一つに，InfoMax（情報最大化）原理がある．これは，出力変数の情報量（エントロピー）を最大化するように変換を学習する枠組みである．InfoMaxにおいては，神経回路の情報伝達能力を最大化するという考えに基づき，非線形関数を通じた変換の出力エントロピーを最大化する．

具体的には，非線形活性化関数 $g(\cdot)$ を用いた出力

$$
\mathbf{y} = g(\mathbf{W} \mathbf{x})
$$

に対し，出力のエントロピー $H(\mathbf{y})$ を最大化するように $\mathbf{W}$ を調整する．ただし，$g(\cdot)$ は例えばシグモイド関数のような非線形性を持つ関数とする．

InfoMax原理に基づくICAの学習則は，出力の対数尤度を最大化する勾配上昇法として導出される．例えば，出力の対数尤度を $L(\mathbf{W})$ としたとき，

$$
\nabla_{\mathbf{W}} L(\mathbf{W}) \propto \left( \mathbf{I} + (\mathbf{1} - 2\mathbf{y}) \mathbf{x}^\top \right) \mathbf{W}^{-\top}
$$

といった形の学習則が得られる（ここで，$\mathbf{1}$ は全ての成分が1のベクトル）．このようにして，$\mathbf{y}$ の統計的独立性が最大化されるような $\mathbf{W}$ を求めることが可能となる．

InfoMax ICAは，確率密度関数の仮定を明示せずに信号の非ガウス性を利用する点で有効であり，実際の信号分離問題において高い性能を示すことが多い．また，非ガウス性の測度としてはクルトーシスやネガエントロピーなども用いられ，これによりFastICAなどの手法も導出されている．

以上より，独立成分分析は，観測データを生成する潜在変数の独立性という前提に基づき，情報理論的な原理に従ってその分離を行う手法であり，InfoMaxはその実現方法の一つとして広く用いられている．

## 低速特徴分析
**Slow Feature Analysis (SFA)** とは, 複数の時系列データの中から低速に変化する成分 (slow feature) を抽出する教師なし学習のアルゴリズムである \citep{Wiskott2002-vb,Wiskott2011-uz}．潜在変数 $y$ の時間変化の2乗である $\left(\frac{dy}{dt}\right)^2$を最小にするように教師なし学習を行う．初期視覚野の受容野 \citep{Berkes2005-i} や格子細胞・場所細胞などのモデルに応用がされている \citep{Franzius2007-sf}．

生理学的妥当性についてはいくつかの検討がされている．\citep{Sprekeler2007-qm} ではSTDP則によりSFAが実現できることを報告している．古典的な線形Recurrent neural networkでの実装も提案されている \citep{Lipshutz2020-uj}．

より具体的には，観測された高次元の入力信号 $\mathbf{x}(t) \in \mathbb{R}^n$ から，できるだけゆっくりと変化するスカラー出力 $y(t) = g(\mathbf{x}(t))$ を学習によって導出することが目的である．このとき，関数 $g(\cdot)$ は通常，入力に対して線形または非線形な写像である．

SFAの基本的な最適化問題は以下のように定式化される：

$$
\min_{g} \left\langle \left( \frac{d}{dt} g(\mathbf{x}(t)) \right)^2 \right\rangle_t
$$

ただし，$\langle \cdot \rangle_t$ は時間平均を意味する．このままでは自明な定数解（全く変化しない出力）が得られるため，以下のような制約条件を課す：

1. **零平均**：$\langle y(t) \rangle_t = 0$
2. **単位分散**：$\langle y(t)^2 \rangle_t = 1$
3. **異なる特徴間の直交性**（複数のslow featureを抽出する場合）：$\langle y_i(t) y_j(t) \rangle_t = 0\quad (i \ne j)$

これらの制約により，情報量がありながらも変化の遅い特徴を抽出することが可能となる．実際のアルゴリズムでは，まず入力信号に対して一定の非線形写像（例えば多項式基底関数など）を適用した後，主成分分析（PCA）によって前処理を行い，その後時間的変化の最小化問題を一般化固有値問題として解くことでslow featuresを得る．

まずデータセットの生成を行う．\citep{Wiskott2002-vb}で用いられているトイデータを用いる．

Slow Feature Analysis (SFA) は，時系列データに含まれる情報のうち，時間的に最もゆっくりと変化する成分（slow features）を抽出するための教師なし学習アルゴリズムである．このアルゴリズムでは，観測された高次元の信号 $\mathbf{x}(t) \in \mathbb{R}^n$ に対して，線形または非線形な写像 $y(t) = g(\mathbf{x}(t))$ を学習し，その出力が時間的に滑らかになるように設計される．特に線形SFAの場合，写像 $g(\mathbf{x})$ は線形関数 $\mathbf{w}^\top \mathbf{x}(t)$ として表され，その時間微分の2乗平均 $\left\langle \left( \frac{d}{dt} \mathbf{w}^\top \mathbf{x}(t) \right)^2 \right\rangle_t$ を最小化することが目的となる．

この最適化問題を解くためには，まず入力データ $\mathbf{x}(t)$ を前処理し，時間平均を引くことでゼロ平均化する．次に，共分散行列 $\mathbf{C}_x = \langle \tilde{\mathbf{x}}(t) \tilde{\mathbf{x}}(t)^\top \rangle_t$ を求め，これに対して固有値分解 $\mathbf{C}_x = \mathbf{E} \mathbf{D} \mathbf{E}^\top$ を適用することで主成分空間を構成し，白色化変換 $\mathbf{z}(t) = \mathbf{D}^{-1/2} \mathbf{E}^\top \tilde{\mathbf{x}}(t)$ を得る．この変換により，$\mathbf{z}(t)$ は単位分散かつ直交性を持つ特徴ベクトルとなる．

白色化されたデータに対して時間微分を近似的に計算し，$\dot{\mathbf{z}}(t) = \mathbf{z}(t+1) - \mathbf{z}(t)$ と定義することで，その共分散行列 $\mathbf{C}_{\dot{z}} = \langle \dot{\mathbf{z}}(t) \dot{\mathbf{z}}(t)^\top \rangle_t$ を構築することができる．SFAにおける主たる目的は，この微分共分散行列に関する最小固有値問題を解くことである．すなわち，$\mathbf{C}_{\dot{z}}$ に対する固有値分解または特異値分解（SVD）を行い，最小固有値に対応する固有ベクトル $\mathbf{u}_1$ を求めることで，最もゆっくりと変化する成分 $y(t) = \mathbf{u}_1^\top \mathbf{z}(t)$ を得ることができる．複数のslow featuresを得たい場合は，対応する小さい固有値順に固有ベクトルを選択することで可能となる．

最終的に，元のデータ空間におけるslow featuresを得るためには，逆変換を施して $\mathbf{W} = \mathbf{E} \mathbf{D}^{-1/2} \mathbf{U}$ とし，$\mathbf{U}$ は選択された固有ベクトルからなる行列である．この射影行列 $\mathbf{W}$ を用いることで，元の信号 $\tilde{\mathbf{x}}(t)$ からslow feature $y(t) = \mathbf{W}^\top \tilde{\mathbf{x}}(t)$ を得ることができる．このようにして，SFAはSVDを通じて効率的に解くことが可能であり，低速に変化する潜在表現を抽出するための強力な手法となっている．

## 自己組織化マップ

### 競合学習
Feature discovery by competitive learning

### 自己組織化マップと視覚野の構造
**自己組織化マップ**（Self-Organizing Map; SOM）は、Kohonenによって提案された教師なし学習アルゴリズムであり、高次元データを低次元（通常は2次元）の格子状マップに写像することにより、データのトポロジ的構造を保ちながら可視化する手法である。SOMは、**競合学習**（competitive learning）と呼ばれる学習規則に基づいており、入力パターンに最も近い出力ユニット（ニューロン）が「勝者」となり、その近傍のユニットとともに重みが更新される。競合学習はSOMに限らず、出力ニューロンが互いに競い合い、最も適合するものだけが活性化されるような学習機構を指す。SOMではこの競合に加えて、空間的な隣接性を重視した協調的な重み更新が行われる点が特徴的である。これにより、類似した入力はマップ上の近い位置に投影されるようになり、結果として**トポグラフィックマッピング** (topographic mapping) が実現される。

視覚野にはコラム構造が存在する．こうした構造は神経活動依存的な発生  (activity dependent development) により獲得される．本節では視覚野のコラム構造を生み出す数理モデルの中で，**自己組織化マップ** (self-organizing map) {cite:p}`Kohonen1982-mn`, {cite:p}`Kohonen2013-yt`を取り上げる．

自己組織化マップを視覚野の構造に適応したのは{cite:p}`Obermayer1990-gq` {cite:p}`N_V_Swindale1998-ri`などの研究である．視覚野マップの数理モデルとして自己組織化マップは受容野を考慮しないなどの簡略化がなされているが，単純な手法にして視覚野の構造に関する良い予測を与える．他の数理モデルとしては自己組織化マップと発想が類似している **Elastic net**  {cite:p}`Durbin1987-bp` {cite:p}`Durbin1990-xx` {cite:p}`Carreira-Perpinan2005-gy`　(ここでのElastic netは正則化手法としてのElastic net regularizationとは異なる)や受容野を明示的に設定した {cite:p}`Tanaka2004-vz`， {cite:p}`Ringach2007-oe`などのモデルがある．総説としては{cite:p}`Das2005-mq`，{cite:p}`Goodhill2007-va` ，数理モデル同士の関係については{cite:p}`2002-nm`が詳しい．

自己組織化マップでは「抹消から中枢への伝達過程で損失される情報量」，および「近い性質を持ったニューロン同士が結合するような配線長」の両者を最小化するような学習が行われる．包括性 (coverage) と連続性 (continuity) のトレードオフとも呼ばれる {cite:p}`Carreira-Perpinan2005-gy` (Elastic netは両者を明示的に計算し，線形結合で表されるエネルギー関数を最小化する．Elastic netは本書では取り扱わないが，MATLAB実装が公開されている
<https://faculty.ucmerced.edu/mcarreira-perpinan/research/EN.html>) ． 連続性と関連する事項として，近い性質を持つ細胞が脳内で近傍に存在するような発生/発達過程を**トポグラフィックマッピング (topographic mapping)** と呼ぶ．トポグラフィックマッピングの数理モデルの初期の研究としては{cite:p}`Von_der_Malsburg1973-bz` {cite:p}`Willshaw1976-zo` {cite:p}`Takeuchi1979-mi`などがある．

発生の数理モデルに関する総説 {cite:p}`Van_Ooyen2011-fz`, {cite:p}`Goodhill2018-ho`

### 単純なデータセット
SOMにおける $n$ 番目の入力を $\mathbf{v}(t)=\mathbf{v}_n\in \mathbb{R}^{D} (n=1, \ldots, N)$，$m$番目のニューロン $(m=1, \ldots, M)$ の重みベクトル（または活動ベクトル, 参照ベクトル）を $\mathbf{w}_m(t)\in \mathbb{R}^{D}$ とする {cite:p}`Kohonen2013-yt`．また，各ニューロンの物理的な位置を $\mathbf{x}_m$ とする．このとき，$\mathbf{v}(t)$ に対して $\mathbf{w}_m(t)$ を次のように更新する．

まず，$\mathbf{v}(t)$ と $\mathbf{w}_m(t)$ の間の距離が最も小さい (類似度が最も大きい) ニューロンを見つける．距離や類似度としてはユークリッド距離やコサイン類似度などが考えられる．

$$
\begin{align}
&[\text{ユークリッド距離}]: c = \underset{m}{\operatorname{argmin}}\left[\|\mathbf{v}(t)-\mathbf{w}_m(t)\|^2\right]\\
&[\text{コサイン類似度}]: c  = \underset{m}{\operatorname{argmax}}\left[\frac{\mathbf{w}_m(t)^\top\mathbf{v}(t)}{\|\mathbf{w}_m(t)\|\|\mathbf{v}(t)\|}\right]
\end{align}
$$

この，$c$ 番目のニューロンを **勝者ユニット** (best matching unit; BMU) と呼ぶ．コサイン類似度において，$\mathbf{w}_m(t)^\top\mathbf{v}(t)$ は線形ニューロンモデルの出力となる．このため，コサイン距離を採用する方が生理学的に妥当でありSOMの初期の研究ではコサイン類似度が用いられている {cite:p}`Kohonen1982-mn`．しかし，コサイン類似度を用いる場合は $\mathbf{w}_m$ および $\mathbf{v}$ を正規化する必要がある．ユークリッド距離を用いると正規化なしでも学習できるため，SOMを応用する上ではユークリッド距離が採用される事が多い．ユークリッド距離を用いる場合，$\mathbf{w}_m$ は重みベクトルではなくなるため，活動ベクトルや参照ベクトルと呼ばれる．ここでは結果の安定性を優先してユークリッド距離を用いることとする．

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