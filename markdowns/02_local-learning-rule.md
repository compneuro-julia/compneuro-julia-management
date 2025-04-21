# 第2章：発火率モデルと局所学習則
## 神経細胞の概要
#### 脳における細胞の種類と数
脳は膨大な数の細胞によって構成されており，主に**神経細胞** (neuron) と**グリア細胞** (glial cells)の二種類に分類される．ヒトの脳には約860億個の神経細胞が存在し\footnote{Azevedoらは4人の成人男性の死後脳に対して，等方性分画法 (isotropic fractionator, \citep{herculano2005isotropic}) を用い，神経細胞の数を推定した．等方性分画法ではまず，固定した脳組織を解離および懸濁し，核が同濃度になるように撹拌する．その後，核を染色した後に血球計算盤で核の濃度を数え，濃度に懸濁液の総量をかけて全体の核の数，すなわち細胞数を概算する．懸濁液中の一部を抗NeuN抗体で免疫染色し，NeuNを発現している核の数を数える．そして，NeuN陽性細胞の割合と全体の細胞数から，神経細胞と非神経細胞の数が概算できる．結果として，神経細胞は大脳に約163億個，小脳に約690億個，その他の脳領域（大脳基底核・間脳・脳幹等）に約69億個，脳全体に約860億個あると推定された \citep{azevedo2009equal}．しかし，この約860億個という数字は絶対的なものではないことに注意が必要である \citep{goriely2024eighty}．推定値の標準偏差は約81億個であり，各被験者間では約730億個から約990億個とばらつきがあった．また神経細胞のマーカーとしてNeuNが使用されたが，NeuN陰性の神経細胞も存在するため，大脳皮質以外の神経細胞数は過小評価されている．}，グリア細胞も同数またはそれ以上の数が含まれると推定されている \citep{azevedo2009equal, von2016search}．これらの細胞は非常に高密度に詰まっており，例えばヒト大脳皮質の一次視覚野には1mm$^3$あたりで約7.9万個の神経細胞が存在する \citep{garcia2024neuronal}．

#### 神経細胞の形態
神経細胞の形態は他の細胞と大きく異なり，**細胞体** (soma, cell body)，**樹状突起**(dendrite)，**軸索** (axon) という三つの主要構造からなる．細胞体には細胞核があり，タンパク質合成やエネルギー代謝など基本的な細胞機能が行われる．樹状突起は木の枝のように複雑に分岐した構造で，他の神経細胞からの入力（シナプス入力）を受け取る部位である．軸索は通常1本の細長い突起であり，細胞体で統合された情報を他の神経細胞へと送る電気信号を伝導する．軸索の起始部には，細胞体との接合部である**軸索小丘**（あるいは軸索起始円錐; axon hillock）が存在し，それより遠位の領域は**軸索初節**（axon initial segment, AIS） と呼ばれる．AISにはトリガー帯 (trigger zone)と呼ばれる，電気信号の発生，すなわち**活動電位**（詳細は後述）の出発点として極めて重要な役割を果たす部位が存在する．AISには電位依存性ナトリウムチャネルが高密度に存在し，膜電位が閾値を超えると活動電位がここで生成される．AISは，ニューロンが入力を受けて出力すべきかを判断する**意思決定点**とも呼ばれる．

軸索の先端には軸索終末 (nerve terminal) があり，シナプスを形成している膨大部はシナプス前終末あるいはブトン (synaptic bouton) と呼ばれる．ブトンは**樹状突起スパイン** (dendritic spine) と結合している．

#### 神経細胞の電気的活動
神経細胞は，主に**電気的活動**によって情報を処理・伝達する．この活動は，細胞膜を挟んだ**イオンの移動**に基づいており，特に**イオンチャネル**と**イオントランスポータ**の働きが重要である．神経細胞の膜は静止時に内側が負に帯電しており，この状態は主に**ナトリウム・カリウムポンプ**（Na⁺/K⁺ ATPase）によって維持される．外部からの入力によって膜電位が上昇し，ある**閾値**（ただし一定ではない）を超えると，AISに存在する電位依存性ナトリウムチャネルが開き，ナトリウムイオンの流入によって膜が急激に脱分極する．この過程で生じる電位変化が**活動電位** (action potential) あるいは **スパイク** (spike) と呼ばれる信号であり，軸索を伝導して末端まで到達する．活動電位が発生することを**発火** (firing) とも呼ぶ．スパイクの後には，一時的に再発火が困難となる**不応期**（refractory period）が存在し，これにより信号の一方向性が保たれ，連続的なスパイクの発生頻度が制御される．

活動電位は最終的に**シナプス**（synapse）に到達し，次の細胞に情報を伝える．この伝達には大きく分けて二種類のシナプスがある．**化学シナプス**では，活動電位の到達により**シナプス小胞**が開口放出 (exocytosis) し，内部に蓄えられた**神経伝達物質**（neurotransmitter）が細胞間隙に放出される．この物質は次の細胞の**受容体**（receptor）に結合し，膜電位を変化させる．膜電位が脱分極方向に変化する場合は**興奮性シナプス後電位** (excitatory postsynaptic potential; EPSP)，過分極方向であれば**抑制性シナプス後電位**（ inhibitory postsynaptic potential; IPSP）と呼ばれる．一方，**電気シナプス**では**ギャップ結合**を通じてイオン電流が直接隣の細胞に流れ，より高速で同期的な通信が可能である．なお，シナプスで繋がる2つの細胞を，伝達の流れに即してシナプス前細胞 (pre-synaptic cell)およびシナプス後細胞 (post-synaptic cell) と呼ぶ．

#### 神経細胞の種類
神経細胞はその形態や機能，伝達物質の種類により多くのサブタイプ（subtype）に分類されるが，最も基本的な区別は**興奮性ニューロン**（excitatory neuron）と**抑制性ニューロン**（inhibitory neuron）である．興奮性ニューロンは主にグルタミン酸（glutamate）を放出し，標的細胞を脱分極に導いて興奮させる．抑制性ニューロンは主にGABA（γ-アミノ酪酸, gamma-aminobutyric acid）あるいはグリシン (glycine) を放出し，標的の膜電位を過分極させて抑制する．皮質においては，神経細胞の約80%が興奮性，約20%が抑制性とされる．

特に大脳皮質や海馬において，興奮性ニューロンの代表的な形態として知られるのが**錐体細胞**（pyramidal neuron）である．錐体細胞は三角形に近い細胞体を持ち，1本の長い太い**尖端樹状突起**（apical dendrite）\footnote{尖端樹状突起の先端はいくつもの枝を持ち，房状分枝（tuft）と呼ばれる．}と複数の**基底樹状突起**（basal dendrites）を持つのが特徴である．これにより空間的に広く分布した入力を統合でき，かつ軸索はしばしば長距離にわたって他の皮質領域や皮質下構造に投射する．これらの細胞は大脳皮質では第5層や第3層に多く存在し，皮質内外の広範な情報伝達を担う．皮質回路において中心的な情報出力の担い手として，認知・運動・記憶などの高次機能に不可欠である．

神経細胞の伝達物質の一貫性に関しては，**Daleの法則**（Dale’s principle）が古くから知られている．この法則は，「一つの神経細胞はその全ての出力部位で同一の神経伝達物質を放出する」という原則である．たとえば錐体細胞はどのシナプスでもグルタミン酸を放出し，同様に抑制性ニューロンであればGABAを一貫して使用する．現在では，補助的な神経ペプチドや共放出物質の存在が知られており，Daleの法則は厳密には修正されているものの，「主たる伝達物質の一貫性」という点では今も有効な原理とされている．

このように，神経細胞はその構造，電気的性質，機能的分類において精緻な多様性と秩序を持ち，脳回路全体の動的バランスと情報処理を支えている．とりわけ，興奮性ニューロンと抑制性ニューロンの適切な協調は，神経活動の安定化と時間的精度の制御において決定的な役割を果たしている．

#### グリア細胞の種類
神経細胞が情報の受容・統合・出力といった処理の中心を担うのに対し，グリア細胞はかつて単なる支持組織 (糊, glia) として捉えられていたが，現在では神経細胞と並んで神経系の恒常性維持・可塑性制御・免疫応答において不可欠な役割を担う能動的な細胞群であると認識されている．

まず**アストロサイト**（astrocyte）は，中枢神経系において最も豊富なグリア細胞であり，星状の形態を持つ．アストロサイトは血管と神経細胞の間を仲介し，**血液脳関門**（blood-brain barrier; BBB）の形成，イオン濃度の調節，神経伝達物質の再取り込み，さらにシナプスの形成と除去の調整に関与する．神経回路の機能に対して能動的に影響を与える点で，単なる支持細胞という枠を超えた存在である．

次に，**オリゴデンドロサイト**（oligodendrocyte）は中枢神経系において**ミエリン鞘**（myelin sheath）を形成する細胞である．1個のオリゴデンドロサイトは複数の軸索に分岐を伸ばし，それぞれにミエリンを巻き付ける．ミエリンは絶縁体として機能し，**跳躍伝導**を可能にすることでスパイクの伝導速度を著しく高める．オリゴデンドロサイトは伝達速度の調整も行っており，スパイクタイミングの調節等に寄与している．

これに対し，**シュワン細胞**（Schwann cell）は末梢神経系（peripheral nervous system）に存在し，オリゴデンドロサイトと類似の役割を果たす．ただし，シュワン細胞は1つの細胞が1つの軸索の1セグメントのみにミエリンを形成するという点でオリゴデンドロサイトとは異なる．また，シュワン細胞は神経損傷後の再生過程の促進にも関与する．

最後に，**ミクログリア**（microglia）は中枢神経系内に存在する免疫細胞であり，発生学的には他のグリア細胞とは起源が異なる（造血系由来）．ミクログリアは脳内の異物の貪食（ファゴサイトーシス）やアポトーシス細胞の除去を担い，また炎症性サイトカインの分泌を通じて神経炎症応答を調節する．またミクログリアがシナプスの刈り込み（synaptic pruning）にも関与することが報告されており，神経回路の発達と可塑性にも寄与していると考えられている．

## 神経細胞の発火率モデル
神経細胞は複雑な構造と機能を有する特殊な細胞であるが，その基本的な働きを理解するためには，ある程度抽象化された単純なモデルの導入が有用である．本章から第5章までは，この目的のもとに形式ニューロンや発火率モデルといった抽象的な数理モデルを用いることとし，チャネル動態などを含む詳細な生物物理モデルについては第6章で扱う\footnote{神経細胞のモデル化においては，詳細なモデルから抽象モデルを導出する記述順も考えられるが，本書では，コードのまとまりを考慮して，先に単純なモデルから入り，徐々に生物物理学的に忠実なモデルへと発展させる構成とする．}．本章で取り上げる**発火率モデル**（firing rate model）は，神経細胞の発火活動を平均的な頻度，すなわち**発火率**（firing rate）として記述する枠組みであり，この連続的な発火率により情報を表現する形式を**発火率による符号化**（rate coding）と呼ぶ．

### 離散時間モデル
発火率モデルでは，シナプス前細胞の活動の重みづけ和に基づき，シナプス後細胞の出力が決定される．まず，時間を離散的に扱い，時間発展を考慮しない場合のモデルを導入しよう．$n$ 個のシナプス前細胞の活動をベクトル $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top \in \mathbb{R}^n$，シナプス重みを $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top \in \mathbb{R}^n$，バイアス項を $b \in \mathbb{R}$，シナプス後細胞の出力を $y \in \mathbb{R}$ とすると，離散時間発火率モデルは以下のように定式化される：

$$
\begin{equation}
y = f(\mathbf{w}^\top \mathbf{x} + b) = f\left(\sum_{i=1}^n w_i x_i + b\right)
\end{equation}
$$

ここで，$f(\cdot)$ は入出力関係を表す関数であり，**活性化関数**（activation function）または**伝達関数**（transfer function）と呼ばれる．生理学的観点からこの式を解釈すると，$\mathbf{x}$ はシナプス前細胞から送られる発火率（に比例する量），$\mathbf{w}$ は各シナプス結合の強度を反映した重みであり，その内積 $\mathbf{w}^\top \mathbf{x}$ はシナプス後細胞に流入する総電流に相当する\footnote{2つの神経細胞間は1つのシナプス結合しか繋がっていないわけではなく，冗長な結合が存在する．ここではそうした複数のシナプス結合によるシナプス後細胞への影響の総和を取ってシナプス重みとしている．}．バイアス項 $b$ は，発火閾値（神経細胞の興奮性などの電気的特性）や定常的な興奮性入力などを含む項として解釈される．出力 $y$ は，神経細胞の平均的な発火率（に比例する量）とみなすことができ，活性化関数 $f$ はこの電流入力に応じた発火頻度の変化を表す関数，すなわち**周波数–電流曲線** (F-I曲線, frequency-current curve) に対応する．このF-I曲線の具体的形状と導出については，第6章で詳しく扱う．

活性化関数には線形関数と非線形関数の両方が用いられるが，後者の方が生理的現象に即している．たとえば，活性化関数を恒等写像 $f(x) = x$ とした場合，このモデルは線形回帰と同型になり，こうしたモデルを**線形ニューロンモデル** (linear neuron model) と呼ぶ．一方，非線形な活性化関数としては，Heavisideの階段関数，符号関数，シグモイド関数，双曲線正接関数（tanh），ReLU（Rectified Linear Unit）関数などがある．

#### 形式ニューロンとパーセプトロン
この離散時間モデルは，McCullochとPittsによって1943年に提案された**形式ニューロン**（formal neuron, McCulloch–Pitts neuron）に起源を持つ \citep{mcculloch1943logical}．形式ニューロンでは，活性化関数として次のようなHeaviside関数が用いられる：

$$
\begin{equation}
\Theta(x) = \begin{cases}
1 & (x \geq 0) \\
0 & (x < 0)
\end{cases}
\end{equation}
$$

このモデルは出力が0か1のいずれかであり，発火率ではなくスパイクの発火の有無を二値的に表現するものである．したがって，形式ニューロンは「入力の重みづけ和がある閾値 $\theta \ (=-b)$ を超えるかどうか」によって出力を決定する**閾値判定器**として機能する．

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
\tau \frac{d\mathbf{y}(t)}{dt} = -\mathbf{y}(t) + f(\mathbf{W} \mathbf{x}(t) + \mathbf{M} \mathbf{y}(t)+\mathbf{b})
\end{equation}
$$

ここで，$\tau$ は時定数であり，神経細胞の応答の時間スケールを決定するパラメータである．このモデルにおいて第一項の $-\mathbf{y}(t)$ は，入力が存在しない場合に神経活動が自然に減衰する性質を記述している．また $\mathbf{x}(t)$ が一定であり，$\frac{d\mathbf{y}(t)}{dt}=0$ を満たす **平衡点** (equilibrium point, fixed point) $\mathbf{y}^*$ が存在するとき，平衡点は離散時間モデルの更新式と同様の形式 $\mathbf{y}^* = f(\mathbf{W}\mathbf{x} + \mathbf{M} \mathbf{y}^* + \mathbf{b})$ を満たす．

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
\tau \frac{d\mathbf{u}(t)}{dt} &= -\mathbf{u}(t) + \mathbf{W} \mathbf{x}(t) + \mathbf{M} \mathbf{y}(t) + \mathbf{b} \\
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
\tau_\mathrm{E} \frac{d\mathbf{y}_\mathrm{E}(t)}{dt} &= -\mathbf{y}_\mathrm{E}(t) + f_\mathrm{E} \left[ \mathbf{W}_\mathrm{EE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{EI} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{E}(t) \right] \\
\tau_\mathrm{I} \frac{d\mathbf{y}_\mathrm{I}(t)}{dt} &= -\mathbf{y}_\mathrm{I}(t) + f_\mathrm{I} \left[ \mathbf{W}_\mathrm{IE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{II} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{I}(t) \right]
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
\boldsymbol{\tau} \frac{d\mathbf{y}(t)}{dt} = -\mathbf{y}(t) + f(\mathbf{M} \mathbf{y}(t) + \mathbf{x}(t))
\end{equation}
$$

この形式にまとめることで，Wilson–Cowanモデルは，前節で紹介した一般的な連続時間発火率モデルの枠組みの中に位置づけることができる．なお，外部入力の構造をより詳細に反映したい場合には，$\mathbf{x}(t) \to \mathbf{W} \mathbf{x}(t) + \mathbf{b}$ と置換し，入力重みと定常項（バイアス）を明示的に導入することも可能である．

#### 抑制安定化ネットワークと安定性解析
Wilson–Cowanモデル等でモデル化される興奮性および抑制性神経細胞群が相互作用する神経回路網は，各シナプス結合強度のバランスが取れていなければ活動は不安定（発散あるいは静止）となる\footnote{シナプス結合強度のバランスを保ったモデルを学習させる手法として\citep{soldado2022paradoxical}などがある．}．抑制性神経集団により神経回路網全体の活動が安定化されている神経回路網（あるいはその状態）を**抑制安定化ネットワーク** (inhibition-stabilized network; ISN) と呼ぶ \citep{sadeh2021inhibitory}．

ISNでは，興奮性細胞間の結合が強いために本来は活動が発散してしまうところを，抑制性細胞の抑制があることでネットワークが安定化されている．このような状態で生じる現象が**逆説的効果** (paradoxical effect) であり \citep{tsodyks1997paradoxical}，これは抑制性細胞に興奮性刺激を加えると，逆に抑制性細胞群の活動が減少するという現象である．逆説的効果は抑制性細胞の活動が高まると興奮性細胞への抑制が強まり，結果として興奮性細胞からの抑制性細胞への入力が減少するために生じる．逆の過程として，抑制性細胞に抑制性刺激を加えると逆に活動が増加するという現象も逆説的効果と呼ばれる．このような逆説的効果が生理的に存在することを示した実験研究としては \citep{ozeki2009inhibitory,kato2017network,sanzeni2020inhibition} があり，ISNという枠組みが生体脳においても成立しうることを支持している．

Wilson–CowanモデルはISN状態を含むため，ISNの数理的性質を理論的に解析する際の基礎モデルとして広く用いられている．あるモデルがISNであるかどうかを調べるには，まず**固定点解析**（fixed point analysis）\footnote{安定性解析まで含めて固定点解析と呼ぶ場合もある．} により系の平衡点（固定点）を求める．次に，その平衡点が安定かどうかは，線形安定性解析（linear stability analysis）によって判定される \citep{strogatz2024nonlinear}．線形安定性解析は，具体的には，非線形モデルを平衡点まわりで線形化し，ヤコビ行列（Jacobian matrix）を導出したうえで，その固有値の実部の符号を調べれば，安定性の有無を判定できる．以下では，興奮性・抑制性細胞が統合された形式でのWilson-Cowan方程式を例として，安定性解析を行う．なお，外部入力 $\mathbf{x}$ は一定であるとし，全ての細胞の活性化関数 $f(\cdot)$ が同一であるとする．まず，平衡点 $\mathbf{y}^*$ に小さな摂動$\delta \mathbf{y}(t)$を加え，$\mathbf{y}(t) = \mathbf{y}^* + \delta \mathbf{y}(t)$ として元の微分方程式に代入する：

$$
\begin{equation}
\tau \frac{d}{dt} (\mathbf{y}^* + \delta \mathbf{y}) = -(\mathbf{y}^* + \delta \mathbf{y}) + f(\mathbf{M} (\mathbf{y}^* + \delta \mathbf{y}) + \mathbf{x})
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
ここで $\frac{d\mathbf{y}^*}{dt}=0$ かつ $-\mathbf{y}^*+f(\mathbf{M} \mathbf{y}^* + \mathbf{x})=0$ が成り立つことも踏まえ，$\mathcal{O}(\|\delta \mathbf{y}\|^2)$ を無視して整理すると，平衡点周囲の線形系が得られる：

$$
\begin{align}
\tau \frac{d\delta \mathbf{y}}{dt} &= -\delta \mathbf{y} + \mathbf{D}_f \mathbf{M} \delta \mathbf{y}\\
&=\left(-\mathbf{I} + \mathbf{D}_f \mathbf{M} \right) \delta \mathbf{y} = \mathcal{J} \delta \mathbf{y}
\end{align}
$$

ここで，$\mathcal{J}:=-\mathbf{I} + \mathbf{D}_f \mathbf{M}$ を平衡点 $\mathbf{y}^*$ におけるヤコビ行列という．平衡点が漸近的に安定 (asymptotically stable) であるためには，ヤコビ行列 $\mathcal{J}$ の全ての固有値 $\lambda$ の実部が負，すなわち $\mathrm{Re}(\lambda) < 0$ を満たせばよい．この場合，ネットワークは安定となる．

なお，機能的なRNNの内部機構を明らかにする手法としても，固定点解析および安定性解析は有用である．例えば，パルス入力の符号を記憶するフリップフロップ（flip-flop）課題では，入力ごとに「+1」または「−1」のパルスが与えられ，RNNはその最新の入力値を出力として保持する．この課題を実行するRNNでは，各記憶状態が安定固定点として実装されており，状態遷移は鞍点を通じて実現される \citep{sussillo2013opening}．このように，RNNが「記憶の遷移」をどのように表現しているかを，固定点空間の構造として記述することができる．一方，正弦波のような周期的な動的パターンを生成する課題では，RNNの出力は時間とともに変化し続けるため，状態空間の軌道は固定点から大きく離れる場合が多い．にもかかわらず，各周波数に対して対応する静的入力を与えた条件下でRNNの固定点を求め，その周囲の線形化を行うと，得られる線形系は振動的な不安定性（虚部をもつ複素固有値）を示す．その虚部の大きさは，目標とする正弦波の角周波数（radians/sec）と数値的に一致しており，線形近似によってRNNの出力振動の周期性を的確に説明できることが示されている．したがって，このような動的挙動であっても，固定点周辺の線形ダイナミクスが振動パターンの生成機構を支配していると解釈できる．なお，こうした機能的なRNNを訓練する方法に関しては主に第5章で詳解する．

#### 神経集団モデルと神経場モデル
Wilson–Cowanモデルと密接に関連し，より大域的・集団的な神経活動を記述する枠組みとして，**神経集団モデル**（neural mass model, neural population model）および**神経場モデル**（neural field model）があり，これらのモデルに関して簡単に触れておく．いずれも個々のニューロンの詳細な活動ではなく，神経細胞集団の平均的な膜電位や発火率の時間変化を対象とし，脳波などのマクロな神経活動を記述するための理論的枠組みを提供する．

神経集団モデルでは，皮質のマイクロカラムや局所回路といった小規模な神経集団を1ユニットとしてモデル化し，Wilson–Cowanモデルと同様に，平均発火率や膜電位のダイナミクスを扱う．神経集団モデルの例としては局所神経回路をモデル化したJansen-Ritモデル \citep{jansen1995electroencephalogram} や，てんかん活動の動態を記述するWendlingモデル \citep{wendling2002epileptic} などがある．

一方，神経場モデル（neural field model）では，神経活動を空間的に連続な関数として記述し，広範囲における神経活動の時空間的なダイナミクスを扱う \citep{coombes2014tutorial, cook2022neural}．神経場モデルはWilsonおよびCowan \citep{wilson1973mathematical}, Nunez \citep{nunez1974brain}, 甘利 \citep{amari1975homogeneous, amari1977dynamics} らの研究に基づいており，ここでは甘利による定式化（**甘利モデル**, Amari model）を簡単に説明する．甘利モデルでは，まず神経場の定義域 $\Omega$（一次元の皮質断面や二次元の皮質平面など）を考える．$\Omega$ における神経活動は以下のような積分–微分方程式によって与えられる：

$$
\begin{equation}
\tau \frac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w(x, x') f(u(x', t)) dx' + I(x,t)
\end{equation}
$$

ここで，$x, x' \in \Omega$ は神経場における位置を表す．$u(x,t)$ は位置 $x$ における時刻 $t$ の発火率，$w(x,x')$ は位置 $x'$ から $x$ への結合重み，$f(\cdot)$ は活性化関数，$I(x,t)$ は外部入力を表す．神経場モデルは，皮質進行波（cortical travelling waves）\sitep{muller2018cortical} 等の現象を理論的に説明する手段となる．

## Hebb則とシナプス可塑性
### Hebb則
神経回路はどのようにして自己組織化するのだろうか．1940年代にHebbにより提案された学習則は「細胞Aが反復的または持続的に細胞Bの発火に関与すると，細胞Aが細胞Bを発火させる効率が向上するような成長過程または代謝変化が一方または両方の細胞に起こる」というものであった \citep{Hebb1949-iv}．すなわち，発火に時間的相関のある細胞間のシナプス結合を強化するという学習則である．これを**Hebbの学習則** (Hebbian learning rule) あるいは**Hebb則** (Hebb's rule) という．Hebb則は（Hebb自身ではなく）Shatzにより"cells that fire together wire together"（共に活動する細胞は共に結合する）と韻を踏みながら短く言い換えられている \citep{Shatz1992-he}．

数式を用いてHebb則を表現してみよう。まず，発火率モデルを定義する．$n$個のシナプス前細胞と$m$個のシナプス後細胞の発火率をそれぞれ $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$ とし，シナプス前細胞と後細胞のあいだのシナプス結合強度を $\mathbf{W} \in \mathbb{R}^{m \times n}$ とすると，前細胞と後細胞の活動の関係は活性化関数 $f(\cdot)$ を用いて $\mathbf{y} = f(\mathbf{W}\mathbf{x})$ と表現できる。このとき，連続時間の形式において，一般化されたHebb則は次のように表される：

$$
\begin{equation}
\tau \frac{d\mathbf{W}}{dt} = \phi(\mathbf{y}) \varphi(\mathbf{x})^\top
\end{equation}
$$

ここで，$\tau$ は学習の時定数であり，その逆数 $\eta := 1/\tau$ は**学習率**（learning rate）と呼ばれ，学習の速さを決定するパラメータである。関数 $\varphi(\cdot)$ および $\phi(\cdot)$ は，それぞれシナプス前細胞および後細胞の活動に応じてシナプス重みの変化を決定する変換関数である。特に $\varphi(\cdot)$，$\phi(\cdot)$ を恒等写像（恒等関数）とした場合，Hebb則は次のように簡潔な形で書ける：

$$
\begin{equation}
\tau \dfrac{d\mathbf{W}}{dt} = \mathbf{y} \mathbf{x}^\top\quad \left(= (\textrm{後細胞の活動}) \cdot (\textrm{前細胞の活動})^\top\right)
\end{equation}
$$

このような単純な形式のHebb則を**線形Hebb則**と呼び，狭義にはこれをもってHebb則とすることが多い。一方で，$\varphi(\cdot)$，$\phi(\cdot)$ を非線形関数とした拡張形式は**非線形Hebb則**と呼ばれ，本章で取り扱う。

### シナプス可塑性とLTP・LTD
LTPの実験的発見  

このHebb則の神経生理学的な基盤を裏付けるものとして，1973年にBlissとLømoによってウサギの海馬において**長期増強**（Long-Term Potentiation, LTP）が発見された \citep{Bliss1973-vj}．彼らの実験では，海馬のSchaffer側枝からCA1錐体細胞への経路に高頻度の電気刺激を加えることで，その後のシナプス応答が長時間にわたって増強される現象が観察された．この持続的なシナプス強度の増加は，まさにHebb則に対応する生理的現象と見なされ，Hebbian plasticityの実体と考えられるようになった．LTPはグルタミン酸作動性シナプスで観察されることが多く，特にNMDA受容体が関与することで知られている．この受容体は膜電位依存的にMg²⁺ブロックが外れることにより，カルシウムイオン（Ca²⁺）の流入を許し，それが下流のシグナル伝達を活性化してシナプス後部のAMPA受容体の増加や活性化を引き起こす．

一方，1980年代には**長期抑圧**（Long-Term Depression, LTD）という現象も発見された \citep{Dudek1992-nz}．これは，シナプス前ニューロンとシナプス後ニューロンが低頻度で同時活動した場合に，シナプスの伝達効率が長期にわたって減少する現象である．LTDもまた海馬や小脳などの領域で観察されており，この減弱はHebb則の反対の効果を示すものとして位置づけられる．特に，小脳における登上線維と平行線維の同時活動により引き起こされるLTDは，運動学習のモデルとして重要視されている．LTPと同様に，LTDにおいてもCa²⁺シグナリングが重要な役割を果たすが，その振幅や時間的プロファイルが異なっていることが，シナプス強化と抑圧の分岐をもたらすと考えられている．

これらの発見を通じて，Hebb則は単なる理論的仮説にとどまらず，シナプス可塑性という具体的な細胞メカニズムを通して，神経回路における学習と記憶の基盤であることが明らかにされた．

https://www.science.org/doi/10.1126/science.ads4706
https://pubmed.ncbi.nlm.nih.gov/24183021/
https://pubmed.ncbi.nlm.nih.gov/15450156/
https://pubmed.ncbi.nlm.nih.gov/26139370/
https://pubmed.ncbi.nlm.nih.gov/15450157/
https://pubmed.ncbi.nlm.nih.gov/18275283/
https://pubmed.ncbi.nlm.nih.gov/17332410/

NMDA, Ca
BDNF 
TrkB

TNF-\alpha

シナプス強度が増加するとは何が変化しているのか，とはspine sizeや受容体の数，

https://www.sciencedirect.com/science/article/pii/S0959438823001034?casa_token=ONmKH_RolAYAAAAA:DpV67Cj98lAp6pZk0_f-hTKCRbQ1wq20NPX_Fm1X0IF-eN6NhICDEAbgDIWdIPe3cLxCsadBJsRn

https://www.jneurosci.org/content/40/14/2828.abstract

### Hebb則の不安定性と修正Hebb則
Hebb則には問題点があり，シナプス結合強度が際限なく増大するか，あるいは消失するかという不安定性がある．これを数式で確認しておこう．前細胞と後細胞がそれぞれ1つの場合を考える．2細胞間の結合強度を$w\ (>0)$ とし，$y=wx$が成り立つとすると，Hebb則は$\dfrac{dw}{dt}=\eta yx=\eta x^2w$となる．この場合，$\eta x^2>1$ なら $\lim_{t\to\infty} w= \infty$, $\eta x^2<1$ なら $\lim_{t\to\infty} w= 0$ となる．当然，生理的にシナプス結合強度が無限大となることはあり得ないが，不安定なほど大きくなってしまう可能性があることに違いはない．このため，Hebb則を安定化させるための修正が必要とされた．

この問題に対して、さまざまな修正Hebb則 (modified hebbian rule) が提案されている．ここでは代表的な学習則である **CLO則**、**Oja則**、そして**BCM則**について説明する．

以下では 入力 $\mathbf{x}\in \mathbb{R}^n$ $y\in \mathbb{R}$ とし，シナプス結合 $\mathbf{w}\in \mathbb{R}^n$ を持つ，単一の神経細胞モデルし，単一の出力$y = f(\mathbf{w}^\top \mathbf{x})$ を持つを仮定する．

#### CLO則
視覚野のニューロンが経験により方向選択性を獲得し、かつ視覚刺激の遮断によってその選択性を失うといった生理的実験の結果を説明する理論として，Cooper, Liberman, Ojaにより閾値制約付き受動的可塑性 (threshold passive modification) と呼ばれる形式の学習則が提案された \citep{Cooper1979-wz}．この学習則は提案者の頭文字を取って，**CLO則** (CLO rule) と呼ばれる．CLO則は、Hebb則に対して出力の大きさに応じた閾値的な修正と重みの減衰項（忘却）を加えることにより、選択性の獲得と重みの安定性の両立を目指した学習則である。

CLO則は、出力 $y$ が可塑性閾値 $\theta_M$ や出力飽和値 $\theta_\mathrm{max}$ によって区分される3つの範囲で異なる更新を行う。CLO則の元の形式は離散時間での更新則であるが，表記を統一するため，連続時間でのCLO則を示す：

$$
\frac{d\mathbf{w}}{dt} =
\begin{cases}
- \lambda \mathbf{w} & (y \geq \theta_{\max}) \\
- \lambda \mathbf{w} + \eta_+ (\theta_{\max} - y) \mathbf{x} & (\theta_M \leq y < \theta_{\max}) \\
- \lambda \mathbf{w} - \eta_- y \mathbf{x} & (y < \theta_M)
\end{cases}
$$

ここで，$\lambda \geq 0$ は重みの減衰（leak）の度合いを決める定数であり，$\eta_+, \eta_-$ はそれぞれLTP・LTDに対応する学習率である．このように、適度な出力のときにのみ強化が起こり、過剰な出力では学習が停止し、出力が小さすぎる場合には抑制が起こるという、三相性の学習則が構築される。これにより、各ニューロンは特定の入力パターンに対してのみ強い応答を示すようになり、他のパターンには反応しなくなる。これは方向選択性や空間選択性のような感覚特異性（specificity）の獲得を数理的に説明する。CLO則はLTPに加えてLTDも組み込み，重みの減衰項も加えているため，不安定性はHebb則よりも低減されている．一方で，複雑で不連続な三相性の学習則を持ち，閾値も固定であるという欠点があった．

#### BCM則
CLO則を踏まえて，Bienenstock, Cooper, Munroにより提案された**BCM則**（Bienenstock–Cooper–Munro則）は、Hebb則の不安定性の修正に加えて、生理学的可塑性の双方向性（LTPとLTD）を統一的に記述することを目指した学習則である \citep{Bienenstock1982-km} \citep{Cooper2012-ec}．BCM則の核となるのは、出力活動の履歴に応じて変化する**修正閾値**（modification threshold）$\theta_m$の導入である。BCM則は次のように表される：

$$
\frac{d\mathbf{w}}{dt} = \eta_w \, \mathbf{x} \, y (y - \theta_m)
$$

関数$\phi$は$\phi(y, \theta_m)=y(y-\theta_m)$などとする．非線形Hebb則の一種である．また $\theta_m:=\mathbb{E}[y^2]$は閾値を決定するパラメータ，**修正閾値** (modification threshold) である．$\theta_m$ は活動履歴に基づいて動的に変化し、たとえば以下のように定義される：

$$
\frac{d\theta_m}{dt} = \eta_\theta (y^2 - \theta_m)
$$

この構造により、出力 $y$ が $\theta_m$ を超えるときにはシナプスが強化され（LTP）、逆に $y < \theta_m$ のときには弱化（LTD）される。このように、BCM則は同一の数式の中でHebbian強化とAnti-Hebbian抑制を両立させている。また、この動的閾値 $\theta_m$ は、ニューロンが自らの「活動水準の平均」を内部的に学習していく仕組みであり、これにより入力空間に対する**選択的な応答性**が獲得される。これは、視覚野ニューロンの方位選択性など、実際の神経生理学的観測とも整合する

#### Oja則
Hebb則を安定化させる別のアプローチとして，結合強度を正規化するという手法が考えられる．学習率を $\eta$ とすると，$\mathbf{w}\leftarrow\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とすれば正規化できる．ここで，$h(\eta):=\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とし，$\eta=0$においてTaylor展開を行うと，

$$
\begin{align}
h(\eta)&\approx h(0) + \eta \left.\frac{dh(\eta^*)}{d\eta^*}\right|_{\eta^*=0} + \mathcal{O}(\eta^2)\\
&=\frac{\mathbf{w}}{\|\mathbf{w}\|} + \eta \left(\frac{\mathbf{x}y}{\|\mathbf{w}\|}-\frac{y^2\mathbf{w}}{\|\mathbf{w}\|^3}\right)+ \mathcal{O}(\eta^2)
\end{align}
$$

ここで $\|\mathbf{w}\|=1$ として，1次近似すれば $h(\eta)\approx \mathbf{w} + \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)$ となる．重みの変化が連続的であるとすると，

$$
\begin{equation}
\frac{d\mathbf{w}}{dt} = \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)
\end{equation}
$$

として重みの更新則が得られる．これを**Oja則** (Oja's rule) と呼ぶ \citep{Oja1982-yd}．こうして得られた学習則において$\|\mathbf{w}\|\to 1$となることを確認しよう．

$$
\begin{equation}
\frac{d\|\mathbf{w}\|^2}{dt}=2\mathbf{w}^\top\frac{d\mathbf{w}}{dt}= 2\eta y^2\left(1-\|\mathbf{w}\|^2\right)
\end{equation}
$$

より，平衡状態 $\frac{d\|\mathbf{w}\|^2}{dt}=0$ において，$\|\mathbf{w}\|= 1$となる．

### 非線形Hebb学習
出力$\mathbf{y}$に非線形関数$g(\cdot)$を適用し，$\mathbf{y}\to g(\mathbf{y})$として置き換えることで非線形Hebb学習となる\citep{Oja1997-hr}\citep{Brito2016-mx}. 

ToDo: 詳細

### 恒常的可塑性
Oja則は更新時の即時的な正規化から導出されたものであるが，より時間スケールの長い過程として恒常的可塑性 (synaptic scaling)により安定化しているという説がある \citep{Turrigiano2008-lm} \citep{Yee2017-fb}．しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない \citep{Zenke2017-el}

https://www.nature.com/articles/s41598-023-32410-0
https://www.cell.com/neuron/fulltext/S0896-6273(20)30188-4?uuid=uuid%3Afdd605ec-6489-4c89-b4c9-73e1eb8f1c1a

https://www.pnas.org/doi/full/10.1073/pnas.1421304111

https://www.sciencedirect.com/science/article/abs/pii/S095943880000091X?via%3Dihub
https://www.cell.com/trends/neurosciences/abstract/S0166-2236(98)01341-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0166223698013411%3Fshowall%3Dtrue

https://www.nature.com/articles/nrn1949

Synaptic scaling

Oja則と同様に，
実装上はノルムで割る（乗法的スケーリング）

### Hebb則の変分原理的導出
Hebb則は数学的に導出されたものではないが，神経回路のダイナミクスがある目的関数（エネルギー関数）を最適化するように設計されていると仮定すれば、Hebb則に対応する重み更新則が自然に導出される。このようなネットワークを**エネルギーベースモデル** (energy-based models) といい，次章で扱う．

エネルギーベースモデルでは、神経活動 $\mathbf{z}$ およびシナプス結合 $\mathbf{W}$ に対して、あるエネルギー関数 $E(\mathbf{z}, \mathbf{W})$ を定義し、ダイナミクスがそのエネルギーを減少させるように構成される。すなわち、神経状態と重みの時間変化は、それぞれ次のようにエネルギーの勾配に基づいて与えられる：

$$
\begin{equation}
\frac{d \mathbf{z}}{dt}\propto-\left(\frac{\partial E}{\partial \mathbf{z}}\right)^\top,\quad\frac{d \mathbf{W}}{dt}\propto-\left(\frac{\partial E}{\partial \mathbf{W}}\right)^\top
\end{equation}
$$

このとき、逆に神経活動のダイナミクスのみが先に与えられている場合でも、それに整合するエネルギー関数を定義すれば、重みの更新則（すなわち学習則）を変分原理的に導出することができる。具体的には，神経細胞の活動ダイナミクスを積分することで神経回路のエネルギー関数 $E$ を導出し，さらに $E$ を重み行列で微分することでHebb則が導出できる \citep{Isomura2020-sn}．Hebb則の導出を連続時間線形ニューロンモデル $\dfrac{d\mathbf{y}}{dt}=-\mathbf{y}+\mathbf{W}\mathbf{x}$ を例にして考えよう（簡単のため $\tau=1$ とした）．ここで $\dfrac{\partial E}{\partial\mathbf{y}}:=-\left(\dfrac{d\mathbf{y}}{dt}\right)^\top$ となるようなエネルギー関数 $E(\mathbf{x}, \mathbf{y}, \mathbf{W})$ を仮定すると，

$$
\begin{equation}
E(\mathbf{x}, \mathbf{y}, \mathbf{W})=-\left(\int -\mathbf{y}+\mathbf{W}\mathbf{x}\ d\mathbf{y}\right)\propto\|\mathbf{y}\|^2-\mathbf{y}^\top \mathbf{W}\mathbf{x} \in \mathbb{R}
\end{equation}
$$

となる．これをさらに$\mathbf{W}$で微分すると，

$$
\begin{equation}
\dfrac{\partial E}{\partial\mathbf{W}}=-\mathbf{x}\mathbf{y}^\top\Rightarrow
\frac{d\mathbf{W}}{dt}=-\left(\frac{\partial E}{\partial \mathbf{W}}\right)^\top=\mathbf{y}\mathbf{x}^\top
\end{equation}
$$

となり，Hebb則が導出できる．

このような導出の仕方は、物理学の解析力学における**変分原理**（variational principle）あるいは **最小作用の原理**（principle of least action）の発想に基づいている．これらの原理においては、ある経路に沿って定義される**作用**（action）と呼ばれる量が極値（多くの場合、極小値）を取るような経路が、実際に物理系が取る経路として実現されるとされる。この考え方と同様に、神経活動やシナプス結合の変化も、あるエネルギー関数（または汎関数）の極値（神経回路が安定している場合には極小値）を実現するようなダイナミクスとして捉えることができる。もちろん、神経活動とシナプス結合が同一のエネルギー関数を最小化しているというのは大きな仮定である。しかし、このように神経回路の時間発展を最適化の視点から記述する立場は、神経可塑性を理論的に理解するための一つの有力な枠組みを提供し得る \citep{isomura2023experimental}．

## パーセプトロンの学習則
ここからは，Hebb則とその派生形による局所学習則を紹介する．前々節で紹介したパーセプトロン

このようなニューロンの抽象化を，分類問題に適用するために提案されたのが**パーセプトロン** (*perceptron*) である．パーセプトロンは，入力と重みの線形結合に対して符号関数（ステップ関数）を適用することにより，2クラス分類を実現する最も基本的な形式の**人工ニューロンモデル**である．

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

#### モデルの構造

入力ベクトル $\mathbf{x} \in \mathbb{R}^p$ に対して，重みベクトル $\mathbf{w} \in \mathbb{R}^{p+1}$（バイアス項 $w_0$ を含む）を用いて線形結合 $z = \mathbf{w}^\top \mathbf{x}'$ を計算する．ただし，$\mathbf{x}' := [1, x_1, x_2, \dots, x_p]^\top$ としてバイアス項を組み込んだ拡張ベクトルを用いる．

次に，活性化関数として**符号関数**を適用する：

$$
\hat{y} = \text{sign}(z) = \begin{cases}
+1 & (z \geq 0) \\
-1 & (z < 0)
\end{cases}
$$

これにより，出力 $\hat{y} \in \{-1, +1\}$ が得られ，2クラス分類を実現する．

#### 学習アルゴリズム：パーセプトロン則

パーセプトロンは**教師あり学習**に基づいて重み $\mathbf{w}$ を更新する．各ステップにおいて，予測と正解が一致していれば何も行わず，誤分類されたときにのみ以下のように重みを更新する：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)}
$$

ここで，$\eta > 0$ は学習率，$y^{(i)}$ は正解ラベルである．この更新則により，パーセプトロンは誤分類を修正する方向に重みを調整する．

もし訓練データが**線形分離可能**であるならば，このアルゴリズムは有限回の更新で必ず収束する（**パーセプトロン収束定理**）．ただし，データが線形分離不可能な場合は，収束せずに振動を続けることがある．

これは単純ではあるが，この微分不可能な関数による学習則は，現代的に**Straight-Through Estimator** (STE) と呼ばれる概念と同一である．STEの考えはスパイキングニューラルネットワークの学習や，ニューラルネットワークの量子化へと発展する．ここでは深く触れず，第7章で改めて紹介を行う．

## 主成分分析
Oja則を用いることで**主成分分析** (Principal component analysis; PCA) という処理をニューラルネットワークにおいて実現できる．

#### 主成分分析
主成分分析 (PCA) は，高次元のデータに内在する低次元の構造を抽出するための線形次元削減法である．この手法は，分散が最大となる方向にデータを射影することにより，元の情報をなるべく保ちながら次元を削減する．

まず，$n$ 個のサンプル $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$ が $d$ 次元の実ベクトル空間 $\mathbb{R}^d$ に属するとし，これらを列ベクトルとしてまとめたデータ行列を $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ とする．PCA では以下の手順を踏む．

PCA の目的は，情報損失（再構成誤差）を最小限に抑えながら，できるだけ少ない次元でデータを表現することである．この観点から，PCA は次の最適化問題の解とみなすこともできる：

$$
\max_{\mathbf{W}_m \in \mathbb{R}^{d \times m}} \operatorname{Tr}(\mathbf{W}_m^\top \mathbf{C} \mathbf{W}_m), \quad \text{s.t. } \mathbf{W}_m^\top \mathbf{W}_m = \mathbf{I}_m,
$$

ここで $\operatorname{Tr}(\cdot)$ はトレース演算，$\mathbf{I}_m$ は $m$ 次の単位行列である．この最適化問題の解は，共分散行列 $\mathbf{C}$ の上位 $m$ 個の固有ベクトルからなる直交行列 $\mathbf{W}_m$ である．

PCA はデータの冗長性を取り除くと同時に，ノイズの低減や可視化の手法としても広く応用される．また，線形変換であるため，計算効率も高いという特徴がある．

svdを用いて実装をする．


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
Oja則に複数の出力を持たせた場合であっても，出力が直交しないため，PCAの第1主成分しか求めることができない．**Sanger則** (Sanger's rule)，あるいは**一般化Hebb則** (generalized Hebbian algorithm; GHA)\footnote{あくまでSangerが「一般化」と呼んでいるだけで，Hebb則の一般化された形式ではない．} は，Oja則に**Gram–Schmidtの正規直交化法** (Gram–Schmidt orthonormalization) を組み合わせた学習則であり，次式で表される．

$$
\begin{equation}
\frac{d\mathbf{W}}{dt} = \eta \left[\mathbf{y}\mathbf{x}^\top - \mathrm{LT}\left(\mathbf{y}\mathbf{y}^\top\right) \mathbf{W}\right]
\end{equation}
$$

$\mathrm{LT}(\cdot)$は行列の対角成分より上側の要素を0にした下三角行列(lower triangular matrix)を作り出す作用素である．Sanger則を用いればPCAの第2主成分以降も求めることができる．

#### 非負主成分分析によるグリッドパターンの創発
内側嗅内皮質(MEC)にある**グリッド細胞** (grid cells) は六角形格子状の発火パターンにより自己位置等を符号化するのに貢献している．この発火パターンを生み出すモデルは多数あるが，**場所細胞** (place cells) の発火パターンを**非負主成分分析** (nonnegative principal component analysis) で次元削減するとグリッド細胞のパターンが生まれるというモデルがある \citep{Dordek2016-ff}．非線形Hebb学習を用いてこのモデルを実装しよう．なお，同様のことは**非負値行列因子分解** (nonnegative matrix factorization; NMF) でも可能である．

関数`HebbianPCA`の`func`引数に非線形関数を渡すことで非線形Hebb学習は実現できる．

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