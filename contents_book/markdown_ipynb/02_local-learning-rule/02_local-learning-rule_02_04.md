#### 再帰型結合の追加
ここまで神経細胞間に順方向結合 (feedforward connection) しかない，すなわち電気的活動（あるいは情報）が順伝播する静的モデルを取り扱った．次に，時間発展とシナプス後細胞群の再帰的結合 (recurrent connection) も考慮した動的なモデルを考える．時刻 $t$ の活動を $\mathbf{x}_t \in \mathbb{R}^n, \mathbf{y}_t \in \mathbb{R}^m$ とし，順方向結合重みを $\mathbf{W} \in \mathbb{R}^{m\times n}$，再帰的結合重みを $\mathbf{M} \in \mathbb{R}^{m \times m}$ とすると\footnote{ここで重みの時間発展は無視しているが，オンライン学習を行う場合は考慮が必要である．}，シナプス後細胞群の離散時間における再帰的発火率モデルは次のように表される：

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

また，出力 $\mathbf{y}(t)$ を内部状態 $\mathbf{u}(t)$ を通じて間接的に記述する形式もある：

$$
\begin{align}
\tau \frac{\mathrm{d}\mathbf{u}(t)}{\mathrm{d}t} &= -\mathbf{u}(t) + \mathbf{W} \mathbf{x}(t) + \mathbf{M} \mathbf{y}(t) + \mathbf{b} \\
\mathbf{y}(t) &= f(\mathbf{u}(t))
\end{align}
$$

この2段階の構造では，$\mathbf{u}(t)$ を膜電位，$\mathbf{y}(t)$ を発火率と解釈することができ，より生理学的に妥当なモデルとなる．

連続時間モデルを数値計算するには離散化する必要がある．単純な離散化手法として，Euler近似を初めに定義した式に用いると，次のように記述できる：

$$
\begin{align}
\mathbf{y}_{t+1} &= \mathbf{y}_t + \frac{\Delta t}{\tau} \left[ -\mathbf{y}_t + f(\mathbf{W} \mathbf{x}_t + \mathbf{M} \mathbf{y}_t + \mathbf{b}) \right]\\
&= (1-\alpha)\cdot \mathbf{y}_t + \alpha \cdot f(\mathbf{W} \mathbf{x}_t + \mathbf{M} \mathbf{y}_t + \mathbf{b})
\end{align}
$$

ただし，$\alpha := \frac{\Delta t}{\tau}$ とした．

#### Wilson–Cowanモデル
前節で導入した連続時間発火率モデルは，WilsonとCowanによって提案された，縮約化された神経回路網の数理モデル（**Wilson–Cowanモデル**）に基づいている \citep{wilson1972excitatory, wilson1973mathematical, wilson2021evolution, chow2020before}．このモデルは，個々の神経細胞の発火活動を扱うのではなく，興奮性あるいは抑制性の神経細胞群における平均的な発火活動の時間変化を連続時間の微分方程式として記述するものである．そのため，Wilson–Cowanモデルは神経集団の力学を対象とする**集団モデル**（population model）の一種であり，特に大脳皮質などにおけるマクロな神経活動の時空間的構造を解析する目的で広く用いられている．元来のWilson–Cowanモデルは2つの変数からなる連立微分方程式で構成され，それぞれが興奮性および抑制性の神経集団の平均発火率を表している．

以下では興奮性 (excitatory) あるいは抑制性 (inhibitory) の細胞群が関わる変数にそれぞれ添え字 $\mathrm{E}$ あるいは $\mathrm{I}$ をつける．また表記を簡略化するため，添え字 $\alpha, \beta$ が $\mathrm{E}$ あるいは $\mathrm{I}$ を表すとする．


$$
\begin{aligned}
\tau_\mathrm{E} \frac{\mathrm{d}y_\mathrm{E}(t)}{\mathrm{d}t} &= -y_\mathrm{E}(t) + f_\mathrm{E} \left[w_\mathrm{EE} y_\mathrm{E}(t) - w_\mathrm{EI} y_\mathrm{I}(t) + x_\mathrm{E}(t) \right] \\
\tau_\mathrm{I} \frac{\mathrm{d}y_\mathrm{I}(t)}{\mathrm{d}t} &= -y_\mathrm{I}(t) + f_\mathrm{I} \left[w_\mathrm{IE} y_\mathrm{E}(t) - w_\mathrm{II} y_\mathrm{I}(t) + x_\mathrm{I}(t) \right]
\end{aligned}
$$

ここで，活性化関数 $f_\mathrm{E}, f_\mathrm{I}$ は各集団に応じた非線形性を表し，通常はシグモイド関数のような単調増加かつ飽和的な関数が用いられる\footnote{Wilson–Cowanモデルのより原型に近い形式では，活性化関数の前に $(1 - r_\alpha \mathbf{y}_\alpha(t))$ を乗じる．この項は，神経細胞の発火率が生理学的に上限をもつという事実，すなわち飽和的性質を数理モデルに明示的に組み込むことを意図して導入されたものである．ただし，$r_\alpha=0$ と設定した場合でも，モデルの定性的挙動には大きな差が見られないことが知られている \citep{wilson2021evolution}．したがって，本書では記述の簡明さを優先し，この飽和項は導入せずにWilson–Cowanモデルを扱うこととする．}．

次に，このWilson-Cowanモデルにおける1変数を1個の神経細胞群の代表値ではなく1個の神経細胞の活動と解釈し，並列化することで多細胞系へと拡張したモデルを導入する\footnote{元のWilson-Cowanモデルと同様に解釈し，多数の神経細胞群間での相互作用のモデルと解釈することも可能である．}．さらに，この拡張された形式が前節で述べた一般的な連続時間発火率モデルと一致する構造を持つことを明示し，両者の関係を明らかにする．

まず，各細胞群の活動を $\mathbf{y}_\alpha(t) \in \mathbb{R}^{n_\alpha}$ とする．ここで $n_\mathrm{E}=n_\mathrm{I}=1$ とすれば元の形のWilson–Cowanモデルとなる．時定数を $\tau_\alpha$，細胞群への入力を $\mathbf{x}_\mathrm{\alpha}(t) \in \mathbb{R}^{n_\alpha}$，細胞群間での結合行列を $\mathbf{W}_{\alpha \beta}\in \mathbb{R}^{n_\alpha\times n_\beta}$とする．
ただし，$\mathbf{y}_\alpha, \mathbf{W}_{\alpha \beta}$ の要素はすべて非負である．この場合，時間的に粗視化した多細胞形式のWilson–Cowan方程式は次のように定式化される：

$$
\begin{aligned}
\tau_\mathrm{E} \frac{\mathrm{d}\mathbf{y}_\mathrm{E}(t)}{\mathrm{d}t} &= -\mathbf{y}_\mathrm{E}(t) + f_\mathrm{E} \left[ \mathbf{W}_\mathrm{EE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{EI} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{E}(t) \right] \\
\tau_\mathrm{I} \frac{\mathrm{d}\mathbf{y}_\mathrm{I}(t)}{\mathrm{d}t} &= -\mathbf{y}_\mathrm{I}(t) + f_\mathrm{I} \left[ \mathbf{W}_\mathrm{IE} \mathbf{y}_\mathrm{E}(t) - \mathbf{W}_\mathrm{II} \mathbf{y}_\mathrm{I}(t) + \mathbf{x}_\mathrm{I}(t) \right]
\end{aligned}
$$

このモデルは，それぞれの神経集団が自己結合および他集団からの入力を受け取り，非線形な活性化関数を通じて平均発火率を変化させることを記述している．

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

なお，機能的なRNNの内部機構を明らかにする手法としても，固定点解析および安定性解析は有用である \citep{sussillo2013opening}．Sussilloらの解析に基づいて2つの例を紹介する．1つ目の例として，パルス入力の符号を記憶するフリップフロップ（flip-flop）課題を考える．この課題では，入力ごとに $+1$ または $−1$ のパルス信号が与えられ，RNNはその最新の入力値を出力として保持する．フリップフロップ課題を実行するRNNを固定点解析すると，各記憶状態が安定固定点として実装されており，状態遷移は鞍点を通じて実現されることが示された．このように，RNNが「記憶の遷移」をどのように表現しているかを，固定点空間の構造として記述することができる．2つ目の例として，正弦波のような周期的な動的パターンを生成する課題を考える．この課題では，RNNの出力は時間とともに変化し続けるため，状態空間の軌道は固定点から大きく離れる場合が多い．しかしながら，各周波数に対して対応する静的入力を与えた条件下でRNNの固定点を求め，その周囲の線形化を行うと，得られる線形系は振動的な不安定性（虚部をもつ複素固有値）を示す．その虚部の大きさは，目標とする正弦波の角周波数と一致しており，線形近似によってRNNの出力振動の周期性を的確に説明できることが示されている．従って，このような動的挙動であっても，固定点周辺の線形ダイナミクスが振動パターンの生成機構を支配していると解釈できる．なお，こうした機能的なRNNを訓練する方法に関しては主に第5章で詳解する．