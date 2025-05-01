## 実時間再帰学習 (RTRL)
次に，実時間再帰学習（real-time recurrent learning; RTRL）\citep{williams1989learning} を用いた際の各パラメータの勾配計算を行う．RTRLではテンソル積およびテンソル縮約を使用するため，適宜第1章を参照してほしい．

前節と同様に，感度行列を $\mathbf{P}_t^{(\theta)}:=\dfrac{\partial \mathbf{h}_t}{\partial \theta}\in \mathbb{R}^{d \times |\theta|}$，即時的感度行列を $\tilde{\mathbf{P}}_t^{(\theta)}:=\dfrac{\partial \mathbf{h}_t}{\partial \theta_t}\in \mathbb{R}^{d \times |\theta|}$ とする．出力に関わるパラメータ $\mathbf{W}_{\mathrm{out}}$ および $\mathbf{b}_\mathrm{out}$ は状態 $\mathbf{h}_t$ に影響しないため，$\mathbf{P}_t^{(\theta)}=\tilde{\mathbf{P}}_t^{(\theta)}=\mathbf{0}\; (\theta\in\{\mathbf{W}_{\mathrm{out}}, \mathbf{b}_\mathrm{out}\})$ である．よって（即時的）感度行列は  $\theta \in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{b}_\mathrm{rec}\}$ において考える．即時的感度行列は、それぞれのパラメータに対応して次のように書き下すことができる：

$$
\begin{align}
\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{in}})} &= \alpha \cdot \mathbf{D}_f(\mathbf{u}_t) \otimes \mathbf{x}_t &&\left(\in \mathbb{R}^{d \times d \times n}\right) \\
\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{rec}})} &= \alpha \cdot \mathbf{D}_f(\mathbf{u}_t) \otimes \mathbf{h}_{t-1} &&\left(\in \mathbb{R}^{d \times d \times d}\right) \\
\tilde{\mathbf{P}}_t^{(\mathbf{b}_\mathrm{rec})} &= \alpha \cdot \mathbf{D}_f(\mathbf{u}_t) &&\left(\in \mathbb{R}^{d \times d}\right)
\end{align}
$$

ここで，$\mathbf{D}_f(\mathbf{u}_t):=\mathrm{diag}(f'(\mathbf{u}_t))$ とした．$\mathrm{diag}(\cdot)$ はベクトルの各成分を対角要素として並べた対角行列を生成する演算子である．また，$\otimes$ はテンソル積を意味する。例えば，$\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{in}})}$ は，対角行列 $\mathrm{diag}(f'(\mathbf{u}_t))$ を入力次元 $n$ の各要素に対してコピーし，コピーされた各行列に対応する $\mathbf{x}_t$ の各要素をスカラーとして掛け合わせた構造を持つ。各成分を明示的に記述すれば，次のようになる：

$$
\begin{equation}
\left(\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{in}})}\right)_{ijk} = \alpha\, f'(u_t^i)\, \delta_{ij}\, x_t^k
\end{equation}
$$

ここで，$u_t^i$ は $\mathbf{u}_t$ の第 $i$ 成分，$x_t^k$ は $\mathbf{x}_t$ の第 $k$ 成分，$\delta_{ij}$ は Kronecker のデルタ（$i = j$ のときに1，それ以外は0）である。なお，$\tilde{\mathbf{P}}_t^{(\mathbf{W}_{\mathrm{rec}})}$ についても同様の構造であり，$\mathbf{x}_t$ を $\mathbf{h}_{t-1}$ に置き換えればよい。

次に，状態遷移のヤコビ行列は

$$
\begin{equation}
\mathbf{J}_t := \dfrac{\partial \mathbf{h}_{t}}{\partial \mathbf{h}_{t-1}}=(1-\alpha)\; \mathbf{I} + \alpha\cdot \mathbf{D}_f(\mathbf{u}_t)\mathbf{W}_{\mathrm{rec}}
\end{equation}
$$ 

であるので，感度行列の更新則は $\mathbf{P}_t^{(\theta)} =\tilde{\mathbf{P}}_t^{(\theta)}  + \mathbf{J}_{t}\mathbf{P}_{t-1}^{(\theta)}$ より，

$$
\begin{alignat}{3}
\mathbf{P}_t^{(\mathbf{W}_{\mathrm{in}})} &=(1-\alpha)\mathbf{P}_{t-1}^{(\mathbf{W}_{\mathrm{in}})} &&+ \alpha \left[\mathbf{D}_f(\mathbf{u}_t) \otimes \mathbf{x}_t  + \mathbf{D}_f(\mathbf{u}_t)\mathbf{W}_{\mathrm{rec}}\,\tilde{\otimes}_1\,\mathbf{P}_{t-1}^{(\mathbf{W}_{\mathrm{in}})}\right]&&\in \mathbb{R}^{d \times d \times n} \\
\mathbf{P}_t^{(\mathbf{W}_{\mathrm{rec}})} &=(1-\alpha)\mathbf{P}_{t-1}^{(\mathbf{W}_{\mathrm{rec}})} &&+ \alpha \left[\mathbf{D}_f(\mathbf{u}_t) \otimes \mathbf{h}_{t-1} + \mathbf{D}_f(\mathbf{u}_t)\mathbf{W}_{\mathrm{rec}}\,\tilde{\otimes}_1\,\mathbf{P}_{t-1}^{(\mathbf{W}_{\mathrm{rec}})}\right]&&\in \mathbb{R}^{d \times d \times d} \\
\mathbf{P}_t^{(\mathbf{b}_\mathrm{rec})} &=(1-\alpha)\mathbf{P}_{t-1}^{(\mathbf{b}_\mathrm{rec})} &&+ \alpha \left[\mathbf{D}_f(\mathbf{u}_t) + \mathbf{D}_f(\mathbf{u}_t)\mathbf{W}_{\mathrm{rec}}\mathbf{P}_{t-1}^{(\mathbf{b}_\mathrm{rec})}\right]&&\in \mathbb{R}^{d \times d} \\
\end{alignat}
$$

と求められる．ここで，$\tilde{\otimes}_1$ は、左側の行列 (2階テンソル) と右側の3階テンソルに対し，3階テンソルの第1軸に沿って縮約を行う演算子である。ここでの演算結果は、3階テンソル $\mathbf{P}_{t-1}$ の第3軸に沿った各スライス $(\mathbf{P}_{t-1})_{::k}$ に対して行列積 $\mathbf{W}_{\mathrm{rec}} (\mathbf{P}_{t-1})_{::k}$ を並列に適用し、それらを第3軸方向に再構成した3階テンソルとなる。

出力層の誤差はBPTTと同様に次のように定義する：

$$
\begin{equation}
\boldsymbol{\delta}_t^{\mathrm{out}}
:=\frac{\partial \mathcal{L}_t}{\partial \mathbf{v}_t}
=\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\frac{\partial \mathbf{y}_t}{\partial \mathbf{v}_t}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\odot g'(\mathbf{v}_t)^\top\quad \in \mathbb{R}^{1\times d}
\end{equation}
$$

即時的損失の状態に対する勾配は

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{v}_t}\frac{\partial \mathbf{v}_t}{\partial \mathbf{h}_t}=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}}\quad \in \mathbb{R}^{1\times d}
\end{equation}
$$

であり，パラメータ $\theta \in\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{rec}},\mathbf{b}_\mathrm{rec}\}$ について $\frac{\partial \mathcal{L}_t}{\partial \theta}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}$ であるので，各パラメータの即時的勾配は次のように求まる：

$$
\begin{alignat}{2}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{\mathrm{in}}}&=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}}\,\tilde{\otimes}_1\,\mathbf{P}_t^{(\mathbf{W}_{\mathrm{in}})}&&\in \mathbb{R}^{d\times n}\\
\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{\mathrm{rec}}}&=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}}\,\tilde{\otimes}_1\,\mathbf{P}_t^{(\mathbf{W}_{\mathrm{rec}})}&&\in \mathbb{R}^{d\times d}\\
\frac{\partial \mathcal{L}_t}{\partial \mathbf{b}_{\mathrm{rec}}}&=\boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{W}_{\mathrm{out}}\mathbf{P}_t^{(\mathbf{b}_{\mathrm{rec}})}&&\in \mathbb{R}^{1\times d}\\
\end{alignat}
$$

また，出力に関わるパラメータの勾配はBPTTと同様に

$$
\begin{align}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{\mathrm{out}}}
&=\mathbf{h}_t\boldsymbol{\delta}_t^{\mathrm{out}}\\
\frac{\partial \mathcal{L}_t}{\partial \mathbf{b}_\mathrm{out}}
&=\boldsymbol{\delta}_t^{\mathrm{out}}
\end{align}
$$

ただし，境界条件として $\mathbf{P}_{0}=\mathbf{0}$ とする。この式を用いて，$\mathbf{P}_t$ を逐次的に求め，$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}$ を即時的に計算して $\mathbf{P}_t$ に乗じれば，$\frac{\partial \mathcal{L}_t}{\partial \theta}$ が求まる．

このように RTRL では時刻ごとに $\mathbf{P}_t^{(\theta)}$ を更新し，それを用いて逐次的に勾配を計算するため，オンライン学習が可能となる。

## RTRLとBPTTの生理学的実装の困難点
時空間的に局所
時間的に局所 (local)

過去向き方式はオンライン性が強く，一度に扱うパラメータ依存を１つの損失にまとめるため，リアルタイム更新が可能であるが，その分、「過去→現在」の微分を保持する大きなテンソル（感度行列）を圧縮する工夫が必要となる。未来向き方式は「現在→未来」の影響を直接扱うため，パラメータ感度の保持は不要だが，未来の損失を参照する逆伝播がオンラインでは難しく，しばしばトランケート（打ち切り）を伴う。  


損失に対する状態感度
状態に対するパラメータ感度

BPTTは


いずれの手法も，時系列モデルの状態更新則が  

$$
\mathbf{h}_t = F\bigl(\mathbf{h}_{t-1},\,\mathbf{x}_t;\,\theta\bigr)
$$  

のように，状態は過去から未来への一方向性を持つため，過去の状態を未来の状態で微分する操作 $\partial \mathbf{h}_{t-1}/\partial \mathbf{h}_t$ は常にゼロとなる．

$\frac{\partial \mathbf{h}_t}{\partial \theta} \in \mathbb{R}^{d \times |\theta|}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \in \mathbb{R}^{d\times d}$

状態感度 (state sensitivity) $\frac{\partial \mathbf{h}_t}{\partial \theta}$

RTRLはパラメータを保持できない．
BPTTは未来から過去へ戻る必要がある．

脳は過去の状態を全て保存して逆向きに再生することは困難である．
再活性化などで可能となっている部分もあるが，全ての状態を保存しておくのは難しい．

海馬においては状態の逆再生 (reverse replay) が行われることが報告されている．

https://pubmed.ncbi.nlm.nih.gov/16474382/
https://www.nature.com/articles/nature04587

https://pmc.ncbi.nlm.nih.gov/articles/PMC6013068/
https://www.science.org/doi/10.1126/science.ads4760
https://www.biorxiv.org/content/10.1101/2023.02.19.529130v4

## 適格度トレースによるRTRLの近似

RFRO (random feedback local online learning) \citep{murray2019local}

$$
\frac{\partial h_{j}(t)}{\partial W_{a b}} = (1 - \alpha) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
$$


$$
\frac{\partial h_{j} (t )}{\partial W_{a b}} = (1 - \alpha ) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
$$

Output error

$$
\begin{align}
\epsilon (t)=\mathbf{y}(t)-\hat{\mathbf{y}}(t)\\
\mathcal{L}=\frac{1}{2T}\sum_{t=1}^T \|\epsilon (t)\|^2
\end{align}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=-\frac{1}{T}\sum_{t=1}^T \mathbf{W}_{out}^\top \epsilon (t)\frac{\partial \mathbf{h}(t)}{\partial \mathbf{W}}
$$

Update rule
$$
\begin{align}
\Delta \mathbf{W}^{out}_t&=\eta \epsilon_{t} \mathbf{h}_t\\
\Delta \mathbf{W}_{rec}(t)&=\eta \mathbf{B}\epsilon(t) \mathbf{P}(t)\\
\Delta \mathbf{W}_{in}(t)&=\eta \mathbf{B}\epsilon (t) \mathbf{Q}(t)\\
\end{align}
$$

Eligibility trace $\mathbf{P}\in \mathbb{R}^{N_{rec}\times N_{rec}}, \mathbf{Q}\in \mathbb{R}^{N_{rec}\times N_{in}}$

$$
\begin{align}
\mathbf{P}_t&=\alpha f'(\mathbf{u}_t)\mathbf{h}_{t-1}^\top+\left(1-\alpha\right)\mathbf{P}_{t-1}\\
\mathbf{Q}_t&=\alpha f'(\mathbf{u}_t)\mathbf{x}_{t-1}^\top+\left(1-\alpha\right)\mathbf{Q}_{t-1}
\end{align}
$$

RFLOに記載

$$
\Delta \mathbf{W}_{\textrm{out}} = \frac{\eta}{T} \sum_{t=1}^T \epsilon(t) h(t)^\top 
$$

$$
\Delta \mathbf{W}_{\textrm{rec}} = \frac{\eta}{T} \sum_{t=1}^T \mathbf{W}_{\textrm{out}}^\top \epsilon(t) \frac{\partial h(t)}{\partial \mathbf{W}_{\textrm{rec}}}
$$

http://frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full

## 合成勾配学習によるBPTTの近似
DNI

Online Learning via Synthetic Gradients

Decoupled Neural Interfaces (DNI)
Jaderberg et al. (2017)

合成勾配，すなわち損失に対するパラメータの勾配を予測することと同一ではないが，
運動において，結果が分かる前に誤差を推定することがある．
この場合は，損失に対する状態の勾配である．

これはBellman方程式と同じである．

誤差や勾配を推定することで

DNI(B), DNI(lambda), 小脳・大脳連関

Decoupled Neural Interfaces（DNI）とは，深層ニューラルネットワークの層間における誤差逆伝播（backpropagation）の依存関係を緩和し，各層の重み更新を部分的に「分離」して行うことで，並列化および生体適合性を高めようとする学習手法である。従来の誤差逆伝播法では，出力層から入力層に向かって誤差信号を逐次的に伝搬させる必要があり，すべての中間層は前段の勾配情報を待ってから自らのパラメータ更新を行う。この逐次依存性は計算グラフの層深が深くなるほど同期のボトルネックとなり，特に生体ニューラル回路や大規模分散システムへの実装に際して問題となる。  

DNIはこの問題に対して，各中間層に「合成勾配（synthetic gradient）」と呼ばれる局所的な誤差信号を予測する小さな補助モデルを付与する。具体的には，ある層ℓの出力 $h^{(\ell)}$ を受けて，補助モデル $M^{(\ell)}$ がその後段で生じるであろう真の勾配 $\partial \mathcal{L}/\partial h^{(\ell)}$ を予測し，これを用いて層ℓの重みを即時に更新する。すなわち，真の勾配が上流から到着するのを待たず，予測された合成勾配 $\hat g^{(\ell)} = M^{(\ell)}(h^{(\ell)})$ に基づいてパラメータ $\theta^{(\ell)}$ を更新することで，学習プロセスの層間同期を解消する。  

合成勾配モデル $M^{(\ell)}$ は通常小規模な多層パーセプトロンで実装され，真の勾配が利用可能になった後にその予測を教師信号として自身も学習する。すなわち，補助モデルは損失関数  
$$
\mathcal{L}_{\rm synth}^{(\ell)} = \bigl\|\hat g^{(\ell)} - g^{(\ell)}\bigr\|^2
$$ 
を最小化するように訓練される。この二重学習構造により，各メインモデルの層は独立に,—「decoupled」— 自らの補助モデルから供給される勾配情報だけで重み更新できるため，層間の待ち時間が排除され，完全な非同期分散学習が可能となる。  

DNIのメリットは第一に計算効率の向上であり，層深ネットワークのスケーラビリティが改善される点である。各層は逐次的な勾配伝搬の待機を必要としないため，ハードウェアパイプラインや分散ノード間で並列に学習更新を実行できる。第二に生物的妥当性の向上が期待される。生体神経回路では長距離の逆伝播による誤差信号の伝送機構は実証されておらず，DNIは局所的予測によって学習信号を得る点で神経回路の活動様式に近い可能性を示す。  

一方で合成勾配の予測誤差が大きい場合，メインモデルの学習が不安定化するリスクがあるため，補助モデルの設計や真の勾配との整合性をいかに保つかが重要となる。また，補助モデル自身の追加パラメータがオーバーヘッドとなるため，メモリおよび計算コストのトレードオフを慎重に評価する必要がある。これらの課題に対しては，合成勾配の正則化や補助モデルの軽量化手法，さらには真の勾配とのハイブリッド学習スケジュールの導入などが提案されている。  

まとめると，Decoupled Neural Interfacesは誤差逆伝播の逐次的制約を局所予測によって解除し，同期なし非同期学習を可能にする枠組みであり，大規模分散学習および生物的学習メカニズムの解明に向けた有力な手法として注目されている。

accumulate BP(λ) アルゴリズムは，強化学習における accumulate TD(λ) に着想を得て，RNNの出力誤差に基づく**将来の勾配（合成勾配）** を，BPTT を用いずに逐次的かつオンラインに学習する手法である。本節では，このアルゴリズムの各ステップを時間順に追い，教科書的な流れで逐次的に解説する。

適格度トレース

---

### 準備：モデル構造と定義

- 時刻 $t$ における RNN の隠れ状態を $h_t$，RNN パラメータを $\Psi$，損失を $L_t$ とする。
- 合成勾配 $\hat{G}_t \approx \frac{\partial L_{>t}}{\partial h_t}$ を出力する**synthesiser** $g(h_t; \theta)$ を学習する。
- 目的：$g(h_t; \theta)$ が正しい未来勾配を予測できるように，$\theta$ を更新する。

---

### ステップ 0: 初期化

\[
\Psi \leftarrow \Psi_0,\quad \theta \leftarrow \theta_0,\quad h \leftarrow 0,\quad \partial h \leftarrow 0,\quad e \leftarrow 0
\]

- $h$：RNN の現在の隠れ状態
- $\partial h$：Jacobian（$\partial h_{t+1}/\partial h_t$）
- $e$：synthesiser の**eligibility trace**

---

### ステップ 1: 新しい入力を処理

\[
h' \leftarrow f(x_t, h; \Psi),\quad L \leftarrow \mathcal{L}(h', y_t)
\]

- 入力 $x_t$ を受けて，RNN は次の状態 $h'$ と出力を生成し，損失 $L$ を計算する。

---

### ステップ 2: 勾配のローカル成分を計算

\[
\partial h \leftarrow \frac{\partial h'}{\partial h},\quad \frac{\partial L}{\partial h'}\quad\text{および}\quad \frac{\partial h'}{\partial \Psi}
\]

- この時点で得られるのは現在の状態 $h_t$ における**局所損失** $L_t$ の勾配のみであり，将来損失 $L_{>t}$ の勾配はまだ得られない。

---

### ステップ 3: 時間差誤差（TD-error）を計算

\[
\delta_t := \left(\frac{\partial L}{\partial h'} + \gamma\,g(h'; \theta)\right)^\top \frac{\partial h'}{\partial h} - g(h; \theta)
\]

- 合成勾配によって将来の勾配を推定し，**誤差 $\delta_t$** として TD 誤差に類似した量を構成する。
- これは「現在の予測 $g(h; \theta)$」と「次の状態での推定値を反映したターゲット値」の誤差を表す。

---

### ステップ 4: eligibility trace を更新

\[
e \leftarrow \gamma\lambda\,\partial h\,e + \nabla_\theta g(h; \theta)
\]

- 時間方向に前向きに伝播する形で，$\theta$ に関する**パラメータごとのトレース**を更新する。
- $\lambda$ によって短期記憶と長期記憶の加重が決まる。

---

### ステップ 5: パラメータ更新

**Synthesiser（θ）の更新**：

\[
\theta \leftarrow \theta + \alpha\,\delta_t^\top e
\]

**RNN パラメータ（Ψ）の更新**：

\[
\Psi \leftarrow \Psi + \eta\,\left(\frac{\partial L}{\partial h'} + g(h'; \theta)\right)^\top \frac{\partial h'}{\partial \Psi}
\]

- ここでの合成勾配を含む勾配が $\Psi$ の更新にも使われるため，synthesiser が誤っていると RNN 自身も誤った方向に学習される点に注意が必要である。

---

### ステップ 6: 状態更新

\[
h \leftarrow h'
\]

- 状態を更新して次の時刻へ進む。

---

### 特徴とポイント

- **λ = 0**：一歩先のブートストラップ合成勾配を使う元の手法（Jaderberg et al., 2017）に相当。
- **λ = 1**：将来のすべての損失を反映した完全な勾配（理論的に BPTT と同等）に一致。
- **ただし BPTT 不要**：いかなる時刻にも「過去の状態に遡る必要がなく」、**逐次的かつオンラインで**更新が可能。

---

このように，accumulate BP(λ) は，TD(λ) の構造と学習理論に基づいて，BPTT の近似を計算的に軽量な形で実現する枠組みである。特に生物学的実装の観点からも，後方パスを用いず，forward trace と局所勾配のみで学習を行う点において有望とされている。

ステップ 2 における「ローカル勾配の計算」では，時刻 $t$ における RNN の状態遷移 $h_t \mapsto h_{t+1}$ および出力 $y_{t+1}$ に関して，以下の3つの勾配を計算する必要があります：

1. $\dfrac{\partial h_{t+1}}{\partial h_t}$  
2. $\dfrac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$  
3. $\dfrac{\partial h_{t+1}}{\partial \Psi}$  

以下ではそれぞれを順に展開する。

---

## 1. $\dfrac{\partial h_{t+1}}{\partial h_t}$：RNN状態のヤコビアン

RNN の状態更新は，一般に次のような形式をとる：

\[
h_{t+1} = f(x_t, h_t; \Psi)
\]

ここで，$f$ は例えば以下のような非線形関数であることが多い（tanh RNN の場合）：

\[
h_{t+1} = \tanh(W_{in} x_t + W_{rec} h_t + b)
\]

このとき，$h_t$ による $h_{t+1}$ のヤコビアンは：

\[
\frac{\partial h_{t+1}}{\partial h_t} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot W_{rec}
\quad \text{ただし} \quad a_t := W_{in} x_t + W_{rec} h_t + b
\]

ここで $\operatorname{diag}[v]$ はベクトル $v$ を対角成分に持つ対角行列である。

---

## 2. $\dfrac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}}$：ローカル損失の勾配

タスクにおける出力が $y_{t+1} = W_{out} h_{t+1}$ で，目標出力が $\hat{y}_{t+1}$，損失が MSE の場合：

\[
\mathcal{L}_{t+1} = \frac{1}{2} \| \hat{y}_{t+1} - y_{t+1} \|^2
= \frac{1}{2} \| \hat{y}_{t+1} - W_{out} h_{t+1} \|^2
\]

このとき，$h_{t+1}$ に関する損失勾配は：

\[
\frac{\partial \mathcal{L}_{t+1}}{\partial h_{t+1}} = -W_{out}^\top (\hat{y}_{t+1} - W_{out} h_{t+1})
\]

---

## 3. $\dfrac{\partial h_{t+1}}{\partial \Psi}$：パラメータに対する勾配

ここでは RNN パラメータ $\Psi = \{W_{in}, W_{rec}, b\}$ に対して偏微分をとる。

それぞれの成分について：

- $\dfrac{\partial h_{t+1}}{\partial W_{in}} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot x_t^\top$
- $\dfrac{\partial h_{t+1}}{\partial W_{rec}} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr] \cdot h_t^\top$
- $\dfrac{\partial h_{t+1}}{\partial b} = \operatorname{diag}\bigl[1 - \tanh^2(a_t)\bigr]$

これらはテンソル形式でまとめて記述されるか，各パラメータに対してベクトル形式で記録される。

---

## まとめ：すべての項の役割

ステップ 2 は合成勾配更新に必要な各種偏微分を局所的に計算するステップであり，すべて forward pass の情報のみに基づいて，かつ $t+1$ 時点までの情報だけで完結する。

したがってこのステップは完全にオンラインかつ BPTT 非依存であり，合成勾配の正確さと伝搬に必要な中間量を準備する要である。

## 摂動を用いた学習則
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439155/full