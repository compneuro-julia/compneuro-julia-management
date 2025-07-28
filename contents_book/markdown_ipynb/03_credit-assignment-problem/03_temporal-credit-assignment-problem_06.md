## 合成勾配学習によるBPTTの近似
DNI

https://pmc.ncbi.nlm.nih.gov/articles/PMC7105376/
https://www.nature.com/articles/s41467-018-03541-0

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