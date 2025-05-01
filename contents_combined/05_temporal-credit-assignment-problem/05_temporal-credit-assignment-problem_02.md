## 経時的誤差逆伝播法 (BPTT)
**経時的誤差逆伝播法** (backpropagation through time; BPTT) \citep{werbos1988generalization,werbos1990backpropagation} を用いた際の，各パラメータの勾配を計算する．BPTTはRNNにおける時間方向の処理を空間的に展開してBPを適用するのと同じであるが，どのような処理が行われており，生理学的に妥当性のある処理であるのかを検証するために，ここでは具体的な勾配を計算する．

まず，出力層の誤差信号を

$$
\begin{equation}
\boldsymbol{\delta}_t^{\mathrm{out}}
:=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{v}_t}\right)^\top
=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\frac{\partial \mathbf{y}_t}{\partial \mathbf{v}_t}\right)^\top=\left(\frac{\partial \mathcal{L}_t}{\partial \mathbf{y}_t}\right)^\top\odot g'(\mathbf{v}_t)\quad \left(\in \mathbb{R}^{m}\right)
\end{equation}
$$  

と定義する。ここで $\odot$ は要素積 (Hadamard product) を表す。また中間層に逆伝播する誤差は時間方向の再帰関係から

$$
\begin{equation}
\boldsymbol{\delta}_t
:=\left(\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\right)^\top=\biggl(\underbrace{\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}}_{\mathclap{\substack{\text{現在時刻の}\\\text{直接寄与}}}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}}_{\mathclap{\substack{\text{次時刻以降への}\\\text{間接寄与}}}}\biggr)^\top
\quad \left(\in \mathbb{R}^{d}\right)
\end{equation}
$$  

が成り立つ．ここで直接寄与項は

$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}=\frac{\partial \mathcal{L}_t}{\partial \mathbf{v}_t}\frac{\partial \mathbf{v}_t}{\partial \mathbf{h}_t}=\left(\boldsymbol{\delta}_t^{\mathrm{out}}\right)^\top\mathbf{W}_{\mathrm{out}}
\end{equation}
$$

であり，間接寄与項は

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}&=\boldsymbol{\delta}_{t+1}^\top\left[(1-\alpha)\mathbf{I}_d+\alpha \frac{\partial f(\mathbf{u}_{t+1})}{\partial \mathbf{u}_{t+1}}\frac{\partial \mathbf{u}_{t+1}}{\partial \mathbf{h}_{t}}\right]\\
&=\left(1-\alpha\right)\boldsymbol{\delta}_{t+1}^\top +\alpha \left[\boldsymbol{\delta}_{t+1} \odot f'(\mathbf{u}_{t+1})\right]^\top \mathbf{W}_{\mathrm{rec}}
\end{align}
$$

である．ここで，$\delta_{t}^\mathrm{h} := \boldsymbol{\delta}_t \odot f'(\mathbf{u}_t)\ \left(\in \mathbb{R}^{d}\right)$ とすると，

$$
\begin{equation}
\boldsymbol{\delta}_t
=\mathbf{W}_{\mathrm{out}}^\top\boldsymbol{\delta}_t^{\mathrm{out}}+\left(1-\alpha\right)\boldsymbol{\delta}_{t+1} +\alpha \mathbf{W}_{\mathrm{rec}}^\top \delta_{t+1}^\mathrm{h}
\end{equation}
$$  

が成立する。ただし，境界条件として $\boldsymbol{\delta}_{T+1}=\mathbf{0}$ とする。パラメータの更新量を $\Delta \theta$ とすると，

$$
\Delta \theta \propto \sum_t \left(\dfrac{\partial \mathcal{L}}{\partial \theta_t}\right)^\top=\sum_t \left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}\right)^\top\boldsymbol{\delta}_t
$$

これらを用いて各重み行列の勾配を時刻方向に和をとる形で求める。

$$
\begin{alignat}{2}
\Delta \mathbf{W}_{\mathrm{rec}} &\propto \sum_t\left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{rec}}^t}\right)^\top
&&=\alpha \sum_t\boldsymbol{\delta}_{t}^\mathrm{h}\mathbf{h}_{t-1}^\top\\
\Delta \mathbf{W}_{\mathrm{in}} &\propto \sum_t\left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{in}}^t}\right)^\top
&&=\alpha \sum_t\boldsymbol{\delta}_{t}^\mathrm{h}\mathbf{x}_t^\top\\
\Delta \mathbf{b}_{\mathrm{rec}} &\propto \sum_t\left(\frac{\partial \mathcal{L}}{\partial \mathbf{b}_{\mathrm{rec}}^t}\right)^\top
&&=\alpha \sum_t\boldsymbol{\delta}_{t}^\mathrm{h}
\end{alignat}
$$



$$
\begin{alignat}{2}
\Delta \mathbf{W}_{\mathrm{out}}&\propto \sum_t\left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\mathrm{out}}^t}\right)^\top
&&=\sum_t \boldsymbol{\delta}_t^{\mathrm{out}}\mathbf{h}_t^\top\\
\Delta \mathbf{b}_{\mathrm{out}}&\propto \sum_t\left(\frac{\partial \mathcal{L}}{\partial \mathbf{b}_{\mathrm{out}}^t}\right)^\top
&&=\sum_t \boldsymbol{\delta}_t^{\mathrm{out}}\\
\end{alignat}
$$

以上が BPTT による重み更新の基本式である。BPの時と同様に，バッチ処理を考慮するため，転置の有無や行列積の順序は変化する．