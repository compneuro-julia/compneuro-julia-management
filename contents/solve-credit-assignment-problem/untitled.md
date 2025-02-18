STDPはlocal learning ruleに
local learning ruleと分類するのはどうなのか？

BP (spiral, zipser & anderson, MNIST classify, autoencoder)
FA・DFA・KP (Fashion MNIST classification)
Predictive coding
Perturbation learning (https://oumpy.github.io/blog/2022/02/directional_gradient_optimization.html)

BPTT
RTRL
Random Feedback (Murray, J. M. Local online learning in recurrent networks with random feedback. eLife 8, pii: e43299 (2019).)

SpikeProp
Surrogate Gradient
#BurstProp
e-prop (A solution to the learning dilemma for recurrent networks of spiking neurons)

Reservior computing (rate, spike)
---
### **Node Perturbation（ノード摂動）の勾配の期待値**（行列形式）

#### **1. 記号の定義**
- 入力ベクトル: $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$
- 重み行列: $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
- 出力ベクトル（ノイズなし）:  
  $$
  \mathbf{y} = \mathbf{W} \mathbf{x} \in \mathbb{R}^{d_{\text{out}}}
  $$
- 損失関数: $ \mathcal{L}(\mathbf{y}) $
- ノイズベクトル: $\boldsymbol{\xi} \in \mathbb{R}^{d_{\text{out}}}$ （各要素がゼロ平均、分散 $\sigma^2$ の摂動）

#### **2. 損失関数の変化量**
ノード摂動法では、出力にノイズを加えたものを用いて損失を計算する:

$$
\mathbf{y'} = \mathbf{y} + \boldsymbol{\xi} = \mathbf{W} \mathbf{x} + \boldsymbol{\xi}
$$

摂動後の損失は:

$$
\mathcal{L}' = \mathcal{L}(\mathbf{y}')
$$

損失の変化量は

$$
\Delta \mathcal{L} = \mathcal{L}' - \mathcal{L} = \mathcal{L}(\mathbf{y} + \boldsymbol{\xi}) - \mathcal{L}(\mathbf{y})
$$

損失関数を $\mathbf{y}$ に対して1次のテイラー展開すると、

$$
\mathcal{L}(\mathbf{y} + \boldsymbol{\xi}) \approx \mathcal{L}(\mathbf{y}) + \boldsymbol{\xi}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} \boldsymbol{\xi}^T H_\mathcal{L} \boldsymbol{\xi} + O(\|\boldsymbol{\xi}\|^3)
$$

ここで、$H_\mathcal{L} = \frac{\partial^2 \mathcal{L}}{\partial \mathbf{y} \partial \mathbf{y}^T}$ はヘッセ行列。

したがって、
$$
\Delta \mathcal{L} \approx \boldsymbol{\xi}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} \boldsymbol{\xi}^T H_\mathcal{L} \boldsymbol{\xi}
$$

#### **3. 重みの更新則**
ノード摂動法では、重み更新は
$$
\Delta \mathbf{W} \propto - \boldsymbol{\xi} \Delta \mathcal{L} \mathbf{x}^T
$$

したがって、
$$
\Delta \mathbf{W} \propto - \boldsymbol{\xi} (\boldsymbol{\xi}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} \boldsymbol{\xi}^T H_\mathcal{L} \boldsymbol{\xi}) \mathbf{x}^T
$$

#### **4. 期待値を取る**
摂動ベクトル $\boldsymbol{\xi}$ はゼロ平均、共分散行列 $\sigma^2 I$ なので、
$$
\mathbb{E}[\boldsymbol{\xi}] = \mathbf{0}, \quad \mathbb{E}[\boldsymbol{\xi} \boldsymbol{\xi}^T] = \sigma^2 I
$$

よって、更新則の期待値を取ると、
$$
\mathbb{E}[\Delta \mathbf{W}] \propto -\mathbb{E} [\boldsymbol{\xi} \boldsymbol{\xi}^T] \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T - \frac{1}{2} \mathbb{E}[\boldsymbol{\xi} \boldsymbol{\xi}^T H_\mathcal{L} \boldsymbol{\xi}] \mathbf{x}^T
$$

ここで、$ \mathbb{E}[\boldsymbol{\xi} \boldsymbol{\xi}^T H_\mathcal{L} \boldsymbol{\xi}] $ の期待値は高次の影響を含み、摂動が小さい場合には無視できるため、主に第1項が残る。

$$
\mathbb{E}[\Delta \mathbf{W}] \propto -\sigma^2 \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T
$$

つまり、ノード摂動法の重み更新の期待値は、損失関数の勾配に比例する。

---

### **Weight Perturbation（重み摂動）の勾配の期待値**（行列形式）

#### **1. 記号の定義**
- 重み行列: $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
- 摂動行列: $\boldsymbol{\epsilon} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ （各要素がゼロ平均、分散 $\sigma^2$）
- 出力ベクトル（摂動なし）: $\mathbf{y} = \mathbf{W} \mathbf{x}$
- 損失関数: $ \mathcal{L}(\mathbf{y}) $
- 摂動後の重み: $\mathbf{W'} = \mathbf{W} + \boldsymbol{\epsilon}$

#### **2. 損失関数の変化量**
$$
\mathbf{y'} = \mathbf{W'} \mathbf{x} = (\mathbf{W} + \boldsymbol{\epsilon}) \mathbf{x} = \mathbf{y} + \boldsymbol{\epsilon} \mathbf{x}
$$

摂動後の損失は
$$
\mathcal{L}' = \mathcal{L}(\mathbf{y}')
$$

損失の変化量は
$$
\Delta \mathcal{L} = \mathcal{L}' - \mathcal{L} = \mathcal{L}(\mathbf{y} + \boldsymbol{\epsilon} \mathbf{x}) - \mathcal{L}(\mathbf{y})
$$

$\mathbf{y}$ に関するテイラー展開を行うと、
$$
\mathcal{L}(\mathbf{y} + \boldsymbol{\epsilon} \mathbf{x}) \approx \mathcal{L}(\mathbf{y}) + (\boldsymbol{\epsilon} \mathbf{x})^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} (\boldsymbol{\epsilon} \mathbf{x})^T H_\mathcal{L} (\boldsymbol{\epsilon} \mathbf{x})
$$

したがって、
$$
\Delta \mathcal{L} \approx (\boldsymbol{\epsilon} \mathbf{x})^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} (\boldsymbol{\epsilon} \mathbf{x})^T H_\mathcal{L} (\boldsymbol{\epsilon} \mathbf{x})
$$

#### **3. 重みの更新則**
重み摂動法では、
$$
\Delta \mathbf{W} \propto -\boldsymbol{\epsilon} \Delta \mathcal{L}
$$

したがって、
$$
\Delta \mathbf{W} \propto -\boldsymbol{\epsilon} ((\boldsymbol{\epsilon} \mathbf{x})^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{1}{2} (\boldsymbol{\epsilon} \mathbf{x})^T H_\mathcal{L} (\boldsymbol{\epsilon} \mathbf{x}))
$$

#### **4. 期待値を取る**
$\boldsymbol{\epsilon}$ はゼロ平均、共分散 $\mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\epsilon}^T] = \sigma^2 I$ より、
$$
\mathbb{E}[\boldsymbol{\epsilon} (\boldsymbol{\epsilon} \mathbf{x})^T] = \sigma^2 \mathbf{x}^T
$$

よって、
$$
\mathbb{E}[\Delta \mathbf{W}] \propto -\sigma^2 \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T
$$

つまり、重み摂動法の重み更新の期待値は、損失関数の勾配に比例する。

---

### **結論**
$$
\mathbb{E}[\Delta \mathbf{W}] \propto -\sigma^2 \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T
$$

したがって、**Node Perturbation と Weight Perturbation の両方で、重みの期待値の更新が勾配に比例することが示された。**