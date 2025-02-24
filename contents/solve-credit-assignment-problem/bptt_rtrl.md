Backpropagation through time and the brain
https://www.sciencedirect.com/science/article/pii/S0959438818302009

RNN

状態
$$
\mathbf{h}(t+1)=\left(1-\frac{1}{\tau}\right)\mathbf{h}(t)+\frac{1}{\tau}f(\mathbf{W}\mathbf{h}(t)+\mathbf{W}_{in}\mathbf{x}(t+1)+\mathbf{b})
$$

出力は
$$
\mathbf{y}(t)=\mathbf{W}\mathbf{h}(t)
$$

BPTT (Backpropagation through time)
backpropagation through time (BPTT) (Rumelhart et al., 1985) in order to compare it with the learning rules presented above. The derivation here follows Lecun (1988).

RTRL (Real-time recurrent learning)
Williams RJ, Zipser D. 1989. A learning algorithm for continually running fully recurrent neural networks. Neural Computation 1:270–280. DOI: https://doi.org/10.1162/neco.1989.1.2.270



RFRO (Random feedback local online learning)


$$
\frac{\partial h_{j} (t )}{\partial W_{a b}} = (1 - \frac{1}{\tau} ) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \frac{1}{\tau} \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \frac{1}{\tau} \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}} ,
$$

----
## **BPTT（Backpropagation Through Time）とRTRL（Real-Time Recurrent Learning）の数式表現**

### **1. BPTT（Backpropagation Through Time）**
BPTTはRNN（Recurrent Neural Network）の学習において誤差逆伝播法（Backpropagation）を時間方向に適用する手法である．時系列データに対する損失関数の勾配を効率的に計算するために，時間展開したネットワーク上で逆伝播を行う．

#### **(1) 順伝播**
考えるRNNの状態更新と出力の式は、次のように書けます。

$$
\mathbf{h}_t = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\mathbf{y}_t = g(\mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y)
$$

ここで、
- $h_t$は時刻$t$の隠れ状態
- $x_t$は入力
- $y_t$は出力
- $W_h, W_x, W_y$はそれぞれの重み行列
- $b_h, b_y$はバイアス
- $f$は活性化関数（例: tanh, ReLU）
- $g$は出力関数（例: softmax）

#### **(2) 損失関数**
一般的な損失関数$\mathcal{L}$は、時系列全体の誤差の合計として定義されます。

$$
\mathcal{L} = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

ここで$\ell(y_t, \hat{y}_t)$は時刻$t$における損失。

#### **(3) 誤差逆伝播**
BPTTでは、勾配を時間方向に逆伝播させます。

##### **出力層の勾配**
出力層の重み$W_y$に関する勾配：

$$
\frac{\partial L}{\partial W_y} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial y_t} \frac{\partial y_t}{\partial W_y}
$$

##### **隠れ層の勾配**
隠れ層の重み$W_h$に関する勾配は、時間的な依存関係を考慮して連鎖律を適用します。

$$
\delta_t = \frac{\partial L}{\partial h_t} = \frac{\partial \ell_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} + \delta_{t+1} \frac{\partial h_{t+1}}{\partial h_t}
$$

重み$W_h$に関する勾配は以下のようになります。

$$
\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \delta_t \frac{\partial h_t}{\partial W_h}
$$

BPTTでは、計算量削減のために「一定の時間範囲（トランケーション）」のみを考慮する **Truncated BPTT** もよく用いられます。

---

### **2. RTRL（Real-Time Recurrent Learning）**
RTRLは、RNNの各時刻の重み更新をリアルタイムに行う学習アルゴリズムです。BPTTとは異なり、時間展開をせずに、各時刻での勾配を逐次更新します。

#### **(1) 隠れ状態の再掲**
$$
h_t = f(W_h h_{t-1} + W_x x_t + b_h)
$$

#### **(2) 隠れ状態の勾配伝播**
RTRLでは、すべての時刻で重み$W_h$に対する状態の変化率を直接追跡します。

$$
S_t = \frac{\partial h_t}{\partial W_h}
$$

この$S_t$を逐次的に更新する式は以下のようになります。

$$
S_t = f'(W_h h_{t-1} + W_x x_t + b_h) \left( W_h S_{t-1} + \frac{\partial h_t}{\partial W_h} \right)
$$

これを利用して、損失$L$に関する勾配を計算します。

$$
\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} S_t
$$

---

### **BPTTとRTRLの比較**
| 項目 | BPTT | RTRL |
|------|------|------|
| 勾配計算 | 時間展開後に逆伝播 | リアルタイムに勾配更新 |
| 計算コスト | 高い（長期依存の計算が可能） | 非常に高い（$O(T n^4)$の計算量） |
| メモリ使用量 | 長期依存を扱う場合多い | 少ない（即時更新） |
| 応用 | 深層学習で一般的（LSTM, GRU） | ほぼ使われない（計算コストの問題） |

通常、計算コストの問題でRTRLはほとんど使用されず、BPTTやTruncated BPTTが主流です。


##

A Practical Sparse Approximation for
Real Time Recurrent Learning

BPTT

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta_t}=\sum_{t=1}^T\left(\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\right)\frac{\partial \mathbf{h}_t}{\partial \theta_t}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}=\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}+\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}
$$


RTRL

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\frac{\partial \mathbf{h}_t}{\partial \theta}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}\left(\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}\right)
$$

$$
\frac{\partial \mathbf{h}_t}{\partial \theta}=\frac{\partial \mathbf{h}_t}{\partial \theta_t}+\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\frac{\partial \mathbf{h}_{t-1}}{\partial \theta}
$$

$$
\frac{d\mathcal{L}_t}{d \theta}=\frac{d\mathcal{L}_t}{d h_t}\frac{d h_t}{d \theta}
$$

$h_t=f(x_t, h_{t-1}, \theta)$を全微分すると，

$$
\frac{dh_t}{d\theta}=\frac{\partial h_t}{\partial h_{t-1}}\frac{d h_{t-1}}{d \theta}+\frac{\partial h_t}{\partial x_t}\frac{d x_t}{d \theta}+\frac{\partial h_t}{\partial\theta}=\frac{\partial h_t}{\partial h_{t-1}}\frac{d h_{t-1}}{d \theta}+\frac{\partial h_t}{\partial\theta}
$$

$x_t$は$\theta$に依存しない．全微分と偏微分に注意すると，


RFLOに記載

$$
\Delta \mathbf{W}_{\textrm{out}} = \frac{\eta}{T} \sum_{t=1}^T \epsilon(t) h(t)^\top 
$$

$$
\Delta \mathbf{W}_{\textrm{rec}} = \frac{\eta}{T} \sum_{t=1}^T \mathbf{W}_{\textrm{out}}^\top \epsilon(t) \frac{\partial h(t)}{\partial \mathbf{W}_{\textrm{rec}}}
$$
