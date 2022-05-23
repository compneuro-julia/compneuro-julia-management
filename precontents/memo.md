## 学習則
- ある損失を符号化するニューロンがいるとして（例えばreaching error），その損失を産み出した回路網全体を訓練することを考える．このとき，回路網全体をglobalに変化させる，という仮定を置く．これはcredit assingment problemとなる．
- このとき，損失を最小化させる，というのが勾配降下法である．問題は，勾配を神経回路網が得られるのか，ということである．
- 初めにBackpropagationを導入する．BPTTも導入する．その後backpropの問題点を挙げ，backpropを近似する手法を紹介する: A deep learning framework for neuroscience
- 脳内では，真の勾配が得られない場合でも，部分的に沿っているような近似勾配があれば最適化は可能である．
- feedback aligment -> burst dependent 
- contrastive hebbian
- random (with directed gradient)

## Hebb則の安定化
- Hebb則は相関ベースの学習とも言える．ただし，正のフィードバックにより不安定化する．‘fire together, wire together’（共に活動，共に結合）
- 恒常的可塑性 (synaptic scaling)により安定化しているという説がある．
> Turrigiano, Gina G. 2008. “The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses.” Cell 135 (3): 422–35.
- しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない．
> Zenke, Friedemann, Wulfram Gerstner, and Surya Ganguli. 2017. “The Temporal Paradox of Hebbian Learning and Homeostatic Plasticity.” Current Opinion in Neurobiology 43 (April): 166–76.

## Sparse coding
- threshold関数をreluに変えても機能する．
- 入力は-1, 1に正規化されているとよい (正規化前処理しておく？)．
- 閾値計算はdecorrelationを産み出す: Mechanisms of pattern decorrelation by recurrent neuronal circuits

## ELBO
ELBOを導出する．

$$
\begin{aligned}
\log p(\mathbf{x})&=\log \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z}\\
&=\log \int \frac{q(\mathbf{z})}{q(\mathbf{z})}p(\mathbf{x}, \mathbf{z}) d\mathbf{z}\\
&=\log \mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]
\end{aligned}
$$

ここで上に凸な（凹）関数 $f(\mathbf{x})$についてJensenの不等式

$$
\mathbb{E}[f(\mathbf{x})]\leq f(\mathbb{E}[f(\mathbf{x})])
$$

が成立するので，$f(\cdot)=\log(\cdot)$として

$$
\begin{aligned}
\log p(\mathbf{x})&=\log \mathbb{E}_{p(\mathbf{z})}\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]\\
&\geq \mathbb{E}_{p(\mathbf{z})}\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]\\
&=\mathbb{E}_{p(\mathbf{z})}\left[\log\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z})}\right]\\
&=\mathbb{E}_{p(\mathbf{z})}\left[\log p(\mathbf{x}|\mathbf{z})\right]+\mathbb{E}_{p(\mathbf{z})}\left[\log\frac{p(\mathbf{z})}{q(\mathbf{z})}\right]\\
&=\mathbb{E}_{p(\mathbf{z})}\left[\log p(\mathbf{x}|\mathbf{z})\right]-D_\mathrm{KL}\left[q(\mathbf{z})\Vert p(\mathbf{z})\right]\\
\end{aligned}
$$

となる．

## 記号の表記
https://chokkan.github.io/mlnote/notation.html

## 式変換における矢印
- https://tex.stackexchange.com/questions/349259/curved-arrow-describing-a-step-in-a-equation-derivation

- https://tex.stackexchange.com/questions/112570/curved-arrow-describing-a-step-in-a-mathematical-derivation

## Log-trick
以下の目的関数を最大化する$\theta$を求める．

$$
\max_\theta \mathbb{E}_{\mathbf{x}\sim p_{\theta}(\mathbf{x})}[f(\mathbf{x})]:=\max_\theta \int p_{\theta}(\mathbf{x})f(\mathbf{x}) d\mathbf{x}
$$

これをするためにはiterativeな手法を用いて，

$$
\theta_{t+1} =\theta_{t} + \eta_t \nabla_{\theta} \int p_{\theta}(\mathbf{x})f(\mathbf{x}) d\mathbf{x}
$$

とすればよい．しかし，この勾配を求めるのは難しい．そこで，log trick

$$
\nabla_{\theta} \log g(\theta)=\frac{\nabla_{\theta} g(\theta)}{g(\theta)}
$$

を用いる．

$$
\begin{aligned}
\nabla_{\theta} \int p_{\theta}(\mathbf{x})f(\mathbf{x}) d\mathbf{x}&= \int \nabla_{\theta} p_{\theta}(\mathbf{x})f(\mathbf{x}) d\mathbf{x}\quad\text{(勾配を積分の中に入れる)}\\
&=\int p_{\theta}(\mathbf{x}) \nabla_{\theta} \log p_{\theta}(\mathbf{x})f(\mathbf{x}) d\mathbf{x}\quad\text{(log trick))}\\
&=\mathbb{E}_{\mathbf{x}\sim p_{\theta}(\mathbf{x})}[\nabla_{\theta} \log p_{\theta}(\mathbf{x})f(\mathbf{x})]
\end{aligned}
$$

これにより，
$$
\theta_{t+1} =\theta_{t} + \eta_t \mathbb{E}_{\mathbf{x}\sim p_{\theta}(\mathbf{x})}[\nabla_{\theta} \log p_{\theta}(\mathbf{x})f(\mathbf{x})]
$$
として更新すればよい．

$\mathbb{E}_{\mathbf{x}\sim p_{\theta}(\mathbf{x})}[\nabla_{\theta} \log p_{\theta}(\mathbf{x})f(\mathbf{x})]$はモンテカルロ法で求める．

## 強化学習
### General concepts

Return $G_{t}$: future cumulative reward, which can be written in arecursive form

$$
\begin{aligned}
G_{t} &= \sum \limits_{k = 0}^{\infty} \gamma^{k} r_{t+k+1}\\
&= r_{t+1} + \gamma G_{t+1}
\end{aligned}
$$

where $\gamma$ is discount factor that controls the importance of future rewards, and $\gamma \in [0, 1]$. $\gamma$ may also be interpreted as probability of continuing the trajectory.
Value funtion $V_{\pi}(s_t=s)$: expecation of the return

$$
\begin{aligned}
V_{\pi}(s_t=s) &= \mathbb{E} [G_{t}| s_t=s, a_{t:\infty}\sim\pi]\\
& = \mathbb{E} [ r_{t+1} + \gamma G_{t+1}| s_t=s, a_{t:\infty}\sim\pi]
\end{aligned}
$$

With an assumption of Markov process, we thus have:

$$
\begin{aligned}
V_{\pi}(s_t=s) &= \mathbb{E} [r_{t+1} + \gamma V_{\pi}(s_{t+1})|s_t=s, a_{t:\infty}\sim\pi]\\
&= \sum_a \pi(a|s) \sum_{r, s'}p(s', r)(r + V_{\pi}(s_{t+1}=s'))
\end{aligned}
$$

### Temporal difference (TD) learning

With a Markovian assumption, we can use $V(s_{t+1})$ as an imperfect proxy for the true value $G_{t+1}$ (Monte Carlo bootstrapping), and thus obtain the generalised equation to calculate TD-error:

$$
\begin{aligned}
\delta_{t} = r_{t+1} + \gamma V(s_{t+1}) - V(s_{t})
\end{aligned}
$$

Value updated by using the learning rate constant $\alpha$:
$$

\begin{aligned}
V(s_{t}) \leftarrow V(s_{t}) + \alpha \delta_{t}
\end{aligned}
$$

(Reference: https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html)

## ボルツマンマシン
http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/joho-gakushu.html

## Memo
- collapse回避にcontrastive vs reguralized: Yann Lecun, A Path Towards Human-Level AI
- 連想記憶モデルとDenoising autoencoder
- 独立成分分析入門 ～音の分離を題材として～