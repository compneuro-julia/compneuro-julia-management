## 学習則
- ある損失を符号化するニューロンがいるとして（例えばreaching error），その損失を産み出した回路網全体を訓練することを考える．このとき，回路網全体をglobalに変化させる，という仮定を置く．これはcredit assingment problemとなる．
- このとき，損失を最小化させる，というのが勾配降下法である．問題は，勾配を神経回路網が得られるのか，ということである．
- 初めにBackpropagationを導入する．BPTTも導入する．その後backpropの問題点を挙げ，backpropを近似する手法を紹介する: A deep learning framework for neuroscience
- 脳内では，真の勾配が得られない場合でも，部分的に沿っているような近似勾配があれば最適化は可能である．
- feedback aligment -> burst dependent 
- contrastive hebbian
- random (with directed gradient)

## curiosity drivenの強化学習
- https://www.slideshare.net/takmin/curiosity-driven-exploration
- https://arxiv.org/abs/2205.10316
- Grid spaceでの探索　＋運動学習

## 潜在変数モデル
- sparse coding
- boltzmann machine

## neural sampling
- This is a remarkable result: By simply injecting noise into the continuous-time dynamics normally used for MAP inference in sparse coding, we obtain a dynamical system that naturally samples from the desired posterior distribution (eq. 4). With , we recover the SSC dynamics above (eqs. 25-26) where  converges to the MAP estimate.
- https://ar5iv.labs.arxiv.org/html/2204.11150

## Hebb則の安定化
- 線形変換は$w$と$x$を正規化すれば，$wx$でcosine類似度を計算していると言える．
- Hebb則は相関ベースの学習とも言える．ただし，正のフィードバックにより不安定化する．‘fire together, wire together’（共に活動，共に結合）
- 恒常的可塑性 (synaptic scaling)により安定化しているという説がある．
> Turrigiano, Gina G. 2008. “The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses.” Cell 135 (3): 422–35.
- しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない．
> Zenke, Friedemann, Wulfram Gerstner, and Surya Ganguli. 2017. “The Temporal Paradox of Hebbian Learning and Homeostatic Plasticity.” Current Opinion in Neurobiology 43 (April): 166–76.

## gap 結合
- ギャップ結合を介した伝搬には、受動的なものと能動的なものがある。受動的伝播では、ある細胞の膜電位は、活動電位（AP）を誘発することなく、隣接する細胞の膜電位に影響を与える。一方、能動的な伝搬では、ある細胞のAPが近隣の細胞のAPを誘発する。これは、心筋組織や神経系全体で起こる。**APの伝搬には理想的なギャップ結合のコンダクタンスがあることが実験的に知られており、コンダクタンスが弱くなったり強くなったりすると伝搬が阻害される**。
- https://arxiv.org/abs/2205.12185

## 行列分解 (matrix factorization)
- PCA
- NMF
- SVD

## 前処理
- ZCA

## winner take all
- softHebb

## モデルの探求
- 本書では様々なモデルが登場するが，それらはどこから着想されたのだろうか？
- トップダウン型とボトムアップ型．生理現象を微分方程式などの数式で表したもの（トップダウン？）
- ある機能を実現する工学由来のモデル．強化学習が代表例．その機能を脳も実現しているという考えから，工学的モデルを生体内で実現可能なように近似していくことでモデルを構築する．こうしたモデルは生理学的に妥当(biologically plausible)であるという．このため，本書では工学的モデルの紹介⇒近似モデルの紹介，という形式を取る場合が多い．
- どうすれば生理学的に妥当になるかというと：あるニューロンが他のニューロンの内部状態を直接必要としない，計算がlocalで完結する等（追記必要）
- 注意したいのは生理学的に妥当というのは「現時点で」という但し書きが概して付けられるということである．新しい機能が生体内で発見される可能性もあれば，過去可能だと思われていたことが生体内の条件では実現できない場合もある．

## 学習則の解明の先には？
- 完全に個人的な意見であることに注意．
- 工学的に神経系を置き換える．
- 補助人工心臓や人工内耳が成功しているのは何故か？機能がはっきりしているから．置き換えられる．
- 神経系を置き換えにくいのは何故か？脊髄とかはいけそう．
- 神経系の特徴は汎用性があり（各素子が大体同じ），機能を柔軟に変化可能であること（この辺，機能変化の論文を引用．盲人の視覚野など）．
- 現状のBCIとかも成功しているが，汎用性がない．BMIは脳をコンピュータとして見立ててこそだが，脳はコンピュータではない（アナロジーに関する研究を引用．表現を穏やかにする）．非同期処理
- 神経系の機能を明らかにすることは置き換えることを可能とする．
- もちろん，機械系で置き換えなくても再生医療等で置き換えるのも重要な方法である．

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