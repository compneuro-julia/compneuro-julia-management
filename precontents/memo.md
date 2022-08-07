## bib

````
{cite:p}`2002-nm`

## 参考文献
```{bibliography}
:filter: docname in docnames
```
````

# memo
シナプス重みと対数正規分布はシナプスの結合部分の末尾にコラムとして入れる．重みの初期化と絡める．

# Neuromatch
https://compneuro.neuromatch.io/tutorials/intro.html

# 強化学習
- 状態$s$の集合を$\mathcal{S}$, 行動$a$の集合を$\mathcal{A}$とする．
- 報酬$r$の分布を$p(r| s, a)$とする．
- 状態遷移の分布を$p(s'| s, a)$とする．
- 状態$s$のときの行動$a$の方策 (policy) を$\pi (a| s)$とする．

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

## intro: Julia言語の使い方

## intro: 微分方程式

## 付録：Julia, MATLAB, Pythonの対応表

## マルコフ連鎖


## 確率微分方程式


## 学習とコスト関数
学習の定義⇒コスト関数の導入
主な学習：教師あり学習，教師なし学習，強化学習
- 教師あり学習
- 教師無し学習
- 強化学習

線形回帰で勾配法を含めて説明する．
線形回帰⇒Hebb学習⇒パーセプトロン？

## その他
- 冒頭に微分方程式と確率微分方程式の説明いれる
- ギブスサンプリングは先に紹介して後の章でmcmcやると書く
- LIFのresetは一種の境界条件
- Hebbと同様のことはカハールも考えていた
  - A history of spike-timing-dependent plasticity
- 学習の定義について⇒コスト関数の導入
- 指数関数型シナプスの項を書き直す．微分方程式による表現の$\tau_d$にtypo (tau_sが正解)
- 2020のneural compを引用
- control as inference

## LTP LTD
http://www.scholarpedia.org/article/BCM_theory
Postulate 1 states that plasticity will occur only in synapses that are stimulated presynaptically. This is what biologists refer to as synapse specificity. Synapse specificity has strong support for both LTP and LTD (Dudek and Bear, 1992). In addition this assumption is consistent with the observation that more presynaptic activity results in a higher degree of plasticity, although this might not be linear.

There is now substantial evidence both in hippocampus and neocortex (Dudek and Bear, 1992, Mulkey and Malenka, 1992, Artola and Singer, 1992, Kirkwood and Bear, 1994, Mayford et al., 1995) in support of postulate 2. There is significant evidence that active synapses undergo LTD or LTP depending on the level of postsynaptic spiking or depolarization in a manner that is consistent with the BCM theory, as shown in Figure 4.

A direct test of the postulate of the moving threshold -- that after a period of increased activity θM increases, promoting synaptic depression, while after a period of decreased activity θM decreases, promoting synaptic potentiation -- has been tested by studying LTD and LTP of layer III synaptic responses in slices of visual cortex prepared from 4-6 week-old light-deprived and control rats (Kirkwood et al., 1996). This experiment shows that in deprived animals θM is lower than in normal animals. In control slices from the hippocampus no change in θM is observed.

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

## 潜在変数モデル LVM(latent variable model)
- sparse coding
- boltzmann machine

## neural sampling
- This is a remarkable result: By simply injecting noise into the continuous-time dynamics normally used for MAP inference in sparse coding, we obtain a dynamical system that naturally samples from the desired posterior distribution (eq. 4). With , we recover the SSC dynamics above (eqs. 25-26) where  converges to the MAP estimate.
- https://ar5iv.labs.arxiv.org/html/2204.11150
- ベイズ線形回帰（厳密解） -> MCMC -> GSM


## ベイズ用語
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2939867/

https://med.uth.edu/nba/wp-content/uploads/sites/29/2019/04/correlations.pdf

https://bsd.neuroinf.jp/w/index.php?title=%E7%A5%9E%E7%B5%8C%E7%AC%A6%E5%8F%B7%E5%8C%96

### Expected utility
the average expected reward associated with a particular decision, α, when the state of the environment, y, is unknown. It can be computed by calculating the average of the utility function, U(α, y), describing the amount of reward obtained when making decision α if the true state of the environment is y, with regard to the posterior distribution, p(y|x), describing the degree of belief about the state of the environment given some sensory input, x: R(α) = ∫ U(α, y) p(y|x) dy

### Likelihood
the function specifying the probability p(x|y,M) of observing a particular stimulus x for each possible state of the environment, y, under a probabilistic model of the environment, M

#### 尤度
刺激$x$, 環境の状態を$y$, モデルを$M$としたときのの
$p(x|y,M)$

### Marginalization
the process by which the distribution of a subset of variables, y1, is computed from the joint distribution of a larger set of variables, {y1, y2} p(y1) = ∫ p(y1, y2) dy2. (This could be important if, for example, different decisions rely on different subsets of the same set of variables.) Importantly, in a sampling-based representation, in which different neurons represent these different subsets of variables, simply “reading” (e.g. by a downstream brain area) the activities of only those neurons that represent y1 automatically implements such a marginalization operation

### Maximum a posteriori (or MAP) estimate
in the context of probabilistic inference, it is an approximation by which instead of representing the full posterior distribution, only a single value of y is considered that has the highest probability under the posterior. (Formally, the full posterior is approximated by a Dirac-delta distribution, an infinitely narrow Gaussian, located at its maximum.) As a consequence, uncertainty about y is no longer representedMaximum likelihood estimateas the MAP estimate, it is also an approximation, but the full posterior is approximated by the single value of y which has the highest likelihood

### Posterior
the probability distribution p(y|x,M) produced by probabilistic inference according to a particular probabilistic model of the environment, M, giving the probability that the environment is in any of its possible states, y, when stimulus x is being observed

### Prior
the probability distribution p(y|M) defining the expectation about the environment being in any of its possible states, y, before any observation is available according to a probabilistic model of the environment, M

### Probabilistic inference
the process by which the posterior is computed. It requires a probabilistic model, M, of stimuli x and states of the environment y, containing a prior and a likelihood. It is necessary when environmental states are not directly available to the observer they can only be inferred from stimuli through inverting the relationship between y and x through Bayes’ rule: p(y|x,M) = p(x|y,M) p(y|M)/Z, where Z is a factor independent of y, ensuring that the posterior is a well-defined probability distribution. Note, that the posterior is a full probability distribution, rather than a single estimate over environmental states, y. In contrast with approximate inference methods, such as maximum likelihood or maximum a posteriori that compute single best estimates of y, the posterior fully represents the uncertainty about the inferred variables

### Probabilistic learning
the process of finding a suitable model for probabilistic inference. This itself can be viewed as a problem of probabilistic inference at a higher level, where the unobserved quantity is the model, M, including its parameters and structure. Thus, the complete description of the results of probabilistic learning is a posterior distribution, p(M|X), over possible models given all stimuli observed so far, X. Even though approximate versions, such as maximum likelihood or MAP, compute only a single best estimates of M, they still need to rely on representing uncertainty about the states of the environment, y. The effect of learning is usually a gradual change in the posterior (or estimate) as more and more stimuli are observed, reflecting the incremental nature of learning

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
