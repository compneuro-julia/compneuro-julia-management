## 状態価値の推定
状態価値 $v_\pi(s)$ や行動価値 $q_\pi(s, a)$ は，環境との相互作用を通して推定を行う必要がある．まずは状態価値の推定について考えよう．以下では（方策 $\pi$ に従った際の）状態 $s$ の価値の推定値を $V(s)$ とする．また，終端時刻 $T$ が有限である場合のみを考える．

### モンテカルロ法
期待値を近似的に推定する手法として**モンテカルロ法** (Monte-Carlo method) がある．モンテカルロ法を用いると，一般に確率変数 $X$ および関数 $f$ がある場合，$\mathbf{E}[f(X)]$ の推定値 $\mu$ は，サンプル平均 $\mu=\frac{1}{N}\sum_{n=1}^N f(x_n)$ として与えられる．ただし，$x_n$ は$X$ の実現値（観測値）である．$x_n$ を全て保持せず，逐次的に（オンラインで）モンテカルロ推定を行う場合，

$$
\begin{equation}
\mu_{n}= \mu_{n-1}+\frac{1}{n} \left[f(x_n)-\mu_{n-1}\right]
\end{equation}
$$

と表される（$\mu_n$ は $n$ 回目の更新時の推定値である）．サンプル平均を取る手法は $X$ の分布 $p(X)$ が定常 (stationary) である場合はよいが，非定常 (non-stationary) である場合，すなわち $X$ の分布が時刻 $n$ に伴って変化する場合，過去と現在のサンプルに同様の重みを与えることは推定が悪くなる要因となりうる．このような非定常環境では$1/n$ の代わりに固定の学習率 $\alpha\ (0\leq \alpha \leq 1)$ を用い，現在のサンプルに大きな重みを与える，すなわち指数移動平均 (exponential moving average; EMA) を取る手法がより適している．

$$
\begin{equation}
\mu_{n}= \mu_{n-1}+\alpha \left[f(x_n)-\mu_{n-1}\right]
\end{equation}
$$

強化学習では，時間あるいは方策の変化に伴って状態価値や行動価値が変化する非定常環境を仮定することが多く，基本的には指数移動平均による推定を使用する．この手法を用いて状態価値 $v_\pi(s)$ を推定することを考えよう．$v_\pi(s)$ は状態 $s$ における 収益 $G_t$ の期待値であるため，1試行ごとに終端時刻まで軌道（エピソード）を記録し，各状態における $G_t$ を計算して，それにより推定値を次のように更新する方法が考えられる．

$$
\begin{equation}
V(s_t)\leftarrow V(s_t)+\alpha \left[G_t - V(s_t)\right]
\end{equation}
$$

強化学習の文脈では，この価値推定手法を指して（狭義の）モンテカルロ法と呼ぶ．モンテカルロ法には，$G_t$ が試行が終了するまで得られないという問題点がある．

### Bellman方程式とTD学習
モンテカルロ法はオフライン学習法であり、各試行が終了した後に、試行中に得られたすべての報酬列を記憶し、それに基づいて価値関数をまとめて更新する必要がある。このため、学習は試行の完了を待たねばならず、即時的な推定値の更新はできない。試行の途中で価値推定を逐次的に更新し、すなわちオンラインで学習を行うためには、各状態の価値を次の状態の価値と直接結び付ける関係式が必要となる。この発想に基づき、状態価値を再帰的に定義する**Bellman方程式** (Bellman equation) が導入される。

まず、任意の方策$\pi$のもとでの状態$s$における価値$v_\pi(s)$は、収益$G_t$の期待値として定義され，次のように変形できる：

$$
\begin{align}
v_\pi(s) &:= \mathbb{E}_\pi\left[ G_t \mid s_t = s \right]\\
&=\mathbb{E}_\pi\left[ r_{t+1} + \gamma G_{t+1} \mid s_t = s \right]\quad(\because G_t = r_{t+1} + \gamma G_{t+1})\\
&=\mathbb{E}_\pi\left[ r_{t+1} + \gamma v_\pi(s_{t+1}) \mid s_t = s \right]\quad(\because v_\pi(s_{t+1})=\mathbb{E}_\pi[G_{t+1}\mid s_{t+1} = s'])
\end{align}
$$

$$
\begin{equation}
v_\pi(s) = \mathbb{E}_\pi\left[ r_{t+1} + \gamma v_\pi(s_{t+1}) \mid s_t = s \right]
\end{equation}
$$

が成り立つ。この式こそが、状態$s$における価値を1ステップ先の報酬と次状態の価値に基づいて表現した**Bellman方程式**である。すなわち、Bellman方程式とは、現在の価値と1ステップ将来の価値との間の再帰的な関係を与える式を指す。

モンテカルロ法では収益$G_t$そのものを利用して更新を行っていたが、ここでは$G_t$の代わりに$r_{t+1} + \gamma v_\pi(s_{t+1})$の推定を利用する。このとき、価値関数$v_\pi(s)$は未知であるため、近似推定$V(s)$を用いる。すると、状態$s_t$における推定値$V(s_t)$の更新則は次のように表される。

$$
\begin{equation}
V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
\end{equation}
$$

この学習則を**時間差分学習** (temporal difference learning)、略して**TD学習**と呼ぶ。ここで、$r_{t+1} + \gamma V(s_{t+1})$と$V(s_t)$の差分を**報酬予測誤差** (reward prediction error, RPE) または**TD誤差** (TD error) と呼び、次式で定義する。

$$
\begin{equation}
\delta_t := r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\end{equation}
$$

これを用いると、TD学習の更新則は簡潔に

$$
\begin{equation}
V(s_t) \leftarrow V(s_t) + \alpha \delta_t
\end{equation}
$$

と書き表すことができる。TD学習では、各ステップで報酬$r_{t+1}$を受け取った時点で、次状態$s_{t+1}$の推定値$V(s_{t+1})$と合わせて、直前の状態$s_t$の推定値$V(s_t)$を即座に更新できるため、オンラインでの学習が可能となる。

### 適格度トレースとTD($\lambda$)法
TD学習の更新は「1ステップ先」の情報のみを用いて行われるが、より将来の報酬も考慮に入れるためには、$n$ステップ先までの報酬と価値の情報を組み合わせた $n$-step TD学習が導入される。$n$-step TD学習では、時刻 $t$ における$n$ステップターゲット$G_t^{(n)}$を次のように定義する：

$$
G_t^{(n)} := r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
$$

これは、$n$ステップ分の報酬列を直接和として加算し、その後$n$ステップ目の状態価値$V(s_{t+n})$を割引して足し合わせる構造になっている。そして、この$n$-stepターゲットを用いた$n$-step TD更新則は

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[G_t^{(n)} - V(s_t)\right]
$$

と表される。すなわち、$V(s_t)$を、$n$ステップ先までの情報を反映したターゲット$G_t^{(n)}$に向かって補正する形で更新する。

この$n$-step TDの考え方を発展させ、1ステップから無限ステップ（エピソード終了まで）にわたるターゲットを適切に混合する方法が TD($\lambda$) である。TD($\lambda$)では、さまざまな$n$に対応する$n$-stepターゲットを重み付き平均し、よりなめらかに将来の情報を考慮する。具体的には、時刻$t$におけるTD($\lambda$)ターゲット$G_t^\lambda$は次式で定義される：

$$
G_t^\lambda := (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
$$

ここで、$\lambda\in[0,1]$は混合係数であり、$\lambda$が小さいほど短期の報酬に重み付けされ、$\lambda$が大きいほど長期の報酬に重み付けされる。$\lambda=0$ではTD(0)と一致し、$\lambda=1$ではMonte Carlo法（エピソード全体の累積報酬）に一致する。この定式化により、短期・長期の情報を連続的に調整することが可能となる。

しかしながら、$G_t^\lambda$ の直接計算は実用上困難であり、エピソード終了まで待たなければならない。この問題を解決するために、**適格度トレース** (eligibility trace) を用いたonline TD($\lambda$) が導入される。この方法では、各状態$s$について、時刻$t$における適格度トレース $e_t(s)$ を次式で更新する：

$$
e_t(s) = 
\begin{cases}
\gamma\lambda e_{t-1}(s) + 1 & \text{if } s=s_t \\
\gamma\lambda e_{t-1}(s) & \text{otherwise}
\end{cases}
$$

ここで、$e_{t-1}(s)$は前時刻のeligibility traceであり、訪れた状態$s_t$のtraceを1だけ加算し、他の状態は$\gamma\lambda$倍して減衰させる。これにより、直近に訪れた状態ほど大きな影響を受け、時間とともにその影響は指数関数的に減衰する。

さらに、各ステップでTD誤差$\delta_t$を計算する：

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

そして、すべての状態$s$について、eligibility traceに基づいて次のように価値関数を更新する：

$$
V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)
$$

このように、TD($\lambda$)のonline実装では、1ステップごとにTD誤差を算出し、その誤差を現在および過去に訪れた状態たちに対してeligibility traceに応じて分配していく。これにより、エピソード終了を待たずにリアルタイムで価値関数の学習を進めることが可能となる。

以上のように、$n$-step TDは有限ステップ先をターゲットに用いる手法であり、TD($\lambda$)はこれらを指数加重平均してなめらかに学習する方法である。そしてTD($\lambda$)の実用的なonline実装は、eligibility traceによる局所的な記憶と更新を通じて達成されるのである。

### 報酬予測誤差とドーパミン作動性ニューロン
ここでTD学習と神経科学の対応について紹介する．

大脳基底核

報酬に応答するニューロンがあるとは知られていた．

理論的枠組みは
https://www.jneurosci.org/content/16/5/1936/tab-article-info

ドーパミン作動性ニューロン (dopaminergic neurons) あるいは ドーパミンニューロン (dopamine neurons) は神経伝達物質の一種であるドーパミン (dopamine) を分泌する神経細胞であり，主に中脳の腹側被蓋野 (Ventral tegmental area, VTA) や黒質緻密部 (substantia nigra pars compacta, SNc) に分布している．

TD学習における報酬予測誤差がドーパミン作動性ニューロン (dopaminergic neurons; DA) により符号化されていることがSchultzらにより報告されている {cite:p}`Schultz1997-ih`. 

サルのVTA


ドーパミンニューロンであるとどう同定したか？

https://www.nature.com/articles/nature03015

条件刺激 (conditioned stimulus, CS) と無条件刺激 (unconditioned stimulus, US)


シミュレーションをここに入れる．Schlutz, 
北澤の再現．


A Unified Framework for Dopamine Signals across Timescales

https://www.pnas.org/doi/10.1073/pnas.2316658121

https://www.nature.com/articles/s41593-023-01566-3

ただし，VTAとSNcのドーパミンニューロンの役割は同一ではない．ドーパミンニューロンへの入力が異なっています [(Watabe-Uchida et al., _Neuron._ 2012)](https://www.cell.com/neuron/fulltext/S0896-6273(12)00281-4)． また，細かいですがドーパミンニューロンの発火は報酬量に対して線形ではなく，やや飽和する非線形な応答関数 (Hill functionで近似可能)を持ちます([Eshel et al., _Nat. Neurosci._ 2016](https://www.nature.com/articles/nn.4239))．このため著者実装では報酬 $r$に非線形関数がかかっているものもあります．

先ほどRPEはドーパミンニューロンの発火率で表現されている，といいました．RPEが正の場合はドーパミンニューロンの発火で表現できますが，単純に考えると負の発火率というものはないため，負のRPEは表現できないように思います．ではどうしているかというと，RPEが0（予想通りの報酬が得られた場合）でもドーパミンニューロンは発火しており，RPEが正の場合にはベースラインよりも発火率が上がるようになっています．逆にRPEが負の場合にはベースラインよりも発火率が減少する(抑制される)ようになっています

https://www.nature.com/articles/nature12475



ドーパミンニューロンの短時間の光遺伝学的抑制は内因性の負の報酬予測誤差を模倣する
https://www.nature.com/articles/nn.4191 "https://www.nature.com/articles/nn.4191

発火率というのを言い換えればISI (inter-spike interval, 発火間隔)の長さによってPREが符号化されている(ISIが短いと正のRPE, ISIが長いと負のRPEを表現)ともいえます ([Bayer et al., <span style="font-style: italic;">J.
Neurophysiol</span>. 2007](https://www.physiology.org/doi/full/10.1152/jn.01140.2006 "https://www.physiology.org/doi/full/10.1152/jn.01140.2006"))．

ドーパミンニューロンの活動は報酬予測誤差のみを符号化しているわけではなく，運動 (movement), Salience, Threat等の他の要素に関しても予測誤差を計算していると報告されています．
これを一般化予測誤差という

https://www.nature.com/articles/s41593-024-01705-4


予測価値(分布) $V(x)$ですが，これは線条体(striatum)のパッチ (SNcに抑制性の投射をする)やVTAのGABAニューロン (VTAのドーパミンニューロンに投射して減算抑制をする, ([Eshel, et al., _Nature_. 2015](https://www.nature.com/articles/nature14855 "https://www.nature.com/articles/nature14855")))などにおいて表現されている．

### Rescorla-Wagnerモデル
TD学習は古典的条件付け (Classical conditioning) のモデルである，Rescorla-Wagner (RW) モデル {cite:p}`rescorla1972theory` と予測誤差に基づいて学習を進めるという点で関連がある．RWモデルは条件刺激 (CS) と無条件刺激 (US) の間

$$
\Delta V_i = \eta \left(\lambda - \sum_j V_j\right)
$$

https://www.jstage.jst.go.jp/article/janip/66/2/66_66.2.4/_pdf

### eligibility traceの利用とTD($\lambda$) 則
eligibility trace