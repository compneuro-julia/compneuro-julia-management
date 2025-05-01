## 分布型強化学習
分布型強化学習 (Distributional reinforcement learning)

TD学習の拡張である．

Dabneyら

$$
Z^\pi (s, a) = 
$$

分布型Bellmann方程式 (distributional Bellman equation) 

$$
Z(s_t) = R(s_t) + \gamma Z(s_{t+1})
$$

期待値をとると，$V(s_t)=\mathbb{E}[Z(s_t)]$ となる．

Quantileはノンパラ
PPCやDPCはパラメトリック


https://arxiv.org/abs/1710.10044

### sign関数を用いたDistributional RLと分位点回帰

それでは，なぜ予測価値 $V_i$は$\tau_i$ 分位点に収束するのでしょうか．Extended Data Fig.1のように平衡点で考えてもよいのですが，後のために分位点回帰との関連について説明します．分位点回帰については記事を書いたので先にそちらを読んでもらうと分かりやすいと思います

実はDistributional RL (かつ，RPEの応答関数にsign関数を用いた場合)における予測報酬 $V_i$の更新式は，分位点回帰(Quantile
regression)を勾配法で行うときの更新式とほとんど同じです．分位点回帰では$\delta$の関数$\rho_{\tau}(\delta)$を次のように定義します． 

$$ \rho_{\tau}(\delta)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right|\cdot |\delta|=\left(\tau-\mathbb{I}_{\delta
\leq 0}\right)\cdot \delta 
$$ 

そして，この関数を最小化することで回帰を行います．ここで$\tau$は分位点です．また$\delta=r-V$としておきます．今回，どんな行動をしても未来の報酬に影響はないので$\gamma=0$としています．

ここで， 

$$ 
\frac{\partial \rho_{\tau}(\delta)}{\partial \delta}=\rho_{\tau}^{\prime}(\delta)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot \operatorname{sign}(\delta) 
$$ 

なので，$r$を観測値とすると， 

$$
\frac{\partial \rho_{\tau}(\delta)}{\partial V}=\frac{\partial \rho_{\tau}(\delta)}{\partial \delta}\frac{\partial \delta(V)}{\partial V}=-\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\operatorname{sign}(\delta) 
$$ 

となります．ゆえに$V$の更新式は 

$$ 
V \leftarrow V - \beta\cdot\frac{\partial \rho_{\tau}(\delta)}{\partial V}=V+\beta \left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\operatorname{sign}(\delta) 
$$ 

です．ただし，$\beta$はベースラインの学習率です．個々の$V_i$について考え，符号で場合分けをすると


$$ 
\begin{cases} V_{i} \leftarrow V_{i}+\beta\cdot |\tau_i|\cdot\operatorname{sign}\left(\delta_{i}\right)
&\text { for } \delta_{i}>0\\ V_{i} \leftarrow V_{i}+\beta\cdot |\tau_i-1|\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i} \leq 0 \end{cases} 
$$ 

となります．$0 \leq
\tau_i \leq 1$であり，$\tau_i=\alpha_{i}^{+} / \left(\alpha_{i}^{+} + \alpha_{i}^{-}\right)$であることに注意すると上式は次のように書けます． 

$$ 
\begin{cases} V_{i} \leftarrow V_{i}+\beta\cdot
\frac{\alpha_{i}^{+}}{\alpha_{i}^{+}+\alpha_{i}^{-}}\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i}>0\\ V_{i} \leftarrow V_{i}+\beta\cdot
\frac{\alpha_{i}^{-}}{\alpha_{i}^{+}+\alpha_{i}^{-}}\cdot\operatorname{sign}\left(\delta_{i}\right) &\text { for } \delta_{i} \leq 0 \end{cases} 
$$ 

これは前節で述べたDistributional
RLの更新式とほぼ同じです．いくつか違う点もありますが，RPEが正の場合と負の場合に更新される値の比は同じとなっています．

このようにRPEの応答関数にsign関数を用いた場合，報酬分布を上手く符号化することができます．しかし実際のドーパミンニューロンはsign関数のような生理的に妥当でない応答はせず，RPEの大きさに応じた活動をします．そこで次節ではRPEの応答関数を線形にしたときの話をします．

#### 分位点・エクスペクタイル回帰
本章では分位点・エクスペクタイル回帰 (quantile/expectile regression) を用いて

- Quantileはノンパラ
- PPCやDPCはパラメトリック

Distributional Reinforcement Learning in the Brainに
> Quantile-like codes are non-parametric codes, as they do not a priori assume a specific form of a probability distribution with associated parameters. Previous studies have proposed different population coding schemes. For example, probabilistic population codes (PPCs) [73,74] and distributed distributional codes (DDCs) [75,76] employ population coding schemes from which various statistical parameters of a distribution can be read out, making them parametric codes. As a simple example, a PPC might encode a Gaussian distribution, in which case the mean would be reflected in which specific neurons are most active, and the variance would be reflected in the inverse of the overall activity [73].

## 分位点・エクスペクタイル回帰
### 分位点回帰 (Quantile Regression)
線形回帰(linear regression)は，誤差が正規分布と仮定したとき(必ずしも正規分布を仮定しなくてもよい)の$X$(説明変数)に対する$Y$(目的変数)の期待値$E[Y]$を求める，というものであった．**分位点回帰(quantile regression)** では，Xに対するYの分布における分位点を通るような直線を引く．

**分位点**(または分位数)において，代表的なものが**四分位数**である．四分位数は箱ひげ図などで用いるが，例えば第一四分位数は分布を25:75に分ける数，第二四分位数(中央値)は分布を50:50に分ける数である．同様に$q$分位数($q$-quantile)というと分布を$q:1-q$に分ける数となっている．分位点回帰の話に戻る．下図は$x\sim U(0, 5),\quad y=3x+x\cdot \xi,\quad \xi\sim N(0,1)$とした500個の点に対する分位点回帰である．赤い領域はX=1,2,3,4でのYの分布を示している．深緑，緑，黄色の直線はそれぞれ10, 50, 90%tile回帰の結果である．例えば50%tile回帰の結果は，Xが与えられたときのYの中央値(50%tile点)を通るような直線となっている．同様に90%tile回帰の結果は90%tile点を通るような直線となっている．

分位点回帰の利点としては，外れ値に対して堅牢(ロバスト)である，Yの分布が非対称である場合にも適応できる，などがある ([Das et al., *Nat Methods*. 2019](https://www.nature.com/articles/s41592-019-0406-y))．

### エクスペクタイル回帰 (Expectile regression)
エクスペクタイル(expectile)は([Newey and Powell 1987](https://www.jstor.org/stable/1911031?seq=1)) によって導入された統計汎関数 (statistical functional; SF)の一種であり，期待値(expectation)と分位数(quantile)を合わせた概念である．簡単に言えば，中央値(median)の一般化が分位数(quantile)であるのと同様に，期待値(expectation)の一般化がエクスペクタイル(expectile)である．

### 勾配法を用いた分位点回帰・エクスペクタイル回帰
予測誤差$\delta$と$\tau$の関数を

$$
\begin{align}
\text{分位点回帰：}&\quad
\rho_q(\delta; \tau)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right|\cdot |\delta|=\left(\tau-\mathbb{I}_{\delta \leq 0}\right)\cdot \delta\\
\text{エクスペクタイル回帰：}&\quad
\rho_e(\delta; \tau)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right|\cdot \delta^2
\end{align}
$$

と定義する．$\rho_q(\delta; \tau)$のみ，チェック関数 (check function)あるいは非対称絶対損失関数(asymmetric absolute loss function)と呼ぶ．ただし，$\tau$は分位点(quantile)，$\mathbb{I}$は指示関数(indicator function)である．この場合，$\mathbb{I}_{\delta \leq 0}$は$\delta \gt 0$なら0, $\delta \leq 0$なら1となる．このとき，目的関数は 

$$
L_{\tau}(\delta)
=\sum_{i=1}^n \rho(\delta_i; \tau)
$$

である．$\rho(\delta; \tau)$を色々な $\tau$についてplotすると次図のようになる．

分位点の場合，$\rho_q(\delta; \tau)$がチェックマーク✓に類似していることからこのような名前が付いている．

$L_\tau$を最小化するような$\theta$の更新式について考える．まず，


$$
\begin{align}
\text{分位点回帰：}&\quad
\frac{\partial \rho_q(\delta; \tau)}{\partial \delta}= \rho_q^{\prime}(\delta; \tau)=\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\operatorname{sign}(\delta)\\
\text{エクスペクタイル回帰：}&\quad
\frac{\partial \rho_e(\delta; \tau)}{\partial \delta}= \rho_e^{\prime}(\delta; \tau)=2\left|\tau-\mathbb{I}_{\delta \leq 0}\right| \cdot
\delta
\end{align}
$$

である (ただし$\text{sign}(\cdot)$は符号関数)．さらに

$$
\frac{\partial L_{\tau}}{\partial \theta}=\frac{\partial L_{\tau}}{\partial \delta}\frac{\partial \delta(\theta)}{\partial \theta}=-\frac{1}{n} \rho^{\prime}(\delta; \tau) X
$$ 

が成り立つので，$\theta$の更新式は$\theta \leftarrow \theta + \alpha\cdot \dfrac{1}{n} \rho^{\prime}(\delta; \tau) X$と書ける ($\alpha$は学習率である)．分位点回帰を単純な勾配法で求める場合，勾配が0となって解が求まらない可能性があるが，目的関数を滑らかにすることで回避できるという研究もある ([Zheng. *IJMLC*. 2011](https://link.springer.com/article/10.1007/s13042-011-0031-2))．この点，Expectileならこの問題を回避できる (?)．

## 分布型TD学習
分布型TD学習 (Distributional TD learning) は

Distributional TD learningではRPEの正負に応じて，予測報酬の更新を異なる学習率($\alpha_{i}^{+}, \alpha_{i}^{-}$)を用いて行う． 

$$ 
\begin{cases} V_{i}(x) \leftarrow V_{i}(x)+\alpha_{i}^{+} f\left(\delta_{i}\right) &\text{for }
\delta_{i} \gt 0\\ V_{i}(x) \leftarrow V_{i}(x)+\alpha_{i}^{-} f\left(\delta_{i}\right) &\text{for } \delta_{i} \leq 0 \end{cases} 
$$ 

ここで，シミュレーションにおいては$\alpha_{i}^{+}, \alpha_{i}^{-}\sim U(0,
1)$とする($U$は一様分布)．さらにasymmetric scaling factor $\tau_i$を次式により定義する． 

$$ 
\tau_i=\frac{\alpha_{i}^{+}}{\alpha_{i}^{+}+ \alpha_{i}^{-}} 
$$ 

なお，$\alpha_{i}^{+}, \alpha_{i}^{-}\in [0, 1]$より$\tau_i \in
[0,1]$である． 

Classical TD learningとDistributional TD learningにおける各ニューロンのRPEに対する発火率を表現したのが次図となる．

Classical TD learningではRPEに比例して発火する細胞しかないが，Distributional TD learningではRPEの正負に応じて発火率応答が変化していることがわかる． 特に$\alpha_{i}^{+} \gt \alpha_{i}^{-}$の細胞を**楽観的細胞 (optimistic cells)**，$\alpha_{i}^{+}\lt
\alpha_{i}^{-}$の細胞を**悲観的細胞 (pessimistic
cells)** と著者らは呼んでいる．実際には2群に分かれているわけではなく，gradientに遷移している．収束する予測価値が細胞ごとに異なることで，$V$には報酬の期待値ではなく複雑な形状の報酬分布が符号化される．その仕組みについて，次項から見ていこう．

### 分位数(Quantile)モデルと報酬分布の符号化

#### RPEに対する応答がsign関数のモデルと報酬分布の分位点への予測価値の収束
さて，Distributional RLモデルでどのようにして報酬分布が学習されるかについてみていこう．この項ではRPEに対する応答関数$f(\cdot)$が符合関数(sign function)の場合を考える．結論から言うと，この場合はasymmetric scaling factor $\tau_i$は分位数(quantile)となり，**予測価値
$V_i$は報酬分布の$\tau_i$分位数に収束する**．
    
どういうことかを簡単なシミュレーションで見てみよう．今，報酬分布を平均2, 標準偏差5の正規分布とする (すなわち$r \sim N(2, 5^2)$となります)．また，$\tau_i = 0.25, 0.5, 0.75 (i=1,2,3)$とする．このとき，3つの予測価値 $V_i \ (i=1,2,3)$はそれぞれ$N(2, 5^2)$の0.25, 0.5,
0.75分位数に収束する．下図はシミュレーションの結果である．左が$V_i$の変化で，右が報酬分布と0.25, 0.5, 0.75分位数の位置 (黒短線)となっています．対応する分位数に見事に収束していることが分かる．