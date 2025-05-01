# 第9章：神経回路網による不確実性の表現とベイズ推論
## ベイズ脳仮説と不確実性の表現

(書き直す)

変分ベイズ推論は

https://arxiv.org/abs/1901.07945

ベイズ脳仮説（The Bayesian Brain Hypothesis）は、脳が確率的推論に基づいて感覚情報を処理し、外界の状態を推定しているという理論である。この仮説において、脳は世界に関する内部モデルを構築しており、そこに入力される不完全かつ雑音を含む感覚情報をもとに、ベイズの定理を用いて外界の隠れた原因を推測する。ベイズの定理は、ある観測 $x$ が与えられたときに、その観測を引き起こしたと考えられる原因 $z$ の確率を次のように与える：

$$
p(z \mid x) = \frac{p(x \mid z) \cdot p(z)}{p(x)}.
$$

ここで、$p(z \mid x)$ は観測 $x$ をもとにした原因 $z$ の事後確率、$p(x \mid z)$ は原因 $z$ に基づいて観測される $x$ の尤度、$p(z)$ は原因に対する事前確率、そして $p(x)$ は観測全体の周辺尤度である。この定理に従って、脳は感覚情報に対して最も妥当な解釈を与える原因を推定することになる。

脳内の知覚処理は、単に入力された情報を逐次的に処理するのではなく、過去の経験や学習によって形成された事前分布 $p(z)$ に基づいて、現在の感覚入力 $x$ を統合的に解釈する。たとえば、視覚において曖昧な像が網膜に映った場合でも、脳はこれまでに得た視覚的知識を用いて、その像が何であるかを推測する。この過程では、感覚入力の不確かさに応じて尤度 $p(x \mid z)$ を評価し、それを既存の事前分布と統合することで、最終的な事後分布 $p(z \mid x)$ を得る。

このようなベイズ的推論の過程は、近年の予測符号化（predictive coding）の理論とも密接に関連している。予測符号化モデルにおいては、脳は高次の神経回路から低次の回路へと予測信号を送り、それと実際の感覚入力との間に生じる予測誤差を下から上へと伝播させる。この誤差が学習や推論の駆動源となり、内部モデルが更新される。数式で表すと、予測誤差は次のように定義される：

$$
\epsilon = x - \hat{x}(z),
$$

ここで $\hat{x}(z)$ は原因 $z$ に基づく感覚入力の予測値である。脳はこの予測誤差 $\epsilon$ を最小化する方向に内部表現 $z$ を更新することで、より正確な知覚や認知を実現している。これは、事後確率 $p(z \mid x)$ を最大化する（すなわち MAP 推定を行う）操作に相当する。

このような理論は、知覚だけでなく注意、意思決定、運動制御、学習など、さまざまな脳機能に適用可能であり、実際、神経科学の実験においてもベイズ的推論と整合する結果が多数報告されている。たとえば、期待された刺激に対して視覚野の神経活動が抑制される現象は、予測が成功し誤差が小さくなったことを反映していると解釈される。また、注意の効果は、事前分布 $p(z)$ の重みづけの変化として理解される。

以上のように、ベイズ脳仮説は、脳の情報処理を確率論的推論としてとらえることで、感覚から行動に至る広範な認知機能を統一的に説明する枠組みを提供している。脳は不確実性を内包する世界の中で、限られた情報をもとに最も妥当な仮説を選び、常にそれを更新し続けるベイズ推論器として機能しているのである。

### ベイズ脳仮説
Knill, David C., and Alexandre Pouget. 2004. “The Bayesian Brain: The Role of Uncertainty in Neural Coding and Computation.” Trends in Neurosciences 27 (12): 712–19.

### 神経活動による不確実性の表現
ここまでは最尤推定やMAP推定などにより，パラメータ(神経活動，シナプス結合)の点推定を行ってきた．**不確実性(uncertainty)** を神経回路で表現する方法として主に2つの符号化方法，**サンプリングに基づく符号化(sampling-based coding; SBC or neural sampling model)** および**確率的集団符号化(probabilistic population coding; PPC)** が提案されている．SBCは神経活動が元の確率分布のサンプルを表現しており，時間的に多数の活動を集めることで元の分布の情報が得られるというモデルである．PPCは神経細胞集団により，確率分布を表現するというモデルである．

- (Walker et al., 2022)がまとめ．
- (Fiser et al., 2010)の比較表を入れる．
- 神経活動の変動性 (neural variability)
- 自発活動が事前分布であるという説 {cite:p}`Fiser2010-kw`, {cite:p}`Berkes2011-it`.
- {cite:p}`Hoyer2002-ci`
- {cite:p}`Sanborn2016-en`

## ベイズ脳仮説と神経活動による不確実性の表現

が外界の状態を推定する際には**不確実性 (uncertainty)** を考慮する必要がある．例えば外界は3次元なのに対し，網膜像は2次元であり，脳は不良設計問題を解かねばならない．時間の推定においては時間経過を直接的に示す感覚情報はないため，不確実性を常に含む．これらのような不確実性を含んだ推定において脳がベイズ推定を用いているというのが**ベイズ脳仮説 (Bayesian brain hypothesis)** である (Knill & Pouget, 2004)．ここで外界の状態を$x$, それによって生まれた感覚刺激を$y$, 脳内の神経結合を$W$としよう．**事前分布 (prior)** を$p(x|W)$とし，**尤度 (likelihood)** を$p(y|x,\ W)$とすると，**事後分布 (posterior)**は

$$
\begin{equation}
p\left( x \middle| y \right) = \frac{p\left( y \middle| x,\ W \right)p(x|W)}{p(y|W)}
\end{equation}
$$

しかし，ここでの問題は次の2点である．すなわち，

1.  神経回路で確率分布を如何にして表現するか．

2.  規格化定数 $Z = p\left( y \middle| W \right) = \int p\left( y \middle| x,\ W \right)p\left( x \middle| W \right)\ dx$をどう計算するか．

- Neural Sampling Codes
- Probabilistic Population Coding
- Distributed distributional code
RS Zemel, P Dayan, and A Pouget. Probabilistic interpretation of population codes. Neural Computation, 10(2):403–430, 1998. [8] MSahani and P Dayan. Doubly distributional population codes: Simultaneous representation of uncertainty and multiplicity. Neural Computation, 15(10):2255–2279, 2003.

## 神経回路における不確実性の表現

　神経細胞あるいは細胞集団が確率分布を表現するにはどうすればよいだろうか．神経細胞の活動がある変数を表現していると仮定しよう．単一の細胞の瞬時的な活動がある変数の点推定に対応していると考えれば，単一の細胞の多数の活動あるいは多数の細胞の瞬時的な活動により分布は表現できると考えられる (Fig.2)．

**Fig. 2**. 神経活動による確率分布表現の2種類の方法．(Fiser, Berkes, Orbán, & Lengyel, 2010)より引用．(a)多数の細胞の瞬時的な活動により分布を表現する符号化 (e.g. probabilistic population codes; PPCs)．(b)単一の細胞の多数の活動により分布を表現する符号化 (e.g. neural sampling codes; NSCs)．Table1は両者の比較．著者らはSampling-based codeの方が優れていると考えている．

多数の細胞の瞬時的な活動により分布を表現する符号化としては**probabilistic population codes** (Ma, Beck, Latham, & Pouget, 2006)や**distributional TD learning** (Dabney et al., 2020; Lowet, Zheng, Matias, Drugowitsch, & Uchida, 2020)などが該当する．一方で単一の細胞の多数の活動により分布を表現する符号化は**サンプリングに基づいた符号化 (sampling-based coding)** あるいは**神経サンプリング (neural sampling)** と呼ぶ．神経サンプリングの基盤となる現象は**神経活動の変動性 (neural variability)** である．これは感覚を処理する皮質領野（例えば視覚野）において同じ入力であっても神経細胞の活動が時間や試行に応じて変動する現象のことである (Stein, Gossen, & Jones, 2005)．これが単なるノイズなのか機能があるのかに関しては様々な説が提案されているが，神経活動の変動性によりMCMCが行われているという仮説は(Hoyer & Hyvärinen, 2002)において（自分の知る限り）初めて提案された．(Sanborn & Chater, 2016)は”Bayesian Brains without Probabilities”というキャッチーな題だが，MCMCとBayesian Brainの勉強にはなる．