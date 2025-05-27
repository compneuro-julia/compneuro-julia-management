## シナプス伝達のモデル

量子仮説
連続か離散か
神経伝達物質は量子的に放出される．

# Current / Conductance-based シナプス

## 化学シナプスの2つの記述形式

具体的なシナプスのモデルの前に，この節では化学シナプスにおけるシナプス入力(synaptic drive)の2つの形式，**Current-based シナプス**と**Conductance-based シナプス**について説明する．簡単に言うと，Current-based シナプスは入力電流が変化するというモデルで，Conductance-based シナプスはイオンチャネルのコンダクタンス (電気抵抗の逆数，電流の流れやすさ) が変化するというモデルである \citep{Cavallari2014-jx}．以下では例として，次のLIFニューロンの方程式におけるシナプス入力を考える．

$$
\begin{equation}
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_m I_{\text{syn}}(t)    
\end{equation}
$$

ただし，$\tau_m$ は膜電位の時定数，$V_m(t)$ は膜電位，$V_\text{rest}$ は静止膜電位，$R_m$ は膜抵抗である．ここで，シナプス入力の電流$I_{\text{syn}}(t)$ \footnote{シナプス(synapse)入力であることを明らかにするためにsynと添え字をつけている．} が2つのモデルにおいて異なる部分となる．

## Current-based シナプス
Current-based シナプスは単純に**入力電流が変化**するというモデルで，モデルを簡素化したい場合によく用いられる．シナプス入力 $I_{\text{syn}}(t)$はシナプス効率(synaptic efficacy)\footnote{シナプス強度(Synaptic strength)とは違い，受容体の種類(GABA受容体やAMPA受容体，およびそのサブタイプなど)によって決まる．} を $J_{\text{syn}}$ (単位はpA) とし，シナプスの動態(synaptic kinetics)を $s_{\text{syn}}(t)$ とすると，次式のようになる．ただし，シナプスの動態とは前細胞に注目すれば神経伝達物質の放出量，後細胞に注目すれば神経伝達物質の結合量やイオンチャネルの開口率を表す．

$$
\begin{equation}
I_{\text{syn}}(t)=\underbrace{J_{\text{syn}}s_{\text{syn}}(t)}_{電流の変化}    
\end{equation}
$$

ただし，$s_{\text{syn}}(t)$ は，例えば次節で紹介する $\alpha$ 関数を用いる場合, 

$$
\begin{equation}
s_{\text{syn}}(t)=\dfrac{t}{\tau_s} \exp \left(1-\dfrac{t}{\tau_s}\right)    
\end{equation}
$$

のようになる．

## Conductance-based シナプス
Conductance-based シナプスはイオンチャネルの**コンダクタンスが変化**するというモデルである．例えば Hodgkin-Huxley モデルはConductance-based モデルの1つである．Current-basedよりもConductance-based の方が生理学的に妥当である．例えば抑制性シナプスは膜電位が平衡電位と比べて脱分極側にあるか，過分極側にあるかで抑制的に働くか興奮的に働くかが逆転するが，これはCurrent-based シナプスでは再現できない．Conductance-based モデルにおけるシナプス入力は$I_{\text{syn}}(t)$は次のようになる． 

$$
\begin{equation}
I_{\text{syn}}(t)=\underbrace{g_{\text{syn}}s_{\text{syn}}(t)}_{コンダクタンスの変化}\cdot\ \left(V_{\text{syn}}-V_{m}(t)\right)    
\end{equation}
$$

ただし，$g_{\text{syn}}$ (単位はnS)はシナプスの最大コンダクタンス\footnote{$g_{\text{syn}}$がシナプスの最大コンダクタンスとなるのは $s_{\text{syn}}$の最大値を1に正規化する場合である．正規化は必須ではないので，単なる係数と思うのがよい．}，$V_{\text{syn}}$ (単位はmV) はシナプスの平衡電位を表す．これらも$J_{\text{syn}}$と同じく，シナプスにおける受容体の種類によって決まる定数である．

注意しなければならないことは，$s_{\text{syn}}(t)\leq 0$としたとき Current-based モデルにおける $J_{\text{syn}}$ は正の値(興奮性)と負の値(抑制性)を取るが，$g_{\text{syn}}$は正の値のみである，ということである\footnote{これはコンダクタンスが電気抵抗の逆数であり，基本的に抵抗は正の値しか取らないことからも分かる．なお電子回路においては素子の抵抗値が見かけ上，負の値を取る場合もあり**負性抵抗** (negative resistance) と呼ばれる}．Conductance-basedモデルで興奮性と抑制性を決定しているのは，平衡電位$V_{\text{syn}}$である．興奮性シナプスの平衡電位は高く，抑制性シナプスの平衡電位は低いため，膜電位を引いた符号はそれぞれ正と負になる．


## シナプス入力の重みづけ

ここまでは, シナプス前細胞と後細胞がそれぞれ1つずつである場合について考えていたが, 実際には多数の細胞がネットワークを作っている．また, それぞれの入力は均等ではなく, 異なるシナプス強度 (Synaptic strength)を持つ．この場合のシナプス入力の計算について述べておく．

シナプス前細胞が$N_{\text{pre}}$個, シナプス後細胞が$N_{\text{post}}$個あるとする．このとき**シナプス前過程に注目した**シナプス動態を$\boldsymbol{s_{\text{syn}}}\in \mathbb{R}^{N_{\text{pre}}}$, シナプス後細胞の入力電流を$\boldsymbol{I_{\text{syn}}}\in \mathbb{R}^{N_{\text{post}}}$, シナプス結合強度の行列を$W\in \mathbb{R}^{N_{\text{post}} \times N_{\text{pre}}}$とすると, Current-basedの場合は

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=W \boldsymbol{s_{\text{syn}}}  
\end{equation}
$$

となる．ただし, シナプス強度にシナプス効率が含まれるとした. また, Conductance-basedの場合はシナプス後細胞の膜電位を$\boldsymbol{V}_{m}\in \mathbb{R}^{N_{\text{post}}}$として, 

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=\left(V_{\text{syn}}-\boldsymbol{V}_{m}(t)\right)\odot W \boldsymbol{s_{\text{syn}}}
\end{equation}
$$

となる．ただし, $\odot$はHadamard積である．

これらの式は順序を入れ替えることも可能である．シナプス前細胞でスパイクが生じたことを表すベクトルを$\boldsymbol{\delta}_{t,t_{\text{spike}}}\in \mathbb{R}^{N_{\text{pre}}}$とする．ただし, $t_{\text{spike}}$は各ニューロンにおいてスパイクが生じた時刻である． $\boldsymbol{s_{\text{syn}}}$は$\boldsymbol{\delta}_{t,t_{\text{spike}}}$の関数であり, $\boldsymbol{s_{\text{syn}}}(\boldsymbol{\delta}_{t,t_{\text{spike}}})$と表せる．このとき**シナプス後過程に注目した**シナプス動態を$\boldsymbol{s}^\prime_{\text{syn}}\in \mathbb{R}^{N_{\text{post}}}$とすると, Current-basedの場合は

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=\boldsymbol{s}^\prime_{\text{syn}}(W\boldsymbol{\delta}_{t,t_{\text{spike}}})  
\end{equation}
$$

Conductance-basedの場合は

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=\left(V_{\text{syn}}-\boldsymbol{V}_{m}(t)\right)\odot \boldsymbol{s}^\prime_{\text{syn}}(W\boldsymbol{\delta}_{t,t_{\text{spike}}})
\end{equation}
$$

と表すことができる．

シナプス動態を前過程か後過程のどちらに注目したものとするかは, 実装によって様々である．シナプス入力の計算における中間の値を学習に用いるということもあるため, 単なる計算量の観点だけではどちらを選ぶかは決めることができない (計算量だけならシナプス変数に先に重み行列をかけた方がよい場合が多い)．実装の中で異なってくるのは計算順序と保持するベクトルの要素数である． 同じ実装の中で2つとも用いる場合もあるので注意してほしい．


## 指数関数型シナプスモデル
シナプスのモデルは複数あるが, 良く用いられるのが**指数関数型シナプスモデル**(exponential synapse model)である．このモデルは生理学的な過程を無視した現象論的モデルであることに注意しよう．指数関数型シナプスモデルには2つの種類, **単一指数関数型モデル** (single exponential model)と**二重指数関数型モデル** (double exponential model)がある．

数式の説明の前にモデルの挙動を示す．次図は2種類のモデルにおいて$t=0$でスパイクが生じてからのシナプス後電流の変化を示している．ただし, 実際のシナプス後電流はこれに**シナプス強度** (Synaptic strength)\footnote{シナプス強度というのは便宜上の呼称で, 実際には神経伝達物質の種類や, その受容体の数など複数の要因によって決定されている. また, このシナプス強度はシナプス重みということもある．これはどちらかと言えば機械学習の表現に引っ張られたものである．このため, このサイトでは重みという語も使う．}を乗じて総和を取ったものとなる．

## 単一指数関数型モデル(Single exponential model)
シナプス前ニューロンにおいてスパイクが生じてからのシナプス後電流の変化はおおよそ指数関数的に減少する, というのが単一指数関数型モデルである\footnote{薬学動態の静注1コンパートメントモデルと同じ式である．}. 式は次のようになる．


$$
\begin{equation}
f(t)=\frac{1}{\tau_{s}}\exp\left(-\frac{t}{\tau_s}\right)    
\end{equation}
$$

この関数を時間的なフィルターとして, 過去の全てのスパイクについての総和を取る．

$$
\begin{equation}
r(t)=\sum_{t_{k}< t} f\left(t-t_{k}\right)
\end{equation}
$$

ここで${r(t)}$は前節におけるシナプス動態($s_{\text{syn}}$)で, $t_{k}$はあるニューロンの$k$番目のスパイクの発生時刻である．${t_{k}<t}$の意味は現在の時刻$t$までに発生したスパイクについての和を取るという意味である．なお，スパイクが生じてから, ある程度の時間が経過した後はそのスパイクの影響はないと見なせるので, 一定の時間までの総和を取るのがよい．

別の表記法としてスパイク列に対する畳み込みを行うというものもある．畳み込み演算子を$*$とし, シナプス前細胞のスパイク列を$S(t)=\sum_{t_{k}< t} \delta\left(t-t_{k}\right)$とする (ただし, $\delta$はDiracのdelta関数において$\delta(0)=1$とした関数)．このとき, $r(t)=f*S(t)$と表すことができる．畳み込み演算子を用いると簡略な表記ができるが，実装上は他と同じ手法を用いる．

### 微分方程式による表現
上の手法ではニューロンの発火時刻を記憶し, 時間毎に全てのスパイクについての和を取る必要がある．そこで, 実装する場合は次の等価な微分方程式を用いる．

$$
\begin{equation}
\frac{dr}{dt}=-\frac{r}{\tau_{s}}+\frac{1}{\tau_{s}} \sum_{t_{k}< t} \delta\left(t-t_{k}\right)   
\end{equation}
$$

ここで$\tau_s$はシナプスの時定数(synaptic time constant)である． また, $\delta(\cdot)$はDiracのdelta関数です(ただし$\delta(0)=1$です). これをEuler法で差分化すると 

$$
\begin{equation}
r(t+\Delta t)=\left(1-\frac{\Delta t}{\tau_{s}}\right)r(t)+\frac{1}{\tau_{s}}\delta_{t,t_{k}} 
\end{equation}
$$

となる．ここで$\delta_{t,t_{k}}$はKroneckerのdelta関数で, $t=t_{k}$のときに1, それ以外は0となる．また減衰度として$\left(1-\Delta  t/\tau_{d}\right)$の代わりに$\exp\left(-\Delta t/\tau_{d}\right)$を用いる場合もある．

## 二重指数関数型モデル(Double exponential model)
2重の指数関数によりシナプス後電流の立ち上がりも考慮するのが, 二重指数関数型モデル(Double exponential model)である\footnote{薬学動態の内服1コンパートメントモデルと同じ式である．}．$t=0$にシナプス前細胞が発火したときのシナプス後電流の時間変化の関数は次のようになる．

$$
\begin{equation}
f(t)=A\left[\exp\left(-\frac{t}{\tau_d}\right)-\exp\left(-\frac{t}{\tau_r}\right)\right]    
\end{equation}
$$

ただし, ${\tau_r}$は立ち上がり時定数(synaptic rise time constant), ${\tau_d}$は減衰時定数(synaptic decay time constant)である．$\tau_{d}$は$\tau_{s}$と同じく神経伝達物質の減少速度を決定している．$A$は規格化定数で次のように表される．

$$
\begin{equation}
A=\frac{\tau_d}{\tau_d-\tau_r}\cdot \left(\frac{\tau_r}{\tau_d}\right)^\frac{\tau_r}{\tau_r-\tau_d}    
\end{equation}
$$

規格化定数$A$を乗じることで最大値が1となる．ただし, シミュレーションをする上で実際に規格化をする場合は少ない．

### $\alpha$関数
上記の式において, $\tau=\tau_{r}=\tau_{d}$の場合は $\boldsymbol{\alpha}$ **関数** (alpha function, alpha synapse)と呼ぶ \citep{Rall1967-gn}．式としては次のようになる．

$$
\begin{equation}
\alpha(t)=\frac{t}{\tau}\exp\left(1-\frac{t}{\tau}\right)    
\end{equation}
$$

この式は二重指数関数型シナプスの式に単に代入するだけでは導出できない．これらの式の対応については後述する．

### 微分方程式による表現
ここで, 二重指数関数型シナプスの式に対応する, 補助変数$h$を用いた微分方程式を導入する． 

$$
\begin{align} 
\frac{dr}{dt}&=-\frac{r}{\tau_{d}}+h\\
\frac{dh}{dt}&=-\frac{h}{\tau_{r}}+\frac{1}{\tau_{r} \tau_{d}} \sum_{t_{k}< t} \delta\left(t-t_{k}\right) 
\end{align} 
$$

単一指数関数型シナプスの場合と同様にEuler法で差分化すると 

$$
\begin{align} 
r(t+\Delta t)&=\left(1-\frac{\Delta t}{\tau_{d}}\right)r(t)+h(t)\cdot \Delta t\\ 
h(t+\Delta t)&=\left(1-\frac{\Delta t}{\tau_{r}}\right)h(t)+\frac{1}{\tau_{r}\tau_{d}} \delta_{t,t_{j k}}
\end{align}
$$

となる．

念のため, 微分方程式と元の式が一致することを確認しておこう．$t=0$のときにシナプス前細胞が発火したとし, それ以降の発火はないとする．このとき, $h(0)=1/\tau_{r}\tau_{d}, r(0)=0$ である．$h$についての微分方程式の解は

$$
\begin{equation}
h(t)=h(0)\cdot \exp\left(-\frac{t}{\tau_r}\right)    
\end{equation}
$$

となるので, これを$r$についての式に代入して

$$
\begin{equation}
\frac{dr}{dt}=-\frac{r}{\tau_{d}}+h(0)\cdot \exp\left(-\frac{t}{\tau_r}\right) 
\end{equation}
$$

となる．これを解くには両辺に積分因子$\exp({t}/{\tau_d})$をかけてから積分をするかLaplace変換をするかである．今回はLaplace変換を用いる．右辺一項目を移行した後に両辺をLaplace変換すると以下のようになる．

$$
\begin{align}
\mathcal{L}\left[\frac{dr}{dt}+r/\tau_{d}\right]&=\mathcal{L}\left[h(0)\cdot \exp\left(-t/\tau_r\right)\right]\\
sF(s)-r(0)+\frac{1}{\tau_{d}}F(s)&=\frac{h(0)}{s+1/\tau_r}\\
F(s)&=\frac{h(0)}{(s+1/\tau_r)(s+1/\tau_d)}
\end{align}
$$

ただし$r(t)$のLaplace変換を$F(s)$とした. ここで逆Laplace変換を行うと次のようになる．

$$
\begin{align}
r(t)&=\mathcal{L}^{-1}(F(s))\\
&=\mathcal{L}^{-1}\left[\frac{h(0)}{(s+1/\tau_r)(s+1/\tau_d)}\right]\\
&=\mathcal{L}^{-1}\left[\frac{h(0)}{1/\tau_r-1/\tau_d}\left(\frac{1}{s+1/\tau_d}-\frac{1}{s+1/\tau_r}\right)\right]\\
&=\frac{1}{\tau_d-\tau_r}\left[\exp(-t/\tau_d)-\exp(-t/\tau_r)\right]
\end{align}
$$

この式の最大値$r_{\max}$を求めておこう． $r(t)$を微分して0と置いた式の解$t_{\max}$を代入すれば求められる．計算すると, 

$$
\begin{equation}
t_{\max}=\dfrac{\ln(\tau_d/\tau_r)}{1/\tau_r-1/\tau_d},\ \ r_{\max}=\dfrac{1}{\tau_{d}}\cdot \left(\dfrac{\tau_{r}}{\tau_{d}}\right)^{\frac{\tau_{r}}{\tau_d-\tau_{r}}}    
\end{equation}
$$

となる．なお, $\alpha$関数の導出は逆Laplace変換をする前に$\tau=\tau_d=\tau_r$とすればよく, 

$$
\begin{align}
F_\alpha(s)&=\frac{h(0)}{(s+1/\tau)^2}\\
\alpha(t)&=\frac{t}{\tau^2}\exp\left(-\frac{t}{\tau}\right)
\end{align}
$$

となる．若干の係数の違いはあるが, 同じ形の関数が導出された． 

## 動力学シナプスモデル
## チャネル動態の動力学的表現
指数関数型シナプスとモデルの振る舞いはほぼ同一だが, 式の構成が少し異なるモデルとして**動力学モデル** (Kinetic model, またはMarkov kinetic model)がある {cite:p}`Destexhe1994-ro`．動力学モデルはHHモデルのゲート変数の式と類似した式で表される．このモデルではチャネルが開いた状態(Open)と閉じた状態(Close), および神経伝達物質(neurotransmitter)の放出状態(T)の2つの要素に関する状態がある．また, 閉$\to$開の反応速度を$\alpha$, 開$\to$閉の反応速度を$\beta$とする．このとき，これらを表す状態遷移の式は次のようになる．

$$
\begin{equation}
\text{Close}+\text{T}  \underset{\beta}{\overset{\alpha}{\rightleftharpoons}}\text{Open}    
\end{equation}
$$

ここで, シナプス動態を$r$とすると

$$
\begin{equation}
\frac{dr}{dt}=\alpha T (1-r) - \beta r
\end{equation}
$$

となる．ただし, Tはシナプス前細胞が発火したときにインパルス的に1だけ増加するとする．また, $\alpha, \beta$は速度なので, 時定数の逆数であることに注意しよう． $\alpha=2000 \text{ms}^{-1}$, $\beta=200 \text{ms}^{-1}$とすると, シナプス動態は次のようになる．

### Hodgkin-Huxleyモデルにおけるシナプスモデル
これまで明示的にスパイクの発生が表現されたモデルを用いてきたが，HHモデルでは単なる膜電位の変数があるのみである．ここでは前述した動力学的モデルを用いてHHモデルにおけるシナプス動態の記述を行う \citep{Destexhe1994-ro,Batista2014-ax}．

$r_{j}$を$j$番目のニューロンのpre-synaptic dynamicsとすると，$r_{j}$は次式に従う．

$$
\begin{equation}
\frac{\mathrm{d} r_{j}}{\mathrm{d} t}=\left(\frac{1}{\tau_{r}}-\frac{1}{\tau_{d}}\right) \frac{1-r_{j}}{1+\exp \left(-V_{j}+V_{0}\right)}-\frac{r_{j}}{\tau_{d}}
\end{equation}
$$

ただし，時定数 $\tau_r=0.5, \tau_d = 8$ (ms), 反転電位 $V_0 = -20$ (mV)とする．前節で既に$r$の描画は行ったが，パルス波を印加した場合の挙動を確認する．