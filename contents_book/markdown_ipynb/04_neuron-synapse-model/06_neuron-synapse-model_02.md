## 発火ダイナミクスの縮約モデル
### FitzHugh-Nagumoモデル
前節では神経活動のダイナミクスを微分方程式で表したHodgkin-Huxley(HH)モデルを扱った．HHモデルの特徴は，4変数で構成され，各変数が膜電位およびNaチャネルやKチャネルなどの活性/不活性状態を意味することである．このHHモデルをより簡易化し，2変数で神経活動の興奮とその伝播を表そうと提案されたのが**FitzHugh-Nagumo (FHN)モデル** である．FHNモデルはvan der Pol振動子をFitzHughが修正し{cite:p}`FitzHugh1955-bx` {cite:p}`Fitzhugh1961-fp`，南雲らによりトンネル (江崎) ダイオードを用いて電子回路上に実装\footnote{神経活動を再現する電子回路を**ニューリスタ** （neuristor）という．}された {cite:p}`Nagumo1962-ob`という経緯がある．FHNモデルは以下で表される．

$$
\begin{align} 
\frac{dv}{dt} &= c\left(v-\frac{v^3}{3}-u+I_e\right)\\ 
\frac{du}{dt} &= v-bu+a 
\end{align}
$$

$v$は膜電位で，$u$は回復変数(recovery variable)と呼ばれる． FitzHughにより，HHモデルにおける$(V, m)$および$(n, h)$がそれぞれFHNモデルの$v$および$u$に対応すると説明されている {cite:p}`Fitzhugh1961-fp` \footnote{HHモデルにおける$V$と$m$は強い正の相関があり，$n$と$h$は強い負の相関があるため，それぞれの変数の組は1つの変数に縮約されうる．}．$a,b,c$は定数であり，$a=0.7, b=0.8, c=10$がよく使われる．$I_e$は外部刺激電流に対応する．

## 積分発火モデル
### Leaky integrate-and-fire モデル
Lapicque's introduction of the integrate-and-fire model neuron (1907)

生理学的なイオンチャネルの挙動は考慮せず, 入力電流を膜電位が閾値に達するまで時間的に積分するというモデルを**integrate-and-fire** (IF, 積分発火)モデル という．さらに, IFモデルにおいて膜電位の漏れ (leak) \footnote{この漏れはイオンの拡散などによるもの． }も考慮したモデルを **leaky integrate-and-fire** (LIF, 漏れ積分発火) モデル と呼ぶ．ここではLIFモデルのみを取り扱う．

ニューロンの膜電位を$V_m(t)$, 静止膜電位を$V_\text{rest}$, 入力電流\footnote{シナプス入力による電流がどうなるかは，第三章「シナプス伝達のモデル」で扱う．}を$I(t)$, 膜抵抗を$R_m$, 膜電位の時定数を$\tau_m\ (=R_m \cdot C_m)$とすると, 式は次のようになる\footnote{$(V_{m}(t)-V_\text{rest})$の部分は膜電位の基準を静止膜電位としたことにして, 単に$V_m(t)$だけの場合もある． また, 右辺の$RI(t)$の部分は単に$I(t)$とされることもある． 同じ表記だが, この場合の$I(t)$はシナプス電流に比例する量, となっている(単位はmV)． }．

$$
\begin{equation}
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_mI(t)
\end{equation}
$$

ここで, $V_m$が閾値(threshold)\footnote{thから始まるので文字$\theta$が使われることもある．}$V_{\text{th}}$を超えると, 脱分極が起こり, 膜電位はピーク電位 $V_{\text{peak}}$まで上昇する．発火後は再分極が起こり, 膜電位はリセット電位 $V_{\text{reset}}$まで低下すると仮定する\footnote{リセット電位は静止膜電位と同じ場合もあれば, 過分極を考慮して静止膜電位より低めに設定することもある．}．発火後, 一定の期間$\tau_{\text{ref}}$ の間は膜電位が変化しない\footnote{実装によっては不応期の間は膜電位の変化は許容するが発火は生じないようにすることもある．} とする．これを **不応期** (refractory time period) と呼ぶ．

LIFモデルは力学系の視点では，区分的に滑らかな力学系 (piecewise-smooth dynamical system; PWS) の一種であると解釈できる．このような系の代表例としては壁や床に衝突する球（粒子）の運動が挙げられる．要するに，閾値の設定は境界条件であると言える．

以上を踏まえてLIFモデルを実装してみよう．まず必要なパッケージを読み込む．

閾値をadaptiveにする．

### 解析的計算によるF-I curveの描画
ここまでは数値的なシミュレーションによりF-I curveを求めた．以下では解析的にF-I curveの式を求めよう．具体的には，一定かつ持続的な入力電流を$I$としたときのLIFニューロンの発火率(firing rate)が

$$
\begin{equation}
\text{rate}\approx \left(\tau_m \ln \frac{R_mI}{R_mI＋V_\text{rest}-V_{\text{th}}}\right)^{-1}
\end{equation}
$$

と近似できることを示す．まず，$t=t_1$にスパイクが生じたとする．このとき, 膜電位はリセットされるので$V_m(t_1)=V_\text{rest}$である(リセット電位と静止膜電位が同じと仮定する)．$[t_1, t]$における膜電位はLIFの式を積分することで得られる．

$$
\begin{equation}
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_m I
\end{equation}
$$

の式を積分すると, 

$$
\begin{align}
\int_{t_1}^{t} \frac{\tau_m dV_m}{R_mI＋V_\text{rest}-V_m}&=\int_{t_1}^{t} dt\\
\ln \left(1-\frac{V_m(t)-V_\text{rest}}{R_mI}\right)&=-\frac{t-t_1}{\tau_m} \quad (\because V_m(t_1)=V_\text{rest})\\
V_m(t) &=V_\text{rest} + R_mI\left[1-\exp\left(-\frac{t-t_1}{\tau_m}\right)\right] 
\end{align}
$$

となる．$t>t_1$における初めのスパイクが$t=t_2$に生じたとすると, そのときの膜電位は$V_m(t_2)=V_{\text{th}}$である (実際には閾値以上となっている場合もあるますが近似する)．$t=t_2$を上の式に代入して

$$
\begin{align}
V_{\text{th}}&=V_\text{rest} + R_mI\left[1-\exp\left(-\frac{t_2-t_1}{\tau}\right)\right] \\
T&= t_2-t_1 = \tau_m \ln \frac{R_mI}{R_mI＋V_\text{rest}-V_{\text{th}}}
\end{align}
$$

となる．ここで$T$は2つのスパイクの時間間隔 (spike interval)である．$t_1\leq t<t_2$におけるスパイクは$t=t_1$時の1つなので, 発火率は$1/T$となる．よって

$$
\begin{equation}
\text{rate}\approx \frac{1}{T}=\left(\tau_m \ln \frac{R_mI}{R_mI＋V_\text{rest}-V_{\text{th}}}\right)^{-1}
\end{equation}
$$

となる．不応期$\tau_{\text{ref}}$を考慮すると, 持続的に入力がある場合は単純に$\tau_{\text{ref}}$だけ発火が遅れるので発火率は$1/(\tau_{\text{ref}}+T)$となる．

それではこの式に基づいてF-I curveを描画してみよう．

## Izhikevich モデル※
### Izhikevich モデルの定義
**Izhikevich モデル** (または**Simple model**)はHHモデルとLIFモデルの中間のようなモデルである{cite;p}`Izhikevich2003-by`．HHモデルのような生理学的な知見に基づいたモデルは実際のニューロンの発火特性をよく再現できるが，式が複雑化するため，数学的な解析が難しく，計算量が増えるために大規模なシミュレーションも困難となる\footnote{これに関しては必ずしも正しくない．計算機の発達によりHHモデルで大きなモデルをシミュレーションすることも可能である．}．そこで，生理学的な正しさには目をつぶり，生体内でのニューロンの発火特性を再現するモデルが求められた．その特徴を持つのがIzhikevich モデルである (以下ではIzモデルと表記する)．Izモデルは 2変数しかない\footnote{数値計算をする上では簡易的だが，if文が入るために解析をするのは難しくなる．}簡素な微分方程式だが, 様々なニューロンの活動を模倣することができる{cite:p}`Izhikevich2004-xf`．定式化には主に2種類ある．まず，{cite;p}`Izhikevich2003-by`で提案されたのが次式である．

$$
\begin{align}
\frac{dv(t)}{dt}&=0.04v(t)^2 + 5v(t)+140-u(t)+I(t) \\
\frac{du(t)}{dt}&=a(bv(t)-u(t))
\end{align} 
$$

ここで，$v$と$u$が変数であり, $v$は膜電位(membrane potential;単位はmV), $u$は回復電流(recovery current; 単位はpA) である．ここでの「回復」というのは脱分極した後の膜電位が静止膜電位へと戻る，という意味である．対義語はactivationで膜電位の上昇を意味する．$u$は$v$の導関数において$v$の上昇を抑制するように$-u$で入っているため，$u$としてはK$^+$チャネル電流やNa$^+$チャネルの不活性化動態などが考えられる．また，$a$は回復時定数(recovery time constant; 単位はms$^{-1}$)の逆数 (これが大きいと$u$が元に戻る時間が短くなる), $b$は$u$の$v$に対する感受性(共鳴度合い,  resonance; 単位はpA/mV)である．

この式は簡便だが，生理学的な意味づけが分かりにくい．改善された式として{cite:p}`Izhikevich2007-ff`のChapter 8で紹介されているのが次式である．

$$
\begin{align}
C\frac{dv(t)}{dt}&=k\left(v(t)-v_r\right)\left(v(t)-v_t\right)-u(t)+I(t) \\
\frac{du(t)}{dt}&=a\left\{b\left(v(t)-v_{r}\right)-u(t)\right\}
\end{align} 
$$

ここで，$C$は膜容量(membrane capacitance; 単位はpF), $v_r$は静止膜電位(resting membrane potential; 単位はmV), $v_t$は閾値電位(instantaneous threshold potential; 単位はmV), $k$はニューロンのゲインに関わる定数で，小さいと発火しやすくなる (単位はpA/mV)．以後はこちらの式を用いる．

Izモデルの閾値の取り扱いはLIFモデルと異なり，HHモデルに近い．LIFモデルでは閾値を超えた時に膜電位をピーク電位まで上昇させ (この過程は無くてもよい)，続いて膜電位をリセットする．Izモデルの閾値は$v_t$だが, 膜電位のリセットは閾値を超えたかで判断せず，膜電位$v$がピーク電位$v_{\text{peak}}$になったとき（または超えた時）に行う．そのためIzモデルの実際の閾値は膜電位の挙動が変化する(発火状態に移行する)，つまり分岐(bifurcation) が生じる点であり，パラメータの閾値$v_t$との間には差異がある．

さて，膜電位がピーク電位$v_{\text{peak}}$に達したとき (すなわち `if` $v \geq v_{\text{peak}}$)，$u, v$を次のようにリセットする\footnote{バースト発火(bursting)の挙動を表現するためには，速い回復変数(fast recovery variable)と遅い回復変数(slow recovery variable)の2つが必要となる(従って膜電位も合わせて全部で3変数必要)．一方で，IzモデルではLIFモデルのようなif文によるリセットを用いているため，速い回復変数が必要なく，遅い回復変数$u$のみでバースト発火を表現できる．}．

$$
\begin{align} 
u&\leftarrow u+d\\
v&\leftarrow v_{\text{reset}}
\end{align}
$$

とする．ただし, $v_{\text{reset}}$は過分極を考慮して静止膜電位$v_r$よりも小さい値とする．また，$d$はスパイク発火中に活性化される正味の外向き電流の合計を表し，発火後の膜電位の挙動に影響する (単位はpA)．

以上を踏まえて, シミュレーションを行う．まず，必要なパッケージを読み込む．