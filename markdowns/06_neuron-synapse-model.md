# 第6章：ニューロンとシナプスの生物物理学的モデル
## 生物物理モデル

## コンダクタンスベースモデル※

Model neurons: From Hodgkin-Huxley to hopfield
https://link.springer.com/chapter/10.1007/3540532676_37


### Hodgkin-Huxleyモデル
**Hodgkin-Huxleyモデル** (HH モデル)は, ニューロンの膜興奮を表現する，初めに導出された数理モデルである {cite:p}`Hodgkin1952-gy`\footnote{HodgkinおよびHuxleyの論文の図をカラー化して分かりやすくした論文 {cite:p}`Hopper2022-xj` がある．}．HodgkinおよびHuxleyはヤリイカの巨大神経軸索に対して**電位固定法**(voltage-clamp)を用いた実験を行い, 実験から得られた観測結果を元にモデルを構築した {cite:p}`Schwiening2012-pi`．HHモデルには等価な電気回路モデルがあり, **膜の並列等価回路モデル** (parallel conductance model)と呼ばれている．膜の並列等価回路モデルでは, ニューロンの細胞膜をコンデンサ, 細胞膜に埋まっているイオンチャネルを可変抵抗 (動的に変化する抵抗) として置き換える．

**イオンチャネル** (ion channel)は特定のイオン(例えばナトリウムイオンやカリウムイオンなど)を選択的に通す膜輸送体の一種である．それぞれのイオンの種類において, 異なるイオンチャネルがある (同じイオンでも複数の種類のイオンチャネルがある)．また, イオンチャネルにはイオンの種類に応じて異なる**コンダクタンス**（抵抗の逆数で電流の「流れやすさ」を意味する）と**平衡電位** (equilibrium potential) がある．HHモデルでは, ナトリウム (Na$^{+}$) チャネル, カリウム (K$^{+}$) チャネル, 漏れ電流 (leak current) のイオンチャネルを仮定する．漏れ電流のイオンチャネルは当時特定できなかったチャネルで, 膜から電流が漏れ出すチャネルを意味する．なお, 現在では漏れ電流の多くはCl$^{-}$イオン(chloride ion)によることが分かっている．

それでは, 等価回路モデルを用いて電位変化の式を立ててみよう．上図において, $C_m$は細胞膜のキャパシタンス(膜容量), $I_{m}(t)$は細胞膜を流れる電流(外部からの入力電流), $I_\text{Cap}(t)$は膜のコンデンサを流れる電流, $I_\text{Na}(t)$及び $I_K(t)$はそれぞれナトリウムチャネルとカリウムチャネルを通って膜から流出する電流, $I_\text{L}(t)$は漏れ電流である．このとき, 

$$
\begin{equation}
I_{m}(t)=I_\text{Cap}(t)+I_\text{Na}(t)+I_\text{K}(t)+I_\text{L}(t)    
\end{equation}
$$

という仮定をしている．膜電位を$V(t)$とすると, Kirchhoffの第二法則 (Kirchhoff's Voltage Law)より, 

$$
\begin{equation}
\underbrace{C_m\frac {dV(t)}{dt}}_{= I_\text{Cap} (t)}=I_{m}(t)-I_\text{Na}(t)-I_\text{K}(t)-I_\text{L}(t)
\end{equation}
$$

となる．Hodgkinらはチャネル電流$I_\text{Na}, I_K, I_\text{L}$が従う式を実験的に求めた．

$$
\begin{align}
I_\text{Na}(t) &= \bar{g}_{\text{Na}}\cdot m^{3}h(V-E_{\text{Na}})\\
I_\text{K}(t) &= \bar{g}_{\text{K}}\cdot n^{4}(V-E_{\text{K}})\\
I_\text{L}(t) &= \bar{g}_{\text{L}}(V-E_{\text{L}})
\end{align}
$$

ただし, $\bar{g}_{\text{Na}}, \bar{g}_{\text{K}}$はそれぞれNa$^+$, K$^+$の最大コンダクタンスである (ここで上付き棒は上限値であることを示す)．$\bar{g}_{\text{L}}$はオームの法則に従うコンダクタンスで, Lコンダクタンスは時間的に変化はしないと仮定する．また, $m$はNa$^+$コンダクタンスの活性化パラメータ, $h$はNa$^+$コンダクタンスの不活性化パラメータ, $n$はK$^+$コンダクタンスの活性化パラメータであり, ゲートの開閉確率を表している．よって, HHモデルの状態は$V, m, h, n$の4変数で表される．これらの変数は以下の$x$を$m, n, h$に置き換えた3つの微分方程式に従う． 

$$
\begin{equation}
\frac{dx}{dt}=\alpha_{x}(V)(1-x)-\beta_{x}(V)x
\end{equation}
$$

ただし, $V$の関数である$\alpha_{x}(V),\ \beta_{x}(V)$は$m, h, n$によって異なり, 次の6つの式に従う．

$$
\begin{equation}
\begin{array}{ll}
\alpha_{m}(V)=\dfrac {0.1(V+40)}{1-\exp (-0.1(V+40))}, &\beta_{m}(V)=4\exp {(-(V+65)/18)}\\
\alpha_{h}(V)=0.07\exp {(-0.05(V+65))}, & \beta_{h}(V)=1/(1+{\exp {\left(-0.1(V+35)\right)}})\\
\alpha_{n}(V)={\dfrac {0.01(V+55)}{1-\exp {\left(-0.1(V+55)\right)}}},& \beta_{n}(V)=0.125\exp {(-0.0125(V+65))} 
\end{array}
\end{equation}
$$

これまでに説明した式を用いてHHモデルを実装する．まず必要なパッケージを読み込む．

変更しない定数を保持する `struct` の `HHParameter` と, 変数を保持する `mutable struct` の `HH` を作成する．定数は次のように設定する． 

$$
\begin{equation}
\begin{array}{l}
C_m=1.0\ \mu\textrm{F/cm}^2, \bar{g}_{\text{Na}}=120\ \textrm{mS/cm}^2, \bar{g}_{\text{K}}=36\ \textrm{mS/cm}^2, \bar{g}_{\text{L}}=0.3\ \textrm{mS/cm}^2\\
E_{\text{Na}}=50.0\ \textrm{mV}, E_{\text{K}}=-77\ \textrm{mV}, E_{\text{L}}=-54.387\ \textrm{mV} 
\end{array}
\end{equation}
$$

### Connor-Stevensモデル
HHモデルはイカの巨大軸索の活動を再現したものであるが，脊椎動物のニューロンの神経活動を再現するためにHHモデルを修正したモデル (modified Hodgkin-Huxley model) が提案されてきた．その一種である，**Connor-Stevensモデル** はHHモデルに2つ目のカリウム電流（A型カリウム電流）を追加し，低い発火率でも活動を維持できる（振動を維持できる）ようにしたものである {cite:p}`Connor1971-rs,Connor1977-qo`．ここでパラメータは{cite:p}`Dayan2005-ib`に記載のものを使用する．

$$
\begin{equation}
\begin{array}{l}
C_m=1.0\ \mu\textrm{F/cm}^2,\\ 
\bar{g}_{\text{Na}}=120\ \textrm{mS/cm}^2, \bar{g}_{\text{K}}=20\ \textrm{mS/cm}^2, \bar{g}_{\text{A}}=47.7\ \textrm{mS/cm}^2, \bar{g}_{\text{L}}=0.3\ \textrm{mS/cm}^2\\
E_{\text{Na}}=55.0\ \textrm{mV}, E_{\text{K}}=-72\ \textrm{mV}, E_{\text{A}}=-75\ \textrm{mV},E_{\text{L}}=-17\ \textrm{mV} 
\end{array}
\end{equation}
$$

$$
\begin{equation}
\begin{array}{ll}
\alpha_m=\dfrac{0.38(V+29.7)}{1-\exp (-0.1(V+29.7))} & \beta_m=15.2 \exp (-(V+54.7)/18) \\
\alpha_h=0.266 \exp (-0.05(V+48)) & \beta_h=3.8 /(1+\exp (-0.1(V+18))) \\ 
\alpha_n=\dfrac{0.02(V+45.7)}{1-\exp (-0.1(V+45.7))} & \beta_n=0.25 \exp (-0.0125(V+55.7))
\end{array}
\end{equation}
$$

$$
\begin{equation}
\frac{dx}{dt}=\frac{x_\infty-x}{\tau_x}\ (x=a, b)
\end{equation}
$$

$$
\begin{equation}
\begin{array}{l}
a_{\infty}=\left(\dfrac{0.0761 \exp [(V+94.22)/31.84]}{1+\exp ((V+1.17)/28.93)}\right)^{\frac{1}{3}}\\
\tau_a=0.3632+1.158 /(1+\exp ((V+55.96)/20.12)) \\
b_{\infty}=\left[1+\exp ((V+53.3)/14.54)\right]^{-4}\\
\tau_b=1.24+2.678 /(1+\exp [(V+50)/16.027])
\end{array}
\end{equation}
$$

### F-I曲線
HHモデルにおいて，入力電流に対する発火率がどのように変化するかを調べる．次のコードのように入力電流を徐々に増加させたときの発火率を見てみよう．

このような曲線を**frequency-current (F-I) 曲線** (または neuronal input/output (I/O) 曲線)と呼ぶ．$I_\theta$は閾値電流を意味する（ここでは発火率が1Hz以上になる点を閾値と設定している）．F-I曲線の種類に応じてType IおよびIIに分けられる\footnote{Type IIIニューロンも存在する}．


Type I neuron (with A-current) Output ring increases continuously from zero as input current exceeds the ring threshold. 
Type II neuron (without A-current)
Output ring increases discontinuously as input current exceeds the ring threshold.

### 全か無かの法則の反例
ニューロンは電流が流入することで膜電位が変化し, 膜電位がある一定の閾値を超えると活動電位が発生する, というのはニューロンの活動電位発生についての典型的な説明である．膜電位が閾値を超えるか超えないかで活動電位の発生が決まるという法則を， **全か無かの法則** (all-or-none principle) と呼ぶ．後に説明するLIFモデルなどは，全か無かの法則に従って神経活動のモデル化を行っている．しかし，この全か無かの法則の法則は必ずしも成立するわけではない．反例として **抑制後リバウンド** (Postinhibitory rebound; PIR)という現象がある．抑制後リバウンドは過分極性の電流の印加を止めた際に膜電位が静止膜電位に回復するのみならず, さらに脱分極をして発火をするという現象である．この時生じる発火を**リバウンド発火** (rebound spikes)と呼ぶ．この現象が生じる要因として**アノーダルブレイク** (anodal break, またはanode break excitation; ABE)や，遅いT型カルシウム電流 (slow T-type calcium current) が考えられている {cite:p}`Chik2004-ka`．HH モデルはこのうちアノーダルブレイクを再現できるため, シミュレーションによりどのような現象か確認してみよう．これは入力電流を変更するだけで行える．

なぜこのようなことが起こるか, というと過分極の状態から静止膜電位へと戻る際にNa$^+$チャネルが活性化 (Na$^+$チャネルの活性化パラメータ$m$が増加し, 不活性化パラメータ$h$が減少)し, 膜電位が脱分極することで再度Na$^+$チャネルが活性化する, というポジティブフィードバック過程(**自己再生的過程**)に突入するためである (もちろん, この過程は通常の活動電位発生のメカニズムである)． この際, 発火に必要な閾値が膜電位の低下に応じて下がった, ということもできる．

なお，PIRに関連する現象として抑制後促通 (Postinhibitory facilitation; PIF)がある．これは抑制入力の後に興奮入力がある一定の時間内で入ると発火が起こるという現象である {cite:p}`Dodla2006-fj`．

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

LIFモデルは力学系の視点では，区分的に滑らかな力学系 (piecewise-smooth dynamical system; PWS) の一種であると解釈できる．このような系の代表例としては壁に衝突する球（粒子）の運動が挙げられる．

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

## マルチコンパートメントモデル


L5-PC minimal model

Hodgkin-Huxleyの定式化

$k$ をイオンの種類として，

$$
I_k:=g_k m_k^x h_k^y (V_i - E_k)
$$

$g_k$ は最大コンダクタンス，$m_k, h_k$ はそれぞれ活性化ゲート変数，不活性化ゲート変数である．$x, y$ は指数である．$V_i$ は $i$ 番目のコンパートメントの膜電位であり，$E_k$ は平衡電位である．
 
https://github.com/beaherrera/2-compartments_L5-PC_model/tree/master/IonicCurrents

ケーブル方程式の離散化

https://neuronaldynamics.epfl.ch/online/Ch3.S4.html

https://github.com/orena1/NEURON_tutorial/tree/master
https://github.com/orena1/NEURON_tutorial/blob/master/Jupyter_notebooks/Layer_5b_pyramidal_cell_Calcium_Spike.ipynb

Ball and Stick model

 E. Hay, S. Hill, F. Schürmann, H. Markram and I. Segev (2011-07) Models of neocortical layer 5b pyramidal cells capturing a wide range of dendritic and perisomatic active properties. PLoS Comput Biol 7 (7), pp. e1002107
 

three compartment model
https://pmc.ncbi.nlm.nih.gov/articles/PMC4516889/

https://www.jneurosci.org/content/40/44/8513.full#ref-100

https://www.nature.com/articles/s41467-019-11537-7
https://www.science.org/doi/10.1126/science.1127240

https://www.jneurosci.org/content/40/44/8513.full#sec-2


神経細胞の電気的活動を詳細に記述するためには，単一の点としてニューロンをモデル化する単純なモデル（例：leaky integrate-and-fireモデル）では不十分である．特に，樹状突起や軸索といった構造的に異なる部分の電気的性質を記述するためには，multi compartment model（多区画モデル）と呼ばれる手法が用いられる．このモデルでは，ニューロン全体を電気回路として捉え，各構造（区画）を電気的に独立した要素として記述し，それらを電気的に接続することで，ニューロン全体の動態を近似する．

各区画（コンパートメント）は，膜容量 $C_m$，漏洩電導 $g_L$，静止電位 $E_L$ を備えたRC回路として表現される．隣接するコンパートメント同士は，軸索や樹状突起を介した軸索流によって結合され，その伝導は軸内抵抗 $R_a$ または電導 $g_{ij}$ を通じて記述される．

コンパートメント $i$ における膜電位 $V_i(t)$ の時間発展は，ケーブル方程式を離散化した以下の形式で表される：

$$
C_m \frac{dV_i}{dt} = -g_L (V_i - E_L) + \sum_{j \in \mathcal{N}(i)} g_{ij} (V_j - V_i) + I^{\text{ext}}_i(t)
$$

ここで，\(\mathcal{N}(i)\) はコンパートメント \(i\) に隣接するコンパートメントの集合，\(g_{ij}\) は区画 \(i\) と \(j\) を接続する電導，\(I^{\text{ext}}_i(t)\) は外部から注入される電流を表す．この方程式は，すべてのコンパートメントに対して定義され，結果として連立微分方程式系が得られる．

このmulti compartment modelを用いることで，樹状突起でのシナプス入力の時空間的な統合や，軸索起始部で発生した活動電位（action potential, AP）が樹状突起へ逆行性に伝搬する現象（back-propagating action potential, bAP）を記述・再現することが可能となる．bAPは，活動電位が軸索起始部で発生した後，ナトリウムチャネルやカリウムチャネルの存在により樹状突起へと逆向きに伝搬するものであり，樹状突起上のシナプス可塑性に重要な役割を果たすと考えられている．

モデル内でこの現象を再現するには，樹状突起区画にも活動電位の伝搬に関与する電位依存性ナトリウムチャネル（\(I_{\text{Na}}\)）やカリウムチャネル（\(I_{\text{K}}\)）を含めたHodgkin-Huxley型のイオン電流モデルを導入する必要がある．例えば，コンパートメント \(i\) における電流項は以下のように拡張される：

$$
C_m \frac{dV_i}{dt} = -g_L (V_i - E_L) - I_{\text{Na}, i}(V_i, m_i, h_i) - I_{\text{K}, i}(V_i, n_i) + \sum_{j \in \mathcal{N}(i)} g_{ij} (V_j - V_i) + I^{\text{ext}}_i(t)
$$

ここで，\(m_i, h_i, n_i\) はイオンチャネルのゲーティング変数であり，それぞれ別の微分方程式に従って時間発展する．これにより，活動電位の発生とその伝播，さらに逆行性伝播が自然にモデルに組み込まれることになる．


## 点過程モデル
これまで紹介したモデルでは，入力に対する膜電位などの時間変化に基づき発火が起こるかどうか，ということを考えてきた．この節では，発火が生じるまでの過程を考慮せず，発火の時間間隔 (inter-spike interval; ISI) の統計による現象論的モデルを考える．これを**Inter-spike interval (ISI)** モデルと呼ぶ．

神経細胞の発火のようなイベントがいつ（あるいはどこで）生じるのか，を記述する確率過程を**点過程** (point process) と呼ぶ．

ISIモデルは**点過程** (point process) という統計的モデルに基づいており，各モデルにはISIが従う分布の名称がついている．

時間に応じて変化する確率変数のことを**確率過程(stochastic process)** という．さらに確率過程の中で，連続時間軸上において離散的に生起する点事象の系列を**点過程(point process)** という．

この節では，使用頻度の高い **ポアソン過程 (Poisson process) モデル**，ポアソン過程モデルにおいて不応期を考慮した **死時間付きポアソン過程 (Poisson process with dead time, PPD) モデル**，皮質の定常発火においてポアソン過程モデルよりも当てはまりがよいとされる **ガンマ過程 (Gamma process) モデル**について説明する．

なお，SNNにおいて，ISIモデルは主に画像入力の際に**連続値からスパイク列へのエンコード**に用いられる．これに限らず入力として用いられることが多い．

この節は {cite:p}`Shimazaki_undated-ko`, {cite:p}`Pachitariu2010-pm` を参考に執筆した．

### ポアソン過程モデル
ポアソン過程 (Poisson process)は点過程の1つである．ポアソン過程モデルはスパイクの発生をポアソン過程でモデル化したもので，このモデルによって生じるスパイクをポアソンスパイク(Poisson spike)と呼ぶ．ポアソン過程では，時刻$t$までに起こった点の数$N(t)$はポアソン分布に従う．すなわち，点が起こる確率が強度$\lambda$のポアソン分布に従う場合, 時刻$t$までに事象が$n$回起こる確率は$P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$となる． 

ポアソン過程において点が起こる回数がポアソン分布に従うことは，ポアソン過程という名称の由来となっている．これを定義とする場合もあれば，次の4条件を満たす点過程をポアソン過程とするという定義もある．

1. 時刻0における初期の点の数は0 : $P[N(0)=0]=1$ 
2. $[t, t+\Delta t)$に点が1つ生じる確率 : $P[N(t+\Delta t)-N(t)=1]=\lambda(t)\Delta t+o(\Delta t)$
3. 微小時間$\Delta t$の間に点は2つ以上生じない : $P[N(t+\Delta t)-N(t)=2]=o(\Delta t)$
4. 任意の時点$t_1 < t_2 < \cdots< t_n$に対して，増分 $N(t_2)-N(t_1), N(t_3)-N(t_2), \cdots, N(t_n)-N(t_{n−1})$は互いに独立である．

ただし, $o(\cdot)$はLandauの記号(Landauのsmall o)であり, $o(x)$は$x\to 0$のとき，$o(x)/x\to 0$となる微小な量を表す．ポアソン過程に従ってスパイクが生じるとする場合，条件2の強度関数$\lambda(t)$は**発火率**を意味する (また実装において有用)．条件3は不応期より小さいタイムステップにおいては，1つのタイムステップにおいて1つしかスパイクは生じないということを表す．条件4はスパイクは独立に発生する，ということを意味する．また，これらの条件から$N(t)$の分布は強度母数$\lambda(t)$のポアソン分布に従うことが示せる．

強度関数(点がスパイクの場合，発火率)が$\lambda(t)=\lambda$ (定数)となる場合は点の時間間隔(点がスパイクの場合，ISI)の確率変数$T$が強度母数$\lambda$の **指数分布**に従う．なお，指数分布の確率密度関数は確率変数を$T$とするとき，

$$
\begin{equation}
f(t;\lambda )=\left\{{\begin{array}{ll}\lambda e^{-\lambda t}&(t\geq 0)\\0&(t<0)\end{array}}\right.
\end{equation}
$$

となる．このことは4条件とChapman-Kolmogorovの式により求められるが，ややこしいので, $P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$から導出できることを簡単に示す．指数分布の累積分布関数を$F(t; \lambda)$とすると，

$$
\begin{equation}
F(t; \lambda) = P(T< t)=1-P(T\geq t)=1-P(N(t)=0)=1-e^{-\lambda t}
\end{equation}
$$

となる．よって

$$
\begin{equation}
f(t; \lambda)=\frac{dF(t; \lambda)}{dt}=\lambda e^{-\lambda t}
\end{equation}
$$

が成り立つ．

### 定常ポアソン過程
ここからポアソン過程によるスパイクのシミュレーションを実装する．実装方法にはISIが指数分布に従うことを利用したものと，ポアソン過程の条件2を利用したものの2通りがある．実装は後者が楽で計算量も少ないが，後のガンマ過程のために前者の実装を先に行う．

#### ISIの累積により発火時刻を求める手法
ISIが指数分布に従うことを利用してポアソン過程モデルの実装を行う．まずISIを指数分布に従う乱数とする．次にISIを累積することで発火時刻を得る．最後に発火時間を整数値に丸めてindexとすることで$\{0, 1\}$のスパイク列が得られる．ISIの取得には`Random.randexp()`を用いる．この関数は scale 1の指数分布に従う乱数を返す．このscaleは指数分布の確率密度関数を$f(t; \frac{1}{\beta}) = \frac{1}{\beta} e^{-t/\beta}$とした際の$\beta = 1/\lambda$である(この時，平均は$\beta$となる)．よって発火率を`fr`(1/s), 単位時間を`dt`(s)としたときのISIは `isi = 1/(fr*dt) * randexp()`として得ることができる．

まず必要なパッケージを読み込む．

#### $\Delta t$ 間の発火確率が $\lambda\Delta t$ であることを利用する方法
次に2番目のポアソン過程モデルの実装を行う．こちらは$\lambda$を発火率とした場合, 区間 $[t, t+\Delta t)$ の間にポアソンスパイクが発生する確率は$\lambda \Delta t$となることを利用する．これはポアソン過程の条件だが，ポアソン分布から導けることを簡単に示しておく．事象が起こる確率が強度$\lambda$のポアソン分布に従う場合, 時刻$t$までに事象が$n$回起こる確率は$P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$となる．よって, 微小時間$\Delta t$において事象が$1$回起こる確率は

$$
\begin{equation}
P[N(\Delta t)=1]=\dfrac{\lambda \Delta t}{1 !} e^{-\lambda \Delta t}\simeq \lambda \Delta t+o(\Delta t)
\end{equation}
$$

となる．ただし, $e^{-\lambda \Delta t}$についてはTaylor展開による近似を行っている．このことから, 一様分布$U(0,1)$に従う乱数$\xi$を取得し, $\xi<\lambda dt$なら発火$(y=1)$, それ以外では$(y=0)$となるようにすればポアソンスパイクを実装できる．

### 非定常ポアソン過程
これまでの実装は発火率$\lambda$が一定であるとする，定常ポアソン過程 (homogeneous poisson process)であったが，ここからは発火率$\lambda(t)$が時間変化するとする**非定常ポアソン過程** (inhomogeneous poisson process)について考える．とはいえ，定常ポアソン過程における発火率を，時間についての関数で置き換えるだけで実装できる．以下は $\lambda(t)=\sin^2(\alpha t)$（ただし$\alpha$は定数）とした場合の実装である．

### 死時間付きポアソン過程モデル
ポアソン過程は簡易的で有用だが，不応期を考慮していない．そのため，時には生理的範疇を超えたバースト発火が起こる場合もある（複数のニューロンからの発火の重ね合わせ(superposition)であると考えることもできる．）．そこで，ポアソン過程において不応期のようなイベントの生起が起こらない **死時間** (dead time) \footnote{例えば，ガイガー・カウンター(Geiger counter)などの放射線の検出器には放射線の到達を機器の物理的特性として検出できない時間(つまり死時間)がある．そのため放射線の到達数がポアソン分布に従うとした場合，放射線測定装置のモデルとしてPPDが用いられる．}を考慮した**死時間付きポアソン過程** (Poisson process with dead time; PPD または dead time modified Poisson process) というモデルを導入する．

実装においてはLIFニューロンの時と同じような不応期の処理をする．つまり，現在が不応期かどうかを判断し，不応期なら発火を許可しないようにする．

### ガンマ過程モデル
ガンマ過程(gamma process)は点の時間間隔がガンマ分布に従うとするモデルである．ガンマ過程はポアソン過程よりも皮質における定常発火への当てはまりが良いとされている {cite:p}`Shinomoto2003-lz,Maimon2009-mb`．時間間隔の確率変数を$T$とした場合，ガンマ分布の確率密度関数は

$$
\begin{equation}
f(t;k,\theta) =  t^{k-1}\frac{e^{-t/\theta}}{\theta^k\Gamma(k)}
\end{equation}
$$

と表される．ただし，$t > 0$であり， 2つの母数は$k, \theta > 0$である．また，$\Gamma (\cdot)$ はガンマ関数であり，

$$
\begin{equation}
\Gamma (k)=\int _{0}^{\infty }x^{k-1}e^{-x}\,dx
\end{equation}
$$

と定義される．ガンマ分布の平均は$k\theta$だが，発火率はISIの平均の逆数なので，$\lambda=1/k\theta$となる．また，$k=1$のとき，ガンマ分布は指数分布となる．さらに$k$が正整数のとき，ガンマ分布はアーラン分布となる．

ガンマ過程モデルの実装はポアソン過程モデルのISIを累積する手法と同様に書くことができる．ただしこの時，`Distributions.jl`を用いる．基本的には`randexp(shape)`を`rand(Gamma(a,b), shape)`に置き換えればよい (もちろん多少の修正は必要とする)．

なお，前述したようにガンマ過程モデルの方がポアソン過程モデルよりも皮質ニューロンのモデルとしては優れているが，入力画像のエンコーディングをガンマ過程モデルにすることでSNNの認識精度が向上するかどうかはまだ十分に研究されていない．また，{cite:p}`Deger2012-ai`ではPPDやガンマ過程の重ね合わせによるスパイク列を生成するアルゴリズムを考案している．

### 発火率の推定
神経細胞がある時刻 $t_i$ にスパイクを発生させたとき，そのスパイク列は**ディラックのデルタ関数**を用いて

$$
S(t) = \sum_{i} \delta(t - t_i)
$$

と記述される．ここで，$\delta(t - t_i)$ は時刻 $t_i$ における理想的なスパイク（瞬間的な事象）を表しており，$\int_{-\infty}^{\infty} \delta(t - t_i) dt = 1$ を満たす．このスパイク列から滑らかな発火率を得るには，ある**時間窓** $h(t)$ による平滑化（畳み込み）を行う：

$$
r(t) = (S * h)(t) = \int_{-\infty}^{\infty} S(\tau) h(t - \tau) d\tau = \sum_i h(t - t_i)
$$

この $r(t)$ が**時刻 $t$ における発火率**を定義する関数である．

時間窓関数 $h(t)$ としては，正規分布型（ガウスカーネル）や指数関数減衰型などがしばしば用いられる：

- ガウス型： $h(t) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{t^2}{2\sigma^2} \right)$
- 指数減衰型： $h(t) = \frac{1}{\tau} \exp\left( -\frac{t}{\tau} \right) \cdot \Theta(t)$（\(\Theta(t)\) はステップ関数）

一方，神経集団の統計的性質を扱う場合には，**スパイク列を確率的な点過程**として扱い，その条件付き強度関数（conditional intensity function）を発火率とみなす．すなわち，あるニューロンのスパイク列が条件付きポアソン過程であるとき，発火率 $\lambda(t)$ はその時点でスパイクが起こる**瞬間的な確率密度**として定義される：

$$
\lambda(t) = \lim_{\Delta t \to 0} \frac{\mathbb{E}[N(t + \Delta t) - N(t) \mid \mathcal{H}_t]}{\Delta t}
$$

ここで，$\mathcal{H}_t$ は時刻 $t$ までのスパイク履歴，$N(t)$ は時刻 $t$ までのスパイク数である．この形式は発火率をより確率論的に捉えた定式化であり，LNPモデルやGLMなどで重要である．


## ゲイン調整と四則演算

## シナプスの生理とCurrent/Conductance-based シナプス
### シナプスの形態と生理
スパイクが生じたことによる膜電位変化は軸索を伝播し, **シナプス**という構造により, 次のニューロンへと興奮が伝わる. このときの伝達の仕組みとして, シナプスには**化学シナプス**(chemical synapse)とGap junctionによる**電気シナプス**(electrical synapse)がある．  

化学シナプスの場合, シナプス前膜からの**神経伝達物質**の放出, シナプス後膜の受容体への神経伝達物質の結合, イオンチャネル開口による**シナプス後電流**(postsynaptic current; PSC)の発生, という過程が起こる．

しかし, これらの過程を全てモデル化するのは計算量がかなり大きくなるので, 基本的には簡易的な現象論的なモデルを用いる．

このように, シナプス前細胞のスパイク列(spike train)は次のニューロンにそのまま伝わるのではなく, ある種の時間的フィルターをかけられて伝わる．このフィルターを**シナプスフィルター**(synaptic filter)と呼ぶ．本章では, このようにシナプス前細胞で生じた発火が, シナプス後細胞の膜電位に与える過程のモデルについて説明する．


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

## 短期的シナプス可塑性

シナプス前活動に応じて**シナプス伝達効率** (synaptic efficacy) が動的に変化する性質を**短期的シナプス可塑性** (Short-term synaptic plasticity; STSP) といい，このような性質を持つシナプスを**動的シナプス** (dynamical synapses)と呼ぶ．シナプス伝達効率が減衰する現象を短期抑圧 (short-term depression; STD)，増強する現象を短期促通(short-term facilitation; STF)という．さらにそれぞれに対応するシナプスを減衰シナプス，増強シナプスという．ここでは{cite:p}`Mongillo2008-kk`および{cite:p}`Orhan2019-rq`で用いられている定式化を使用する．

$$
\begin{align}
\frac{\mathrm{d} x(t)}{\mathrm{d} t}=\frac{1-x(t)}{\tau_{x}}-u(t) x(t) r(t) \Delta t \\
\frac{\mathrm{d} u(t)}{\mathrm{d} t}=\frac{U-u(t)}{\tau_{u}}+U(1-u(t)) r(t) \Delta t
\end{align}
$$

ただし，$x$を利用可能な神経伝達物質の量, $u$を利用されている神経伝達物質の量(the neurotransmitter utilization), $\tau_x$は神経伝達物質の時定数 , $\tau_u$はutilization, $U$はincrement , $\Delta t$を時間幅とする．ここでは$\tau_x=$(200 ms/1,500 ms; facilitating/depressing),  $\tau_u=$(1,500 ms/200 ms; facilitating/depressing), $U=$(0.15/0.45; facilitating/depressing), $\Delta t=$10msとする．

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
