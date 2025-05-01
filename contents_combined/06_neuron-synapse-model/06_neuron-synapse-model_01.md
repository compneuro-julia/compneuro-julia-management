# 第6章：ニューロンとシナプスの生物物理学的モデル
## 神経細胞のコンダクタンスモデル

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