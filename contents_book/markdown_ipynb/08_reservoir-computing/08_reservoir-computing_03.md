## FORCE法

RLSを用いる．オンライン学習則

自己回帰的なシステムの近似に強い．

Reservoir Computingにおける教師あり学習の手法の1つとして，**FORCE法** と呼ばれるものがあります．**FORCE** (First-Order Reduced and Controlled Error) 法は(Sussillo \& Abbott, 2009)で提案された学習法で，元々は発火率ベースのRNNに対するオンラインの学習法です (具体的な方法については次節で解説します)．さらに(Nicola \& Clopath, 2017)はFORCE法がRecurrent SNNの学習に直接的に使用できる，ということを示しました．この章では(Nicola \& Clopath, 2017)の手法を用いてReservoir ComputingとしてのRecurrent SNNの教師あり学習を行います．

FORCE法は**RLSフィルタ** (recursive least squares filter, 再帰的最小二乗法フィルタ)という**適応フィルタ** (adaptive filter)の一種を学習するアルゴリズムを，RNNの学習に適応したものである．なお，(Sussillo \& Abbott, 2009)ではDelta則を用いることで，RLS法を用いない重みの更新則も紹介されている．

### RLS法による重みの更新
誤差を 

$$
\begin{equation}
\boldsymbol{e}(t)=\hat{\boldsymbol{x}}(t)-\boldsymbol{x}(t)=\phi(t-\Delta t)^\top \boldsymbol{r}(t)-\boldsymbol{x}(t)    
\end{equation}
$$

とした場合\footnote{実際にはこれは真の誤差ではなく，事前誤差(apriori error)と呼ばれるものである．真の誤差は $\phi(t)^\top \boldsymbol{r}(t)-\boldsymbol{x}(t)$ と表される．}，出力重み$\phi$を次の式で更新する．

$$
\begin{align}
\phi(t)&=\phi(t-\Delta t)-P(t) \boldsymbol{r}(t)\boldsymbol{e}(t)^\top\\
P(t)&=P(t-\Delta t)-\frac{P(t-\Delta t) \boldsymbol{r}(t) \boldsymbol{r}(t)^\top P(t-\Delta t)}{1+\boldsymbol{r}(t)^\top P(t-\Delta
t) \boldsymbol{r}(t)} 
\end{align}
$$

また，初期値は $\phi(0)=0, P(0)=I_{N}\lambda^{-1}$ である．$I_{N}$ は $N$ 次の単位行列を意味する．$\lambda$は正則化のための定数である．



### 発火率モデルにおけるFORCE法

### スパイキングモデルにおけるFORCE法
#### Recurrent SNNに正弦波を学習させる
今回はRecurrent SNNのニューロンの活動をデコードしたものが正弦波となるように(すなわち正弦波を教師信号として)訓練することを目標とします．先になりますが，結果は図のようになります．

#### ネットワークの構造と教師信号
ネットワークの構造は図のようになっています．ネットワークには特別な入力があるわけではなく，再帰的な入力によって活動が持続しています(膜電位の初期値をランダムにしているため開始時に発火するニューロン\footnote{ここでの「ニューロン」はこれ以後も含め，Reservoirのユニットを指します．}があり，またバイアス電流もあります)．
まず，Reservoirニューロンの数を$N$とし，出力の数を$N_\text{out}$とします．$i$番目のニューロンの入力はバイアス電流を$I_{\text{Bias}}$として，

$$
\begin{equation}
I_i=s_i+I_{\text{Bias}}    
\end{equation}
$$

と表されます．ただし，$s_i$は 

$$
\begin{equation}
s_{i}=\sum_{j=1}^{N} \omega_{i j} r_{j}    
\end{equation}
$$

として計算されます．$r_j$が$j$番目のニューロンの出力(シナプスフィルターをかけられたスパイク列), $\omega_{i j}$は$j$番目のニューロンから$i$番目のニューロンへの結合重みを意味します．
次にニューロンの活動$r_j$を重み$\phi\in \mathbb{R}^{N\times N_\text{out}}$で線形にデコードし，その出力$\hat{\boldsymbol{x}}(t)$を教師信号$\boldsymbol{x}(t)$に近づけます．すなわち，

$$
\begin{equation}
\hat{\boldsymbol{x}}(t)=\sum_{j=1}^{N} \boldsymbol{\phi}_j r_{j}=\phi^\top\boldsymbol{r}
\end{equation}
$$

とします．ただし，$^\top$を転置記号とし，$\boldsymbol{x}$を列ベクトル，$\boldsymbol{x}^\top$を行ベクトルとします．また，$\boldsymbol{\phi}_j\in \mathbb{R}^{N_\text{out}}$です．
ここから少しややこしいのですが，ネットワークの重み$\Omega=[\omega_{ij}]\in \mathbb{R}^{N\times N}$は 

$$
\begin{equation}
\omega_{i j}=G \omega_{i j}^{0}+Q \boldsymbol{\eta}_{i}^\top \boldsymbol{\phi}_j 
\end{equation}
$$

となっています．$\omega_{i j}^{0}$は固定された再帰重みです．$G, Q$ は定数で，$\eta=[\boldsymbol{\eta}_{i}^\top]\in \mathbb{R}^{N\times N_\text{out}}$ は$-1$か1に等確率に決められた行列です．よって学習するパラメータは$\phi$のみです．よってバイアスを抜いた入力電流$s_{i}$は次のように分割できます．

$$
\begin{align}
s_{i}&=\sum_{j=1}^{N} \omega_{i j} r_{j}\\
&=\sum_{j=1}^{N} \left(G \omega_{i j}^{0}+Q \boldsymbol{\eta}_{i}^\top \boldsymbol{\phi}_j \right)r_{j}\\
&=Q\boldsymbol{\eta}_{i}^\top \hat{\boldsymbol{x}}(t)+\sum_{j=1}^{N} G \omega_{i j}^{0}r_{j}
\end{align}
$$

### 固定重みの初期化
固定された結合重み $\omega_{i j}^{0}$ は $\mathcal{N}(0, (Np)^{-1})$ の正規分布からランダムサンプリングした値である ($N$はニューロンの数，$p$は定数)．ただし，各ニューロンが投射される重みの平均が0になるようにスケーリングする．

### FORCE法の実装


### 補遺：RLS法の導出
ここからはRLS法の導出を行う (cf. Haykin, 2002)．
本項はシミュレーションする上ではスキップしても問題ない．

RLS法では次の損失関数$C\in \mathbb{R}^{N_\text{out}}$を最小化するような重み$\phi=[\boldsymbol{\phi}_j]\in \mathbb{R}^{N\times N_\text{out}}$を求めます．シミュレーション時間を$T$とすると，$C$は

$$
\begin{equation}
C=\int_{0}^T(\hat{\boldsymbol{x}}(t)-\boldsymbol{x}(t))^{2} \mathrm{d} t+\lambda \phi^\top \phi
\end{equation}
$$

です．ただし，$\hat{\boldsymbol{x}}(t), \boldsymbol{x}(t) \in \mathbb{R}^{N_\text{out}}$です．

さて，式の$C$を最小化するような$\phi$を数値的に求めるためには，損失関数の近似が必要です．まず，
時間幅$\Delta t$で$C$を離散化します．さらに$n$ステップ目における重み$\phi(n)$により，$\hat{\boldsymbol{x}}(i)\simeq \phi(n)^\top \boldsymbol{r}(i)$と近似します．このとき，$n$ステップ目の損失関数$C(n)$は

$$
\begin{align}
C(n)&\simeq \sum_{i=0}^{n}(\hat{\boldsymbol{x}}(i)-\boldsymbol{x}(i))^{2}+\lambda \phi(n)^\top \phi(n)\\     
&\simeq \sum_{i=0}^{n}(\phi(n)^\top \boldsymbol{r}(i)-\boldsymbol{x}(i))^{2}+\lambda \phi(n)^\top \phi(n)
\end{align}
$$

となります．ここでL2正則化(ridge)付きの(通常の)最小二乗法の正規方程式により，$C(n)$を最小化する$\phi(n)$は

$$
\begin{align}
\phi(n) &= \left[\sum_{i=0}^{n}(\boldsymbol{r}(i)\boldsymbol{r}(i)^\top+\lambda I_N)\right]^{-1}\left[\sum_{i=0}^{n}\boldsymbol{r}(i)\boldsymbol{x}(i)^\top\right]\\
&=P(n)\psi(n)
\end{align}
$$

となります\footnote{重み$\phi$で$C$を微分し，勾配が0となるときの方程式の解です．}．ただし，

$$
\begin{align}
P(n)^{-1}&= \sum_{i=0}^{n}(\boldsymbol{r}(i)\boldsymbol{r}(i)^\top+\lambda I_N)\ \left(=\int_{0}^T \boldsymbol{r}(t) \boldsymbol{r}(t)^\top \mathrm{d} t+\lambda I_{N}\right)\\
\psi(n)&=\sum_{i=0}^{n}\boldsymbol{r}(i)\boldsymbol{x}(i)^\top
\end{align}
$$

です．$P(n)$は$\boldsymbol{r}(n)$の相関行列の時間積分と係数倍した単位行列の和の逆行列となっています．また，

$$
\begin{equation}
P(n)^{-1}=P(n-1)^{-1}+\boldsymbol{r}(n) \boldsymbol{r}(n)^\top
\end{equation}
$$

となります．ここで，**逆行列の補助定理** (Matrix Inversion Lemma, またはSherman-Morrison-Woodbury Identity) より，

$$
\begin{align}
X&=A+BCD\\
\Rightarrow X^{-1}&=A^{-1} - A^{-1}B(C^{-1}+DA^{-1}B)^{-1}DA^{-1}
\end{align}
$$

となるので，$X={P}(n)^{-1}, A=P(n-1)^{-1}, B= \boldsymbol{r}(n), C=I_{N}, D=\boldsymbol{r}(n)^\top$とすると，

$$
\begin{align}
P(n)&=P(n-1)-\frac{P(n-1) \boldsymbol{r}(n) \boldsymbol{r}(n)^\top P(n-1)}{1+\boldsymbol{r}(n)^\top P(n-1) \boldsymbol{r}(n)} 
\end{align}
$$

が成り立ちます(右辺2項目の分母はスカラーとなります)．
さらに

$$
\begin{align}
\psi(n)&=\psi(n-1)+\boldsymbol{r}(n)\boldsymbol{x}(n)^\top\\
&=P(n-1)^{-1}\phi(n-1)+\boldsymbol{r}(n)\boldsymbol{x}(n)^\top\\
&=\left\{P(n)^{-1}-\boldsymbol{r}(n) \boldsymbol{r}(n)^\top\right\}\phi(n-1)+\boldsymbol{r}(n)\boldsymbol{x}(n)^\top
\end{align}
$$

となります．式(6.22)から式(6.23)へは

$$
\begin{equation}
\phi(n)=P(n)\psi(n) \Rightarrow \psi(n)=P(n)^{-1}\phi(n)
\end{equation}
$$

であること，式(6.23)から式(6.24)へは式(6.18)により，

$$
\begin{equation}
P(n-1)^{-1}=P(n)^{-1}-\boldsymbol{r}(n) \boldsymbol{r}(n)^\top
\end{equation}
$$

であることを用いています．よって，

$$
\begin{align}
\phi(n)&=P(n)\psi(n)\notag\\
&=P(n)\left[\left\{P(n)^{-1}-\boldsymbol{r}(n) \boldsymbol{r}(n)^\top\right\}\phi(n-1)+\boldsymbol{r}(n)\boldsymbol{x}(n)^\top\right]\notag\\
&=\phi(n-1)-P(n)\boldsymbol{r}(n)\boldsymbol{r}(n)^\top\phi(n-1)+P(n)\boldsymbol{r}(n)\boldsymbol{x}(n)^\top\notag\\
&=\phi(n-1)-P(n)\boldsymbol{r}(n)\left[\boldsymbol{r}(n)^\top\phi(n-1)-\boldsymbol{x}(n)^\top\right]\notag\\
&=\phi(n-1)-P(n)\boldsymbol{r}(n)\boldsymbol{e}(n)^\top
\end{align}
$$

となります．式(6.22)と式(6.27)を連続時間での表記法にすると，式(6. 9,10)の更新式となります．

### 小鳥の運動前野との関係 (削除方針)
(Nicola \& Clopath, 2017)では教師信号として正弦波以外にもVan der Pol方程式やLorenz方程式の軌道を用いて実験しています．さらに教師信号としてベートーヴェンの歓喜の歌(Ode to joy)や鳥の鳴き声を用いても学習可能であったようです．

話は少しずれますが，小鳥の運動前野である**HVC** には連鎖的に結合したニューロン群が存在します．これはリズムを生み出すための計時に関わっているといわれています．カナリアのHVCニューロンを実験的に損傷(ablation)させると歌が歌えなくなるという実験がありますが，同様にSNNのHVCパターンをablationすると学習した歌が再生できなくなったようです．このような計時に関わるパターンを**HDTS** (high-dimentional temporal signal)とNicolaらは呼んでいます．HDTSを学習させた後に歓喜の歌を学習させると，HDTSがない場合よりも短い時間かつ高精度で学習できたようです．
さらにHDTSを外部入力とし，同時に映像を学習させる，という実験もしています(HDTSを内的に学習させる場合も行っています)．ネットワークは記録した映像を実時間で再生することができましたが，外部信号のHDTSを加速させることで圧縮再生が可能だったそうです．さらにHDTSを逆にすると，逆再生もできたそうです．
ニューロンの発火のタスク依存的な圧縮は実験的に観察されています(例えばEuston, et al., 2007)．空間的な課題(箱の中に入れて探索させるなど)をラットにさせると，課題中に記憶された場所細胞の順序だった活動は，ラットの睡眠中に圧縮再生されるという実験結果があります．その圧縮比は5.4〜8.1だったそうですが，この比率はSNNが映像を大きな損失なく再生できる圧縮比とほぼ同じであったようです．Nicolaらはさらに進んでSNNを用いて海馬における急速圧縮学習の機構における介在細胞の働きについての研究も行っています(Nicola \& Clopath, 2019)．