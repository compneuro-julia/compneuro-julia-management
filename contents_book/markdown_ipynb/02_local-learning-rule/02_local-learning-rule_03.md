## Hebb則とシナプス可塑性
### Hebb則
神経回路はどのようにして自己組織化するのだろうか．1940年代にHebbにより提案された学習則は「細胞Aが反復的または持続的に細胞Bの発火に関与すると，細胞Aが細胞Bを発火させる効率が向上するような成長過程または代謝変化が一方または両方の細胞に起こる」というものであった \citep{Hebb1949-iv}．すなわち，発火に時間的相関のある細胞間のシナプス結合を強化するという学習則である．これを**Hebbの学習則** (Hebbian learning rule) あるいは**Hebb則** (Hebb's rule) という．Hebb則は（Hebb自身ではなく）Shatzにより"cells that fire together wire together"（共に活動する細胞は共に結合する）と韻を踏みながら短く言い換えられている \citep{Shatz1992-he}．

数式を用いてHebb則を表現してみよう。まず，発火率モデルを定義する．$n$個のシナプス前細胞と$m$個のシナプス後細胞の発火率をそれぞれ $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$ とし，シナプス前細胞と後細胞のあいだのシナプス結合強度を $\mathbf{W} \in \mathbb{R}^{m \times n}$ とすると，前細胞と後細胞の活動の関係は活性化関数 $f(\cdot)$ を用いて $\mathbf{y} = f(\mathbf{W}\mathbf{x})$ と表現できる。このとき，連続時間の形式において，一般化されたHebb則は次のように表される：

$$
\begin{equation}
\tau \frac{\mathrm{d}\mathbf{W}}{\mathrm{d}t} = \phi(\mathbf{y}) \varphi(\mathbf{x})^\top
\end{equation}
$$

ここで，$\tau$ は学習の時定数であり，その逆数 $\eta := 1/\tau$ は**学習率**（learning rate）と呼ばれ，学習の速さを決定するパラメータである。関数 $\varphi(\cdot)$ および $\phi(\cdot)$ は，それぞれシナプス前細胞および後細胞の活動に応じてシナプス重みの変化を決定する変換関数である。特に $\varphi(\cdot)$，$\phi(\cdot)$ を恒等写像（恒等関数）とした場合，Hebb則は次のように簡潔な形で書ける：

$$
\begin{equation}
\tau \dfrac{\mathrm{d}\mathbf{W}}{\mathrm{d}t} = \mathbf{y} \mathbf{x}^\top\quad \left(= (\textrm{後細胞の活動}) \cdot (\textrm{前細胞の活動})^\top\right)
\end{equation}
$$

このような単純な形式のHebb則を**線形Hebb則**と呼び，狭義にはこれをもってHebb則とすることが多い。一方で，$\varphi(\cdot)$，$\phi(\cdot)$ を非線形関数とした拡張形式は**非線形Hebb則**と呼ばれ，本章で取り扱う。

### シナプス可塑性とLTP・LTD
LTPの実験的発見  

このHebb則の神経生理学的な基盤を裏付けるものとして，1973年にBlissとLømoによってウサギの海馬において**長期増強**（Long-Term Potentiation, LTP）が発見された \citep{Bliss1973-vj}．彼らの実験では，海馬のSchaffer側枝からCA1錐体細胞への経路に高頻度の電気刺激を加えることで，その後のシナプス応答が長時間にわたって増強される現象が観察された．この持続的なシナプス強度の増加は，まさにHebb則に対応する生理的現象と見なされ，Hebbian plasticityの実体と考えられるようになった．LTPはグルタミン酸作動性シナプスで観察されることが多く，特にNMDA受容体が関与することで知られている．この受容体は膜電位依存的にMg²⁺ブロックが外れることにより，カルシウムイオン（Ca²⁺）の流入を許し，それが下流のシグナル伝達を活性化してシナプス後部のAMPA受容体の増加や活性化を引き起こす．

一方，1980年代には**長期抑圧**（Long-Term Depression, LTD）という現象も発見された \citep{Dudek1992-nz}．これは，シナプス前ニューロンとシナプス後ニューロンが低頻度で同時活動した場合に，シナプスの伝達効率が長期にわたって減少する現象である．LTDもまた海馬や小脳などの領域で観察されており，この減弱はHebb則の反対の効果を示すものとして位置づけられる．特に，小脳における登上線維と平行線維の同時活動により引き起こされるLTDは，運動学習のモデルとして重要視されている．LTPと同様に，LTDにおいてもCa²⁺シグナリングが重要な役割を果たすが，その振幅や時間的プロファイルが異なっていることが，シナプス強化と抑圧の分岐をもたらすと考えられている．

これらの発見を通じて，Hebb則は単なる理論的仮説にとどまらず，シナプス可塑性という具体的な細胞メカニズムを通して，神経回路における学習と記憶の基盤であることが明らかにされた．

https://www.science.org/doi/10.1126/science.ads4706
https://pubmed.ncbi.nlm.nih.gov/24183021/
https://pubmed.ncbi.nlm.nih.gov/15450156/
https://pubmed.ncbi.nlm.nih.gov/26139370/
https://pubmed.ncbi.nlm.nih.gov/15450157/
https://pubmed.ncbi.nlm.nih.gov/18275283/
https://pubmed.ncbi.nlm.nih.gov/17332410/

NMDA, Ca
BDNF 
TrkB

TNF-\alpha

シナプス強度が増加するとは何が変化しているのか，とはspine sizeや受容体の数，

https://www.sciencedirect.com/science/article/pii/S0959438823001034?casa_token=ONmKH_RolAYAAAAA:DpV67Cj98lAp6pZk0_f-hTKCRbQ1wq20NPX_Fm1X0IF-eN6NhICDEAbgDIWdIPe3cLxCsadBJsRn

https://www.jneurosci.org/content/40/14/2828.abstract

https://www.sciencedirect.com/science/article/pii/S0149763419310942

### Hebb則の不安定性と修正Hebb則
Hebb則には問題点があり，シナプス結合強度が際限なく増大するか，あるいは消失するかという不安定性がある．これを数式で確認しておこう．前細胞と後細胞がそれぞれ1つの場合を考える．2細胞間の結合強度を $w\ (>0)$ とし，線形ニューロンを仮定，すなわち $y=wx$ が成り立つとすると，Hebb則は $\dfrac{dw}{\mathrm{d}t}=\eta yx=\eta x^2w$ となる．この場合，$\eta x^2>1$ なら $\lim_{t\to\infty} w= \infty$, $\eta x^2<1$ なら $\lim_{t\to\infty} w= 0$ となる．当然，生理的にシナプス結合強度が無限大となることはあり得ないが，不安定なほど大きくなってしまう可能性があることに違いはない．このため，Hebb則を安定化させるための修正が必要とされた．

この問題に対して、さまざまな修正Hebb則 (modified hebbian rule) が提案されている．ここでは代表的な学習則である **CLO則**、**Oja則**、そして**BCM則**について説明する．

以下では 入力 $\mathbf{x}\in \mathbb{R}^n$ $y\in \mathbb{R}$ とし，シナプス結合 $\mathbf{w}\in \mathbb{R}^n$ を持つ，単一の神経細胞モデル $y = f(\mathbf{w}^\top \mathbf{x})$ を仮定する．

https://bsd.neuroinf.jp/wiki/Bienenstock-Cooper-Munro%E7%90%86%E8%AB%96#
https://bsd.neuroinf.jp/wiki/%E9%95%B7%E6%9C%9F%E5%A2%97%E5%BC%B7
https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/5-Hebbian.html
https://lcnwww.epfl.ch/gerstner/SPNM/node72.html

#### CLO則
視覚野のニューロンが経験により方向選択性を獲得し、かつ視覚刺激の遮断によってその選択性を失うといった生理的実験の結果を説明する理論として，Cooper, Liberman, Ojaにより閾値制約付き受動的可塑性 (threshold passive modification) と呼ばれる形式の学習則が提案された \citep{Cooper1979-wz}．この学習則は提案者の頭文字を取って，**CLO則** (CLO rule) と呼ばれる．CLO則は、Hebb則に対して出力の大きさに応じた閾値的な修正と重みの減衰項（忘却）を加えることにより、選択性の獲得と重みの安定性の両立を目指した学習則である。

CLO則は、出力 $y$ の値と**修正閾値**（modification threshold） $\theta_m$ および出力飽和値 $\theta_\mathrm{max}$ によって区分される3つの範囲で異なる更新を行う。CLO則の元の形式は離散時間での更新則であるが，表記を統一するため，連続時間でのCLO則を示す：

$$
\begin{align}
\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t} =
\begin{cases}
- \lambda \mathbf{w} & (y \geq \theta_{\max}) \\
- \lambda \mathbf{w} + \eta_+ (\theta_{\max} - y) \mathbf{x} & (\theta_m \leq y < \theta_{\max}) \\
- \lambda \mathbf{w} - \eta_- y \mathbf{x} & (y < \theta_m)
\end{cases}
\end{align}
$$

ここで，$\lambda\ (\geq 0)$ は重みの減衰（leak）の度合いを決める定数であり，$\eta_+, \eta_-\ (> 0)$ はそれぞれ増強・抑圧に対応する学習率である．なお，第2項はHebb則であるが，第3項は**反Hebb則** (anti-Hebbian rule) と呼ばれる．

このように、適度な出力のときにのみ強化が起こり、過剰な出力では学習が停止し、出力が小さすぎる場合には抑制が起こるという、三相性の学習則が構築される。これにより、各ニューロンは特定の入力パターンに対してのみ強い応答を示すようになり、他のパターンには反応しなくなる。これは方向選択性や空間選択性のような感覚特異性（specificity）の獲得を数理的に説明する。CLO則はLTPに加えてLTDも組み込み，重みの減衰項も加えているため，不安定性はHebb則よりも低減されている．一方で，複雑で不連続な三相性の学習則を持ち，修正閾値 $\theta_m$ も固定値であるという欠点があった．$\theta_m$ が固定されていると，$\theta_m$ が大きければLTDしか生じず，小さければLTPのみが生じる．

#### BCM則
CLO則を踏まえて，Bienenstock, Cooper, Munroにより提案された**BCM則**（Bienenstock–Cooper–Munro則）ではLTPとLTDを連続的に記述し，修正閾値 $\theta_m$ は出力活動の履歴に応じて変化するように修正された \citep{Bienenstock1982-km} \citep{Cooper2012-ec}．BCM則は次のように表される：

$$
\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t} = \eta_w \, \mathbf{x} \, y (y - \theta_m)
$$

関数$\phi$は$\phi(y, \theta_m)=y(y-\theta_m)$などとする．非線形Hebb則の一種である．また $\theta_m:=\mathbb{E}[y^2]$は閾値を決定するパラメータ，**修正閾値** (modification threshold) である．$\theta_m$ は活動履歴に基づいて動的に変化し、たとえば以下のように定義される：

$$
\frac{\mathrm{d}\theta_m}{\mathrm{d}t} = \eta_\theta (y^2 - \theta_m)
$$

この構造により、出力 $y$ が $\theta_m$ を超えるときにはシナプスが強化され（LTP）、逆に $y < \theta_m$ のときには弱化（LTD）される。このように、BCM則は同一の数式の中でHebbian強化とAnti-Hebbian抑制を両立させている。また、この動的閾値 $\theta_m$ は、ニューロンが自らの「活動水準の平均」を内部的に学習していく仕組みであり、これにより入力空間に対する**選択的な応答性**が獲得される。これは、視覚野ニューロンの方位選択性など、実際の神経生理学的観測とも整合する

#### Oja則
Hebb則を安定化させる別のアプローチとして，結合強度を正規化するという手法が考えられる．学習率を $\eta$ とすると，$\mathbf{w}\leftarrow\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とすれば正規化できる．ここで，$h(\eta):=\dfrac{\mathbf{w}+\eta \mathbf{x}y}{\|\mathbf{w}+\eta \mathbf{x}y\|}$とし，$\eta=0$においてTaylor展開を行うと，

$$
\begin{align}
h(\eta)&\approx h(0) + \eta \left.\frac{dh(\eta^*)}{\mathrm{d}\eta^*}\right|_{\eta^*=0} + \mathcal{O}(\eta^2)\\
&=\frac{\mathbf{w}}{\|\mathbf{w}\|} + \eta \left(\frac{\mathbf{x}y}{\|\mathbf{w}\|}-\frac{y^2\mathbf{w}}{\|\mathbf{w}\|^3}\right)+ \mathcal{O}(\eta^2)
\end{align}
$$

ここで $\|\mathbf{w}\|=1$ として，1次近似すれば $h(\eta)\approx \mathbf{w} + \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)$ となる．重みの変化が連続的であるとすると，

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t} = \eta \left(\mathbf{x}y-y^2 \mathbf{w}\right)
\end{equation}
$$

として重みの更新則が得られる．これを**Oja則** (Oja's rule) と呼ぶ \citep{Oja1982-yd}．こうして得られた学習則において$\|\mathbf{w}\|\to 1$となることを確認しよう．

$$
\begin{equation}
\frac{\mathrm{d}\|\mathbf{w}\|^2}{\mathrm{d}t}=2\mathbf{w}^\top\frac{\mathrm{d}\mathbf{w}}{\mathrm{d}t}= 2\eta y^2\left(1-\|\mathbf{w}\|^2\right)
\end{equation}
$$

より，平衡状態 $\frac{\mathrm{d}\|\mathbf{w}\|^2}{\mathrm{d}t}=0$ において，$\|\mathbf{w}\|= 1$となる．

### 非線形Hebb学習
出力$\mathbf{y}$に非線形関数$g(\cdot)$を適用し，$\mathbf{y}\to g(\mathbf{y})$として置き換えることで非線形Hebb学習となる\citep{Oja1997-hr}\citep{Brito2016-mx}. 

ToDo: 詳細

### 恒常的可塑性
Oja則は更新時の即時的な正規化から導出されたものであるが，より時間スケールの長い過程として恒常的可塑性 (synaptic scaling)により安定化しているという説がある \citep{Turrigiano2008-lm} \citep{Yee2017-fb}．しかし，この過程は遅すぎるため，Hebb則の不安定化を安定化するに至らない \citep{Zenke2017-el}

https://www.nature.com/articles/s41598-023-32410-0
https://www.cell.com/neuron/fulltext/S0896-6273(20)30188-4?uuid=uuid%3Afdd605ec-6489-4c89-b4c9-73e1eb8f1c1a

https://www.pnas.org/doi/full/10.1073/pnas.1421304111

https://www.sciencedirect.com/science/article/abs/pii/S095943880000091X?via%3Dihub
https://www.cell.com/trends/neurosciences/abstract/S0166-2236(98)01341-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0166223698013411%3Fshowall%3\mathrm{d}true

https://www.nature.com/articles/nrn1949

Synaptic scaling

Oja則と同様に，
実装上はノルムで割る（乗法的スケーリング）

### Hebb則の変分原理的導出
Hebb則は数学的に導出されたものではないが，神経回路のダイナミクスがある目的関数（エネルギー関数）を最適化するように設計されていると仮定すれば、Hebb則に対応する重み更新則が自然に導出される。このようなネットワークを**エネルギーベースモデル** (energy-based models) といい，次章で扱う．

エネルギーベースモデルでは、神経活動 $\mathbf{z}$ およびシナプス結合 $\mathbf{W}$ に対して、あるエネルギー関数 $E(\mathbf{z}, \mathbf{W})$ を定義し、ダイナミクスがそのエネルギーを減少させるように構成される。すなわち、神経状態と重みの時間変化は、それぞれ次のようにエネルギーの勾配に基づいて与えられる：

$$
\begin{equation}
\frac{\mathrm{d}\mathbf{z}}{\mathrm{d}t}\propto-\left(\frac{\partial E}{\partial \mathbf{z}}\right)^\top,\quad \frac{\mathrm{d} \mathbf{W}}{\mathrm{d}t}\propto-\left(\frac{\partial E}{\partial \mathbf{W}}\right)^\top
\end{equation}
$$

このとき、逆に神経活動のダイナミクスのみが先に与えられている場合でも、それに整合するエネルギー関数を定義すれば、重みの更新則（すなわち学習則）を変分原理的に導出することができる。具体的には，神経細胞の活動ダイナミクスを積分することで神経回路のエネルギー関数 $E$ を導出し，さらに $E$ を重み行列で微分することでHebb則が導出できる \citep{Isomura2020-sn}．Hebb則の導出を連続時間線形ニューロンモデル $\dfrac{\mathrm{d}\mathbf{y}}{\mathrm{d}t}=-\mathbf{y}+\mathbf{W}\mathbf{x}$ を例にして考えよう（簡単のため $\tau=1$ とした）．ここで $\dfrac{\partial E}{\partial\mathbf{y}}:=-\left(\dfrac{\mathrm{d}\mathbf{y}}{\mathrm{d}t}\right)^\top$ となるようなエネルギー関数 $E(\mathbf{x}, \mathbf{y}, \mathbf{W})$ を仮定すると，

$$
\begin{equation}
E(\mathbf{x}, \mathbf{y}, \mathbf{W})=-\left(\int -\mathbf{y}+\mathbf{W}\mathbf{x}\,\mathrm{d}\mathbf{y}\right)\propto\|\mathbf{y}\|^2-\mathbf{y}^\top \mathbf{W}\mathbf{x} \in \mathbb{R}
\end{equation}
$$

となる．これをさらに$\mathbf{W}$で微分すると，

$$
\begin{equation}
\dfrac{\partial E}{\partial\mathbf{W}}=-\mathbf{x}\mathbf{y}^\top\Rightarrow
\frac{\mathrm{d}\mathbf{W}}{\mathrm{d}t}=-\left(\frac{\partial E}{\partial \mathbf{W}}\right)^\top=\mathbf{y}\mathbf{x}^\top
\end{equation}
$$

となり，Hebb則が導出できる．

このような導出の仕方は、物理学の解析力学における**変分原理**（variational principle）あるいは **最小作用の原理**（principle of least action）の発想に基づいている．これらの原理においては、ある経路に沿って定義される**作用**（action）と呼ばれる量が極値（多くの場合、極小値）を取るような経路が、実際に物理系が取る経路として実現されるとされる。この考え方と同様に、神経活動やシナプス結合の変化も、あるエネルギー関数（または汎関数）の極値（神経回路が安定している場合には極小値）を実現するようなダイナミクスとして捉えることができる。もちろん、神経活動とシナプス結合が同一のエネルギー関数を最小化しているというのは大きな仮定である。しかし、このように神経回路の時間発展を最適化の視点から記述する立場は、神経可塑性を理論的に理解するための一つの有力な枠組みを提供し得る \citep{isomura2023experimental}．