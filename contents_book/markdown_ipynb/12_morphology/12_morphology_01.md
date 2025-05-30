# 第12章：神経系の発生と発達の数理モデル
## 神経突起の成長モデル
神経細胞は他の細胞に比して特異な形態を持つ．またニューロンの種類およびその役割により基本的な細胞体，樹状突起，軸索等の構造は共通するものの，各部分の形態は異なる．このような形態はどのようにして発達するのだろうか．本節では**神経突起(neurite)** の**成長モデル(growth model)** を取り扱う．神経突起とは神経細胞において細胞体から伸びる突起の総称である．

## 神経突起の木構造
神経突起の形態は**樹状**突起 (dendrites; ギリシャ語で木を意味する*déndron*に由来) に代表されるように（生物としての）木に類似している．さらに分節(segment)に離散化することでグラフ理論における**木**(tree; 連結で閉路を持たないグラフ)として捉えることができる．

シミュレーション用にデータ構造を作成しよう．なお，Juliaで木構造を扱うためのライブラリ`AbstractTrees.jl`は使用しない．`tree_info`はInt型vector（要素数3）のlistであり，接続している分節の番号，遠心性位数，分節の種類(1: 末端, 0:中間)を表す．`seg_vec`は Float型vector（要素数2）のlistであり，分節の2次元極座標ベクトル(半径，角度)を表す．3次元に拡張することも可能であるが，本書では簡単のために2次元とする．多次元配列ではなくvectorのlistにしているのは，成長に伴って要素を追加していく際に配列に結合`cat`するよりlist化して追加`push!`する方が高速なためである．

## Van Peltモデル
Van PeltモデルはVan Peltらによって構築された，神経突起の成長についての現象論的モデルである {cite:p}`Van_Pelt2002-vm`．以下では{cite:p}`Koene2009-hv`に基づいて記述する．なお，このモデルでは軸索誘導分子 (axon guidance molecules) 等の存在は無視している．

神経突起の成長の過程には分岐(branching)，伸長(elongation)，転向(turn)が含まれる．簡略化のため，空間を2次元にし，分節の太さおよび成長円錐が向きを変える時のsegment history tension model（後述）を省略する．またVan Peltモデルを元にした神経回路構築ソフトウェア**NETMORPH** {cite:p}`Koene2009-hv`ではシナプス結合の形成も含めたシミュレーションを行っている．

### 分岐 (branching)
時刻$[t_i, t_i + \Delta t]$において，$j$番目の末端分節(terminal segment)が分岐する確率は

$$
\begin{equation}
p_{i,j} = n_i^{-E}\cdot B_{\infty} e^{\frac{-t_i}{\tau}} \left(e^{\frac{\Delta t}{\tau}} - 1\right)\cdot \frac{2^{-S\gamma_j}}{C_{n_i}}
\end{equation}
$$

で表される．ここで，$B_{\infty}, E, S, \tau$は定数である．$\gamma_j$は$j$番目の末端分節の遠心性位数(centrifugal order)であり，$n_i$は時刻$t_i$における末端分節の総計である．さらに

$$
\begin{equation}
{C_{n_i}} = \frac{1}{n_i}\sum\nolimits_{k = 1}^{n_i} {{2^{ - S{\gamma_k}}}}
\end{equation}
$$

とする．$n_i^{-E}$は末端分節の総計に応じて分岐確率を変化させる項であり，$E$は競合変数(competition parameter)と呼ばれる．
$B_{\infty} e^{\frac{-t_i}{\tau}} \left(e^{\frac{\Delta t}{\tau}} - 1\right)$は経過時間に応じて分岐確率を変化させる項であり，$B_{\infty}$は$E=0$の場合の末端分節での分岐数の漸近的な期待値である．
$\frac{2^{-S\gamma_j}}{C_{n_i}}$の項は末端分節の遠心性位数に応じて分岐確率を変化させる項であり，$C_{n_i}$は正規化定数である．
$S=0$のときは末端分節は全て同じ確率で分岐するが，$S>0$のときは近位の末端分節，$S<0$のときは遠位の末端分節における分岐確率が大きくなる．

### 伸長 (elongation) 
末端分節が伸長する速さ$\nu_e(t_i)\ [\mu m/s]$は正規分布 $\mathcal{N}(\mu_e, \sigma_e^2)$に従うとする {cite:p}`Van_Ooyen2014-fb`．伸長する長さは$\Delta L_j(t_i)=\nu_e(t_i) \cdot \Delta t$となる．

### 転向 (turn)
神経突起は真っ直ぐに伸び続けるわけではなく，向きを時折変えながら伸長する．伸長時に転向するかどうかの確率$p_d(t_i)$を次のようにする．

$$
\begin{equation}
p_d(t_i) = r_L\Delta L_j(t_i)
\end{equation}
$$

ただし，$r_L\ [\mu m^{-1}]$は回転率を表す．確率$p_d(t_i)$により転向する部分は新しい分節として定義する．転向する角度は{cite:p}`Koene2009-hv`では転向角度の履歴を考慮したsegment history tension modelが導入されているが，本書では前述のように省略する．代わりに転向角度は一様分布$U(-\alpha, \alpha)\ \left(\alpha\in \left[0, \frac{\pi}{2}\right]\right)$に従うとする．

分岐した際にも娘枝の長さと角度の設定が必要となる．ここでは長さは末端分節の伸長と同じ正規分布に従うとする．また，分岐角度は2つの娘枝について一様分布$U(0, \beta_1),\ U(-\beta_2, 0)\ \left(\beta_1, \beta_2\in \left[0, \frac{\pi}{2}\right]\right)$にそれぞれ従うとする．

以上をまとめてシミュレーションを実装する．

対称性の破れを考慮していないので，円系に成長している．
ToDo: 神経細胞極性についての記述．

https://www.nature.com/articles/nrn2056