#### 神経集団モデルと神経場モデル
Wilson–Cowanモデルと密接に関連し，より大域的・集団的な神経活動を記述する枠組みとして，**神経集団モデル**（neural mass model, neural population model）および**神経場モデル**（neural field model）があり，これらのモデルに関して簡単に触れておく．いずれも個々のニューロンの詳細な活動ではなく，神経細胞集団の平均的な膜電位や発火率の時間変化を対象とし，脳波などのマクロな神経活動を記述するための理論的枠組みを提供する．

神経集団モデルでは，皮質のマイクロカラムや局所回路といった小規模な神経集団を1ユニットとしてモデル化し，Wilson–Cowanモデルと同様に，平均発火率や膜電位のダイナミクスを扱う．神経集団モデルの例としては局所神経回路をモデル化したJansen-Ritモデル \citep{jansen1995electroencephalogram, david2003neural} や，てんかん活動の動態を記述するWendlingモデル \citep{wendling2002epileptic} などがある．

一方，神経場モデル（neural field model）では，神経活動を空間的に連続な関数として記述し，広範囲における神経活動の時空間的なダイナミクスを扱う \citep{coombes2014tutorial, cook2022neural}．神経場モデルはWilsonおよびCowan \citep{wilson1973mathematical}, Nunez \citep{nunez1974brain}, 甘利 \citep{amari1975homogeneous, amari1977dynamics} らの研究に基づいており，ここでは甘利による定式化（**甘利モデル**, Amari model）を簡単に説明する．甘利モデルでは，まず神経場の定義域 $\Omega$（一次元の皮質断面や二次元の皮質平面など）を考える．$\Omega$ における神経活動は以下のような積分–微分方程式によって与えられる：

$$
\begin{equation}
\tau \frac{\partial u(x,t)}{\partial t} = -u(x,t) + \int_{\Omega} w(x, x') f(u(x', t))\,\mathrm{d}x' + I(x,t)
\end{equation}
$$

ここで，$x, x' \in \Omega$ は神経場における位置を表す．$u(x,t)$ は位置 $x$ における時刻 $t$ の発火率，$w(x,x')$ は位置 $x'$ から $x$ への結合重み，$f(\cdot)$ は活性化関数，$I(x,t)$ は外部入力を表す．神経場モデルは，皮質進行波（cortical travelling waves）\sitep{muller2018cortical} 等の現象を理論的に説明する手段となる．