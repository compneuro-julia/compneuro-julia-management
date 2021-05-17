# シナプス入力の重みづけ
ここまでは, シナプス前細胞と後細胞がそれぞれ1つずつである場合について考えていたが, 実際には多数の細胞がネットワークを作っている。また, それぞれの入力は均等ではなく, 異なるシナプス強度 (Synaptic strength)を持つ。この場合のシナプス入力の計算について述べておく。

シナプス前細胞が$N_{\text{pre}}$個, シナプス後細胞が$N_{\text{post}}$個あるとする。このとき**シナプス前過程に注目した**シナプス動態を$\boldsymbol{s_{\text{syn}}}\in \mathbb{R}^{N_{\text{pre}}}$, シナプス後細胞の入力電流を$\boldsymbol{I_{\text{syn}}}\in \mathbb{R}^{N_{\text{post}}}$, シナプス結合強度の行列を$W\in \mathbb{R}^{N_{\text{post}} \times N_{\text{pre}}}$とすると, Current-basedの場合は

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=W \boldsymbol{s_{\text{syn}}}  
\end{equation}
$$

となる。ただし, シナプス強度にシナプス効率が含まれるとした. また, Conductance-basedの場合はシナプス後細胞の膜電位を$\boldsymbol{V}_{m}\in \mathbb{R}^{N_{\text{post}}}$として, 

$$
\begin{equation}
\boldsymbol{I_{\text{syn}}}(t)=\left(V_{\text{syn}}-\boldsymbol{V}_{m}(t)\right)\odot W \boldsymbol{s_{\text{syn}}}
\end{equation}
$$

となる。ただし, $\odot$はHadamard積である。

これらの式は順序を入れ替えることも可能である。シナプス前細胞でスパイクが生じたことを表すベクトルを$\boldsymbol{\delta}_{t,t_{\text{spike}}}\in \mathbb{R}^{N_{\text{pre}}}$とする。ただし, $t_{\text{spike}}$は各ニューロンにおいてスパイクが生じた時刻である。 $\boldsymbol{s_{\text{syn}}}$は$\boldsymbol{\delta}_{t,t_{\text{spike}}}$の関数であり, $\boldsymbol{s_{\text{syn}}}(\boldsymbol{\delta}_{t,t_{\text{spike}}})$と表せる。このとき**シナプス後過程に注目した**シナプス動態を$\boldsymbol{s}^\prime_{\text{syn}}\in \mathbb{R}^{N_{\text{post}}}$とすると, Current-basedの場合は

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

と表すことができる。

シナプス動態を前過程か後過程のどちらに注目したものとするかは, 実装によって様々である。シナプス入力の計算における中間の値を学習に用いるということもあるため, 単なる計算量の観点だけではどちらを選ぶかは決めることができない (計算量だけならシナプス変数に先に重み行列をかけた方がよい場合が多い)。実装の中で異なってくるのは計算順序と保持するベクトルの要素数である。 同じ実装の中で2つとも用いる場合もあるので注意してほしい。
