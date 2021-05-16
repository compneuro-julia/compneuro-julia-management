# Current / Conductance-based シナプス
## 化学シナプスの2つの記述形式
具体的なシナプスのモデルの前に, この節では化学シナプスにおけるシナプス入力(synaptic drive)の2つの形式, **Current-based シナプス**と**Conductance-based シナプス**について説明する。簡単に言うと、Current-based シナプスは入力電流が変化するというモデルで, Conductance-based シナプスはイオンチャネルのコンダクタンス (電気抵抗の逆数, 電流の流れやすさ)が変化するというモデルである ([Cavallari et al., 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3943173/))。

以下では例として, 次のLIFニューロンの方程式におけるシナプス入力を考える。

$$
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_m I_{\text{syn}}(t)    
$$

とする。ただし, $\tau_m$は膜電位の時定数, $V_m(t)$は膜電位, $V_\text{rest}$は静止膜電位, $R_m$は膜抵抗である。ここで、シナプス入力の電流$I_{\text{syn}}(t)$が[^syn]2つのモデルにおいて異なる部分となる。

[^syn]: シナプス(synapse)入力であることを明らかにするためにsynと添え字をつけている。

## Current-based シナプス
Current-based シナプスは単純に**入力電流が変化**するというモデルで, 簡略化したい場合によく用いられる。シナプス入力$I_{\text{syn}}(t)$はシナプス効率(synaptic efficacy)[^syneff]を$J_{\text{syn}}$ (単位はpA)とし , シナプスの動態(synaptic kinetics)を$s_{\text{syn}}(t)$とすると, 次式のようになる。ただし, シナプスの動態とは, 前細胞に注目すれば神経伝達物質の放出量, 後細胞に注目すれば神経伝達物質の結合量やイオンチャネルの開口率を表す。

$$
\begin{equation}
I_{\text{syn}}(t)=\underbrace{J_{\text{syn}}s_{\text{syn}}(t)}_{電流の変化}    
\end{equation}
$$

ただし, $s_{\text{syn}}(t)$は, 例えば次節で紹介する$\alpha$関数を用いる場合, 

$$
\begin{equation}
s_{\text{syn}}(t)=\dfrac{t}{\tau_s} \exp \left(1-\dfrac{t}{\tau_s}\right)    
\end{equation}
$$

のようになる。

[^syneff]: シナプス強度(Synaptic strength)とは違い, 受容体の種類(GABA受容体やAMPA受容体,  およびそのサブタイプなど)によって決まる。

## Conductance-based シナプス
Conductance-based シナプスはイオンチャネルの**コンダクタンスが変化**するというモデルである。関連して、例えば Hodgkin-Huxley モデルはConductance-based モデルの1つである。Current-basedよりもConductance-based の方が生理学的に妥当である。例えば抑制性シナプスは膜電位が平衡電位と比べて脱分極側にあるか, 過分極側にあるかで抑制的に働くか興奮的に働くかが逆転する。これはCurrent-based シナプスでは再現できない。

Conductance-based モデルにおけるシナプス入力は$I_{\text{syn}}(t)$は次のようになる。 

$$
\begin{equation}
I_{\text{syn}}(t)=\underbrace{g_{\text{syn}}s_{\text{syn}}(t)}_{コンダクタンスの変化}\cdot\ \left(V_{\text{syn}}-V_{m}(t)\right)    
\end{equation}
$$

ただし, $g_{\text{syn}}$ (単位はnS)はシナプスの最大コンダクタンス[^gsyn], $V_{\text{syn}}$ (単位はmV)はシナプスの平衡電位を表す。これらも$J_{\text{syn}}$と同じく, シナプスにおける受容体の種類によって決まる定数である。

[^gsyn]: $g_{\text{syn}}$がシナプスの最大コンダクタンスとなるのは $s_{\text{syn}}$の最大値を1に正規化する場合である。正規化は必須ではないので, 単なる係数と思うのがよい。

注意しなければならないことは, $s_{\text{syn}}(t)\leq 0$としたとき, Current-based モデルにおける$J_{\text{syn}}$は正の値(興奮性)と負の値(抑制性)を取るが, $g_{\text{syn}}$は正の値のみである、ということである [^gsyn2]. Conductance-basedモデルで興奮性と抑制性を決定しているのは, 平衡電位$V_{\text{syn}}$である。興奮性シナプスの平衡電位は高く, 抑制性シナプスの平衡電位は低いため, 膜電位を引いた符号はそれぞれ正と負になる。

[^gsyn2]: これはコンダクタンスが電気抵抗の逆数であり, 基本的に抵抗は正の値しか取らないことからも分かる。なお電子回路においては負性抵抗という,  素子の抵抗値が見かけ上, 負の値を取る場合もある。
