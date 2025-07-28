# Writing ToDo list

全体を丸ごと組みなおそう．
エネルギーベースモデル，生成モデル，ベイズ脳仮説は結びついているので，一緒にまとめたほうがいい．backpropの近似も，厳密に対応するわけではないので，「後に説明する」として置いておく．

---

2章の局所学習→4章のニューラルネットワークにする．

2章でHopfieldモデルを登場させてもよいのでは．エネルギーベースモデルの導入になる（詳細は後で，とする）．Hebb則の説明にもつながる．hebb則の数理的導出はHopfieldモデルの後にする．Hopfield → hebb則 → Perceptronにするか．DAMはコラムか注釈で触れる程度でよいと思う．

4章でpredictive codingを出すのはやめる (厳密に同じではないし)．
リザバーコンピューティングの後に3章と9章を合体させた章を作成する．

bolztmannマシンをどうするかが決着ついていないが…．
競合学習は一部省略する．spikingの話はいれてもいいかも．SoftHebbはいれるべきか…．

3章と9章を合体させた章
生成的知覚，生成モデル，ベイズ脳仮説，エネルギーベースモデル，階層的生成モデル，ベイズ線形回帰，神経サンプリング

---
① → ②（生成的知覚 → 生成モデル）
「感覚入力は世界の原因の影響であり、その背後には何があるのかを脳は推定している。これを数理的に表すと、原因→観測の生成過程＝生成モデルとなる。」

② → ③（生成モデル → ベイズ脳仮説）
「脳が観測から原因を推定するなら、生成モデルを逆にたどる必要がある。これはベイズの定理によって定式化される。」

③ → ④（ベイズ脳仮説 → EBM）
「ただし、明示的に確率分布を定義するのが難しい場合でも、エネルギーを定義すれば、そこから確率分布を誘導できる。これがエネルギーベースモデル。」

HopfieldはEBMの最も単純な具体例
Boltzmanマシン (MCMCは後で詳細)，

④ → ⑤（EBM → 階層的生成モデル）
「実際の世界や脳の情報処理は1層では足りない。より表現力を持たせるには階層的に生成する必要がある。」

MAP推定で行う．

スパース符号化，
予測符号化（スパース符号化も階層化できることは書く），

⑤ → ⑥（階層モデル → ベイズ線形回帰）
「では推論はどうやって行われるか？その一例として、最も基本的な線形生成モデルにおけるベイズ推論を見てみよう。」

⑥ → ⑦（線形回帰 → 神経サンプリング）
「このような推論は数理的には明快だが、脳がこれをどう実装しているかは別の問題。1つの有力な仮説が、神経発火が確率的サンプリングを実現しているという神経サンプリング仮説である。」

Boltzmanマシンでも使用した～などとする．

---

形態の話は，multi compartmentモデルの後にするか．計算論的発生学モデルはコラムで入れる．
Exponentiated gradientsなどは
対数正規分布の話もlocal learningに加える？

---

https://arxiv.org/html/2310.00965v5
を参考に図を作成．


https://simons.berkeley.edu/sites/default/files/docs/9574/backpropagationanddeeplearninginthebrain-timothylillicrap.pdf

Dale's law backpropagation

exponentiated gradients

$$
w_{t+1} = w_t \odot \exp(-\eta \nabla L(w_t) \odot \mathrm{sign}(w_t))
$$

普通のGDは


$$
w_{t+1} = w_t -\eta \nabla L(w_t)
$$

https://www.biorxiv.org/content/10.1101/2024.10.25.620272v1

掛け算での重み更新について．
https://arxiv.org/abs/2506.17768
https://arxiv.org/abs/2106.13914
https://arxiv.org/abs/2006.14560

脳となぜ似るのか？
表現の類似性と収斂進化

Urbanzick & Senn, Neuron 2014は入れるべきか．


基本的な話は発火率モデルですべてすむので，
発火率モデル→spikingモデルとしたほうが全体の流れとしてはよいか？
introも省略していいと思う．時間がない．

neuronの形態の話は別で節を作ろう．

ニューロンとシナプスの生理
発火率モデル
local-learning-rule
energy based
solve-credit-assignment-problem

やはりこれを前提としたモデルの構築が必要なので，発火率ですむ部分は前に持っていく．

第n章 Spiking neural network
ぐらいにしておいてもいいかも．
第1節. ニューロンのモデル
モデルを列挙…

neuron-model
synapse-model
neuronal-computation

リザバーコンピューティング (発火率・spiking)
ベイズ推論
運動学習
強化学習
ネットワーク・形態学