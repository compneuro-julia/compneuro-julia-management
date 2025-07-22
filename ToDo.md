# Writing ToDo list

https://arxiv.org/html/2310.00965v5
を参考に図を作成．

two point methodは結局勾配の定義に基づいているだけ？

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