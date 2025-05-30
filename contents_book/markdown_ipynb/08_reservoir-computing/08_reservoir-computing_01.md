# 第8章：リザバーコンピューティング
本章では，**リザバーコンピューティング**（reservoir computing; RC）と呼ばれる枠組みに基づく再帰型ニューラルネットワーク（recurrent neural network; RNN）およびその学習手法について解説する．リザバーコンピューティングは，主にリザバー（reservoir）と呼ばれるRNNと，読み出し器（readout）と呼ばれる線形ネットワークから構成される．読み出し器は，リザバーRNNの活動に基づいて出力を生成する役割がある．

「リザバー（reservoir）」とは，本来は「貯水池」や「溜め池」を意味する語であり，リザバーコンピューティングにおける比喩的な表現として用いられている．リザバーコンピューティングでは，まず入力信号をランダムに初期化された固定重みによって高次元空間へとマッピングし，その信号をリザバー内に保持する．この保持のイメージは，液体のように信号を溜めるというよりも，池に石を投げ込んだ際に生じる波紋がしばらく残存するように，入力に対する時間的応答がリザバー内に残るという考え方に近い．リザバーに保持された動的な信号は，RNNの各ユニットの活動として表現され，これを読み出し器によって線形変換することで最終的な出力が得られる．出力重みは，教師信号とネットワーク出力との誤差を基に学習される．

このように，一般的なRNNがネットワーク内のすべての結合重みを学習するのに対し，リザバーコンピューティングではRNN部分の結合重みはランダムに初期化されて以降は固定され，学習は読み出し器の出力重みに限定される．一般のRNNと比較すると，リザバーコンピューティングの表現力には制約があるものの，学習対象となるパラメータ数が少ないため，学習の計算コストを大幅に削減できるという利点がある．

https://github.com/google-research/computation-thru-dynamics