## RTRLとBPTTの生理学的実装の困難点
時空間的に局所
時間的に局所 (local)

過去向き方式はオンライン性が強く，一度に扱うパラメータ依存を１つの損失にまとめるため，リアルタイム更新が可能であるが，その分、「過去→現在」の微分を保持する大きなテンソル（感度行列）を圧縮する工夫が必要となる。未来向き方式は「現在→未来」の影響を直接扱うため，パラメータ感度の保持は不要だが，未来の損失を参照する逆伝播がオンラインでは難しく，しばしばトランケート（打ち切り）を伴う。  


損失に対する状態感度
状態に対するパラメータ感度

BPTTは


いずれの手法も，時系列モデルの状態更新則が  

$$
\mathbf{h}_t = F\bigl(\mathbf{h}_{t-1},\,\mathbf{x}_t;\,\theta\bigr)
$$  

のように，状態は過去から未来への一方向性を持つため，過去の状態を未来の状態で微分する操作 $\partial \mathbf{h}_{t-1}/\partial \mathbf{h}_t$ は常にゼロとなる．

$\frac{\partial \mathbf{h}_t}{\partial \theta} \in \mathbb{R}^{d \times |\theta|}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \in \mathbb{R}^{d\times d}$

状態感度 (state sensitivity) $\frac{\partial \mathbf{h}_t}{\partial \theta}$

RTRLはパラメータを保持できない．
BPTTは未来から過去へ戻る必要がある．

脳は過去の状態を全て保存して逆向きに再生することは困難である．
再活性化などで可能となっている部分もあるが，全ての状態を保存しておくのは難しい．

海馬においては状態の逆再生 (reverse replay) が行われることが報告されている．

https://pubmed.ncbi.nlm.nih.gov/16474382/
https://www.nature.com/articles/nature04587

https://pmc.ncbi.nlm.nih.gov/articles/PMC6013068/
https://www.science.org/doi/10.1126/science.ads4760
https://www.biorxiv.org/content/10.1101/2023.02.19.529130v4

https://www.nature.com/articles/s41586-025-08828-z