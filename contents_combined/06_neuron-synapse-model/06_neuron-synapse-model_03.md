## マルチコンパートメントモデル


L5-PC minimal model

Hodgkin-Huxleyの定式化

$k$ をイオンの種類として，

$$
I_k:=g_k m_k^x h_k^y (V_i - E_k)
$$

$g_k$ は最大コンダクタンス，$m_k, h_k$ はそれぞれ活性化ゲート変数，不活性化ゲート変数である．$x, y$ は指数である．$V_i$ は $i$ 番目のコンパートメントの膜電位であり，$E_k$ は平衡電位である．
 
https://github.com/beaherrera/2-compartments_L5-PC_model/tree/master/IonicCurrents

ケーブル方程式の離散化

https://neuronaldynamics.epfl.ch/online/Ch3.S4.html

https://github.com/orena1/NEURON_tutorial/tree/master
https://github.com/orena1/NEURON_tutorial/blob/master/Jupyter_notebooks/Layer_5b_pyramidal_cell_Calcium_Spike.ipynb

Ball and Stick model

 E. Hay, S. Hill, F. Schürmann, H. Markram and I. Segev (2011-07) Models of neocortical layer 5b pyramidal cells capturing a wide range of dendritic and perisomatic active properties. PLoS Comput Biol 7 (7), pp. e1002107
 

three compartment model
https://pmc.ncbi.nlm.nih.gov/articles/PMC4516889/

https://www.jneurosci.org/content/40/44/8513.full#ref-100

https://www.nature.com/articles/s41467-019-11537-7
https://www.science.org/doi/10.1126/science.1127240

https://www.jneurosci.org/content/40/44/8513.full#sec-2


神経細胞の電気的活動を詳細に記述するためには，単一の点としてニューロンをモデル化する単純なモデル（例：leaky integrate-and-fireモデル）では不十分である．特に，樹状突起や軸索といった構造的に異なる部分の電気的性質を記述するためには，multi compartment model（多区画モデル）と呼ばれる手法が用いられる．このモデルでは，ニューロン全体を電気回路として捉え，各構造（区画）を電気的に独立した要素として記述し，それらを電気的に接続することで，ニューロン全体の動態を近似する．

各区画（コンパートメント）は，膜容量 $C_m$，漏洩電導 $g_L$，静止電位 $E_L$ を備えたRC回路として表現される．隣接するコンパートメント同士は，軸索や樹状突起を介した軸索流によって結合され，その伝導は軸内抵抗 $R_a$ または電導 $g_{ij}$ を通じて記述される．

コンパートメント $i$ における膜電位 $V_i(t)$ の時間発展は，ケーブル方程式を離散化した以下の形式で表される：

$$
C_m \frac{dV_i}{dt} = -g_L (V_i - E_L) + \sum_{j \in \mathcal{N}(i)} g_{ij} (V_j - V_i) + I^{\text{ext}}_i(t)
$$

ここで，\(\mathcal{N}(i)\) はコンパートメント \(i\) に隣接するコンパートメントの集合，\(g_{ij}\) は区画 \(i\) と \(j\) を接続する電導，\(I^{\text{ext}}_i(t)\) は外部から注入される電流を表す．この方程式は，すべてのコンパートメントに対して定義され，結果として連立微分方程式系が得られる．

このmulti compartment modelを用いることで，樹状突起でのシナプス入力の時空間的な統合や，軸索起始部で発生した活動電位（action potential, AP）が樹状突起へ逆行性に伝搬する現象（back-propagating action potential, bAP）を記述・再現することが可能となる．bAPは，活動電位が軸索起始部で発生した後，ナトリウムチャネルやカリウムチャネルの存在により樹状突起へと逆向きに伝搬するものであり，樹状突起上のシナプス可塑性に重要な役割を果たすと考えられている．

モデル内でこの現象を再現するには，樹状突起区画にも活動電位の伝搬に関与する電位依存性ナトリウムチャネル（\(I_{\text{Na}}\)）やカリウムチャネル（\(I_{\text{K}}\)）を含めたHodgkin-Huxley型のイオン電流モデルを導入する必要がある．例えば，コンパートメント \(i\) における電流項は以下のように拡張される：

$$
C_m \frac{dV_i}{dt} = -g_L (V_i - E_L) - I_{\text{Na}, i}(V_i, m_i, h_i) - I_{\text{K}, i}(V_i, n_i) + \sum_{j \in \mathcal{N}(i)} g_{ij} (V_j - V_i) + I^{\text{ext}}_i(t)
$$

ここで，\(m_i, h_i, n_i\) はイオンチャネルのゲーティング変数であり，それぞれ別の微分方程式に従って時間発展する．これにより，活動電位の発生とその伝播，さらに逆行性伝播が自然にモデルに組み込まれることになる．