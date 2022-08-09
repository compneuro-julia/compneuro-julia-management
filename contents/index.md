# Juliaで学ぶ計算論的神経科学

このサイトは**計算論的神経科学 (Computational Neuroscience)** をプログラミング言語 [**Julia**](https://julialang.org/)を通して学習することを目標とします．内容に関する指摘やコメントは各ページ末尾のコメント欄からしていただければ幸いです (GitHubアカウントが必要です)．

（2021/04/03追記）本サイト『Juliaで学ぶ計算論的神経科学』の書籍化企画が**講談社サイエンティフィク**で承認されました．出版予定は**2025年春**です．若輩者ゆえ荷が重くはありますが，神経科学・Julia界隈の方々には何卒応援いただきたく存じます．

```{admonition} 記事で使用しているJuliaのバージョン
Julia v1.8.0-rc1
```

## 依存ライブラリ
`IJulia, LinearAlgebra, Random, Parameters, Distributions, Statistics, Plots, PyPlot, ProgressMeter, MAT, MLDatasets, PyCall, StatsBase, Kronecker, BlockDiagonals, ToeplitzMatrices, BenchmarkTools, ImageTransformations, TestImages, ImageIO, ImageMagick, ColorTypes, FFTW`

※ PyCallに関してはPythonライブラリ `networkx`を使用．

## 目次
- [まえがき](https://compneuro-julia.github.io/preface.html)

### 第1部：
1. [はじめに](https://compneuro-julia.github.io/introduction/intro.html)
    1. [神経科学と数理モデル](https://compneuro-julia.github.io/introduction/computational-neuroscience.html)
	1. [記号の表記](https://compneuro-julia.github.io/introduction/notation.html)
    1. [Julia言語の基本構文](https://compneuro-julia.github.io/introduction/usage-julia-lang.html)
    1. [線形代数](https://compneuro-julia.github.io/introduction/linear-algebra.html)
    1. [微分方程式](https://compneuro-julia.github.io/introduction/differential-equation.html)
    1. [線形回帰と最小二乗法](https://compneuro-julia.github.io/introduction/linear-regression.html)
    1. [確率論と情報理論](https://compneuro-julia.github.io/introduction/probability-information-theory.html)
    1. [確率過程と確率微分方程式](https://compneuro-julia.github.io/introduction/stochastic-process-differential-equation.html)

### 第2部：
1. [神経細胞のモデル](https://compneuro-julia.github.io/neuron-model/intro.html)
    1. [神経細胞の形態と生理](https://compneuro-julia.github.io/neuron-model/neuron-physiol.html)
	1. [Hodgkin-Huxleyモデル](https://compneuro-julia.github.io/neuron-model/hodgkin-huxley.html)
	1. [FitzHugh–Nagumoモデル](https://compneuro-julia.github.io/neuron-model/fhn.html)
	1. [Leaky integrate-and-fire モデル](https://compneuro-julia.github.io/neuron-model/lif.html)
	1. [Izhikevich モデル](https://compneuro-julia.github.io/neuron-model/izhikevich.html)
	1. [Inter-spike interval モデル](https://compneuro-julia.github.io/neuron-model/isi.html)
    1. [神経突起の成長モデル](https://compneuro-julia.github.io/neuron-model/neurite-growth-model.html)
 
2. [シナプス伝達のモデル](https://compneuro-julia.github.io/synapse-model/intro.html)
	1. [シナプスの形態と生理](https://compneuro-julia.github.io/synapse-model/synapse-physiol.html)
	1. [Current-based / Conductance-based シナプス](https://compneuro-julia.github.io/synapse-model/current-conductance-synapse.html)
	1. [指数関数型シナプスモデル](https://compneuro-julia.github.io/synapse-model/expo-synapse.html)
	1. [動力学モデル](https://compneuro-julia.github.io/synapse-model/kinetic-synapse.html)
	1. [シナプス入力の重みづけ](https://compneuro-julia.github.io/synapse-model/synaptic-weighted.html)
	1. [動的シナプス](https://compneuro-julia.github.io/synapse-model/dynamical-synapses.html)

3. 神経回路網の構築
    1. 神経細胞間の接続
    1. ランダムネットワークの構築

4. [神経回路網の演算処理](https://compneuro-julia.github.io/neuronal-computation/intro.html)
	1. [ゲイン調節と四則演算](https://compneuro-julia.github.io/neuronal-computation/neuronal-arithmetic.html)
    1. 正規化
    1. 樹状突起計算

### 第3部：
1. [局所学習則](https://compneuro-julia.github.io/learning-rule/intro.html)
    1. 学習と学習則
    2. [Hebb則・BCM理論・Oja則・Sanger則・非線形Hebb則](https://compneuro-julia.github.io/local-learning-rule/hebbian-learning.html)
    3. STDP則
    4. [自己組織化マップと視覚野の構造](https://compneuro-julia.github.io/local-learning-rule/self-organizing-map.html)

2. [エネルギーベースモデル](https://compneuro-julia.github.io/energy-based-model/intro.html)
	1. [エネルギーベースモデル](https://compneuro-julia.github.io/energy-based-model/energy-based-model.html)
	1. [Hopfield モデル](https://compneuro-julia.github.io/energy-based-model/hopfield-model.html) 
    1. [Boltzmann マシン](https://compneuro-julia.github.io/energy-based-model/boltzmann-machine.html) 
    1. [スパース符号化](https://compneuro-julia.github.io/energy-based-model/sparse-coding.html)
    1. [予測符号化](https://compneuro-julia.github.io/energy-based-model/predictive-coding.html)
   
3. [貢献度分配問題の解決策](https://compneuro-julia.github.io/solve-credit-assignment-problem/intro.html)
    1. 貢献度分配問題
	1. [勾配法と誤差逆伝播法](https://compneuro-julia.github.io/solve-credit-assignment-problem/backpropagation-zipser-andersen.html)
    1. [深層線形ニューラルネットの学習ダイナミクス](https://compneuro-julia.github.io/solve-credit-assignment-problem/linear-network-learning-dynamics.html)
	1. [BPTT (backpropagation through time)](https://compneuro-julia.github.io/solve-credit-assignment-problem/bptt.html)
    1. RTRL
    1. 適格度トレースによるRTRLの近似
    1. 予測符号化による誤差逆伝播法の近似
    1. SNNの訓練法
    
### 第4部：
1. [運動制御](https://compneuro-julia.github.io/motor-learning/intro.html)
    1. [躍度最小モデル](https://compneuro-julia.github.io/motor-learning/minimum-jerk.html)
    1. [終点誤差分散最小モデル](https://compneuro-julia.github.io/motor-learning/minimum-variance.html)
    1. [最適フィードバック制御モデル](https://compneuro-julia.github.io/motor-learning/optimal-feedback-control.html)
    1. [無限時間最適制御モデル](https://compneuro-julia.github.io/motor-learning/infinite-horizon-ofc.html)

2. [強化学習](https://compneuro-julia.github.io/reinforcement-learning/intro.html)
    1. [TD学習](https://compneuro-julia.github.io/reinforcement-learning/td-learning.html)

### 第5部：
1. [神経回路網によるベイズ推論](https://compneuro-julia.github.io/bayesian-brain/intro.html)
    1. [ベイズ脳仮説と神経活動による不確実性の表現](https://compneuro-julia.github.io/bayesian-brain/neural-uncertainty-representation.html)
    2. [ベイズ線形回帰](https://compneuro-julia.github.io/appendix/bayesian-linear-regression.html)
    3. [マルコフ連鎖モンテカルロ法](https://compneuro-julia.github.io/bayesian-brain/mcmc.html)
    4. [神経サンプリング](https://compneuro-julia.github.io/bayesian-brain/neural-sampling.html)
    5. [確率的集団符号化](https://compneuro-julia.github.io/bayesian-brain/probabilistic-population-coding.html)
    6. [分位点・エクスペクタイル回帰による分布符号化](https://compneuro-julia.github.io/appendix/quantile-expectile-regression.html)
    7. 自由エネルギー原理

- [付録](https://compneuro-julia.github.io/appendix/intro.html)
	1. [ラット自由行動下の軌跡のシミュレーション](https://compneuro-julia.github.io/appendix/rat-trajectory.html)
    1. [格子細胞のデコーディング](https://compneuro-julia.github.io/appendix/grid-cells-decoding.html)
	1. [Slow Feature Analysis](https://compneuro-julia.github.io/appendix/slow-feature-analysis.html)
    1. [グラフ理論とネットワークモデル](https://compneuro-julia.github.io/appendix/graph-theory-network-model.html)
	1. [有用なリンク集](https://compneuro-julia.github.io/appendix/useful-links.html)
	1. [Jupyter bookの使い方 (Julia言語版)](https://compneuro-julia.github.io/appendix/usage-jupyter-book.html)
    
## 『ゼロから作るSpiking Neural Networks』について
『**ゼロから作るSpiking Neural Networks**』 (通称：SNN本) は**Python**でSpiking neural networksの構築と学習を実装することを目標とした技術同人誌です．本サイトはこの本をベースとして作成しています．技術書典7で頒布し，BOOTHで有料で販売してきましたが，無料で公開することとしました．それでも購入していただける方はBOOTHから購入いただければと思います．なお，物理本は完売し，再販の予定はありません．

```{admonition} 『ゼロから作るSpiking Neural Networks』Links
- [pdf](https://compneuro-julia.github.io/_static/pdf/SNN_from_scratch_with_python_ver2_1.pdf) (Ver. 2.1)
- [GitHub](https://github.com/takyamamoto/SNN-from-scratch-with-Python)
- [BOOTH](https://booth.pm/ja/items/1585421)
```



