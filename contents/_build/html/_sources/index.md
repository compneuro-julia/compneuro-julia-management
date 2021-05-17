# Juliaで学ぶ計算論的神経科学

このサイトは**計算論的神経科学 (Computational Neuroscience)** をプログラミング言語 [**Julia**](https://julialang.org/)を通して学習することを目標とします．内容に関する指摘やコメントは各ページ末尾のコメント欄からしていただければ幸いです (GitHubアカウントが必要です)．

（2021/04/03追記）本サイト『Juliaで学ぶ計算論的神経科学』の書籍化企画が**講談社サイエンティフィク**で承認されました．出版予定は**2025年春**です．若輩者ゆえ荷が重くはありますが，神経科学・Julia界隈の方々には何卒応援いただきたく存じます．

```{admonition} 記事で使用しているJuliaのバージョン
Julia 1.6.0
```

### 依存ライブラリ
`IJulia, LinearAlgebra, Random, Parameters, Distributions, Statistics, Plots, PyPlot, ProgressMeter, MAT, MLDatasets, PyCall, StatsBase, BlockDiagonals, ToeplitzMatrices, BenchmarkTools, TestImages, ImageIO, ImageMagick, ColorTypes`

※ PyCallに関してはPythonライブラリ `scipy`を使用．

## 目次
- [まえがき](https://compneuro-julia.github.io/intro.html)

1. [はじめに](https://compneuro-julia.github.io/introduction/intro.html)
	1. [記号の表記](https://compneuro-julia.github.io/introduction/notation.html)
1. [神経細胞のモデル](https://compneuro-julia.github.io/neuron-model/intro.html)
	1. [Hodgkin-Huxleyモデル](https://compneuro-julia.github.io/neuron-model/hodgkin-huxley.html)
	1. [FitzHugh–Nagumoモデル](https://compneuro-julia.github.io/neuron-model/fhn.html)
	1. [Leaky integrate-and-fire モデル](https://compneuro-julia.github.io/neuron-model/lif.html)
	1. [Izhikevich モデル](https://compneuro-julia.github.io/neuron-model/izhikevich.html)
	1. [Inter-spike interval モデル](https://compneuro-julia.github.io/neuron-model/isi.html)
1. [シナプス伝達のモデル](https://compneuro-julia.github.io/synapse-model/intro.html)
	1. [シナプス伝達](https://compneuro-julia.github.io/synapse-model/synapse.html)
	1. [Current-based / Conductance-based シナプス](https://compneuro-julia.github.io/synapse-model/current-conductance-synapse.html)
	1. [指数関数型シナプスモデル](https://compneuro-julia.github.io/synapse-model/expo-synapse.html)
	1. [動力学モデル](https://compneuro-julia.github.io/synapse-model/kinetic-synapse.html)
	1. [シナプス入力の重みづけ](https://compneuro-julia.github.io/synapse-model/synaptic-weighted.html)
1. [神経回路網の学習則](https://compneuro-julia.github.io/learning-rule/intro.html)
	1. [勾配法と誤差逆伝播法 (Zipser-Andersenモデルを例にして)](https://compneuro-julia.github.io/learning-rule/backpropagation-zipser-andersen.html)
	1. [BPTT (backpropagation through time)](https://compneuro-julia.github.io/learning-rule/bptt.html)
	1. [深層線形ニューラルネットの学習ダイナミクス](https://compneuro-julia.github.io/learning-rule/linear-network-learning-dynamics.html)
1. [情報理論と最適化原理](https://compneuro-julia.github.io/information-theory/intro.html)
	1. [統計と情報理論の基礎](https://compneuro-julia.github.io/information-theory/statistics-information.html)
	1. [Slow Feature Analysis](https://compneuro-julia.github.io/information-theory/slow-feature-analysis.html)
1. [連想記憶モデル](https://compneuro-julia.github.io/associative-memory-model/intro.html)
	1. [エネルギーベースモデル (Energy-based model)](https://compneuro-julia.github.io/associative-memory-model/energy-based-model.html) 
	1. [Amari-Hopfield モデル](https://compneuro-julia.github.io/associative-memory-model/amari-hopfield-model.html) 
	1. [Boltzmann マシン](https://compneuro-julia.github.io/associative-memory-model/boltzmann-machine.html) 
1. [ベイズ脳仮説と生成モデル](https://compneuro-julia.github.io/bayesian-brain/intro.html)
    1. [ベイズ統計の基礎](https://compneuro-julia.github.io/bayesian-brain/bayes-statistics.html)
    1. [スパース符号化 (sparse coding)](https://compneuro-julia.github.io/bayesian-brain/sparse-coding.html)
    1. [予測符号化 (predictive coding)](https://compneuro-julia.github.io/bayesian-brain/predictive-coding.html)
1. [強化学習](https://compneuro-julia.github.io/reinforcement-learning/intro.html)
    1. [TD学習](https://compneuro-julia.github.io/reinforcement-learning/td-learning.html)
1. [運動制御](https://compneuro-julia.github.io/motor-learning/intro.html)
    1. [躍度最小モデル](https://compneuro-julia.github.io/motor-learning/minimum-jerk.html)
    1. [終点誤差分散最小モデル](https://compneuro-julia.github.io/motor-learning/minimum-variance.html)
    1. [最適フィードバック制御モデル (optimal feedback control; OFC)](https://compneuro-julia.github.io/motor-learning/optimal-feedback-control.html)
    1. [無限時間最適制御モデル (infinite-horizon optimal feedback control model)](https://compneuro-julia.github.io/motor-learning/infinite-horizon-ofc.html)
1. [時空間の符号化](https://compneuro-julia.github.io/spatiotemporal-coding/intro.html)
    1. [格子細胞のデコーディング](https://compneuro-julia.github.io/spatiotemporal-coding/grid-cells-decoding.html)
- [付録](https://compneuro-julia.github.io/appendix/intro.html)
	1. [線形回帰と最小二乗法](https://compneuro-julia.github.io/appendix/linear-regression.html)
	1. [分位点回帰とエクスペクタイル回帰](https://compneuro-julia.github.io/appendix/quantile-expectile-regression.html)
	1. [ラット自由行動下の軌跡のシミュレーション](https://compneuro-julia.github.io/appendix/rat-trajectory.html)
	1. [JuliaのTips集](https://compneuro-julia.github.io/appendix/tips.html)
	1. [有用なリンク集](https://compneuro-julia.github.io/appendix/useful-links.html)
	1. [Jupyter bookの使い方 (Julia言語版)](https://compneuro-julia.github.io/appendix/usage-jupyter-book.html)


***

## 『ゼロから作るSpiking Neural Networks』について
『**ゼロから作るSpiking Neural Networks**』 (通称：SNN本) は**Python**でSpiking neural networksの構築と学習を実装することを目標とした技術同人誌です．本サイトはこの本をベースとして作成しています．技術書典7で頒布し，BOOTHで有料で販売してきましたが，無料で公開することとしました．それでも購入していただける方はBOOTHから購入いただければと思います．なお，物理本は完売し，再販の予定はありません．

```{admonition} 『ゼロから作るSpiking Neural Networks』Links
- [pdf](https://compneuro-julia.github.io/_static/pdf/SNN_from_scratch_with_python_ver2_1.pdf) (Ver. 2.1)
- [GitHub](https://github.com/takyamamoto/SNN-from-scratch-with-Python)
- [BOOTH](https://booth.pm/ja/items/1585421)
```



