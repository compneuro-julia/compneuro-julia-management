# Juliaで学ぶ計算論的神経科学

このサイトは**計算論的神経科学 (Computational Neuroscience)** をプログラミング言語 [**Julia**](https://julialang.org/)を通して学習することを目標とします。

内容に関する指摘やコメントは各ページ末尾のコメント欄からしていただければ幸いです (GitHubアカウントが必要です)。

```{admonition} 記事で使用しているJuliaのバージョン
Julia 1.5.0
```

## 目次
- [まえがき](https://compneuro-julia.github.io/intro.html)

1. [はじめに](https://compneuro-julia.github.io/1_intro.html)
	1. 計算論的神経科学とは何か
	1. [記号の表記](https://compneuro-julia.github.io/notation.html)
1. [神経細胞のモデル](https://compneuro-julia.github.io/2_intro.html)
	1. 神経細胞の形態と膜電位変化
	1. [Hodgkin-Huxleyモデル](https://compneuro-julia.github.io/2-2_hodgkinhuxley.html)
	1. [FitzHugh–Nagumoモデル](https://compneuro-julia.github.io/2-3_fhn.html)
	1. [Leaky integrate-and-fire モデル](https://compneuro-julia.github.io/2-4_lif.html)
	1. [Izhikevich モデル](https://compneuro-julia.github.io/2-5_iz.html)
	1. ケーブル理論
	1. Multi-compartment モデル
	1. [Inter-spike interval モデル](https://compneuro-julia.github.io/2-8_isi.html)
	1. 確率的シナプス電流のノイズによる表現 (Langevin方程式 etc.)
	1. 確率的集団モデル (Fokker–Planck 方程式)
	1. 発火率モデル
1. [シナプス伝達のモデル](https://compneuro-julia.github.io/3_intro.html)
	1. [シナプス伝達](https://compneuro-julia.github.io/3-1_synapse.html)
	1. [Current-based vs Conductance-based シナプス](https://compneuro-julia.github.io/3-2_current-conductance-synapse.html)
	1. [指数関数型シナプスモデル](https://compneuro-julia.github.io/3-3_expo-synapse.html)
	1. [動力学モデル](https://compneuro-julia.github.io/3-4_kinetic-synapse.html)
	1. 増強シナプスと減衰シナプス
	1. [シナプス入力の重みづけ](https://compneuro-julia.github.io/3-6_synaptic-weighted.html)
	1. 電気シナプス
4. 神経回路網の構築 (発火率モデル)
5. 神経回路網の構築 (Spikingモデル)
6. 神経回路網の演算処理
7. 神経回路網の学習則
	1. 学習則と貢献度分配問題 (credit assignment problem)
	2. Hebb則
	3. STDP則
	4. Burst発火と可塑性
	5. 競合学習 (competitive learning)
	6. 勾配法と誤差逆伝播法 (backpropagation)
	7. 誤差逆伝播法の近似手法
	8. 経時的貢献度分配問題 (temporal credit assignment problem)
	9. RTRLとBPTT
	10. 適格度トレース (eligibility trace) とRTRLの近似手法
	11. Reservoir computing (FORCE etc.)
8. 神経系の非線形ダイナミクス
9. 連想記憶モデル
	1. Ising モデル
	2. Amari-Hopfield モデル
	3. Boltzmann machine
10. 情報理論と最適化原理
11. [ベイズ脳仮説](https://compneuro-julia.github.io/11_intro.html)
    1. [ベイズ統計の基礎](https://compneuro-julia.github.io/11-1_bayes_statistics.html)
    1. [Sparse coding (Olshausen & Field, 1996) モデル](https://compneuro-julia.github.io/11-2_sparse-coding.html)
    1. [Predictive coding (Rao & Ballard, 1999) モデル](https://compneuro-julia.github.io/11-3_predictive-coding-rao.html)
12. [強化学習](https://compneuro-julia.github.io/12_intro.html)
	1. [TD学習](https://compneuro-julia.github.io/12-1_td_learning.html)
	2. 分布型TD学習
13. 運動制御
14. 時空間の符号化
15. 神経細胞の形態と数理モデル

- [付録](https://compneuro-julia.github.io/appendix_intro.html)
	- [JuliaのTips集](https://compneuro-julia.github.io/tips.html)
	- [有用なリンク集](https://compneuro-julia.github.io/useful_links.html)


***

## 『ゼロから作るSpiking Neural Networks』について
『**ゼロから作るSpiking Neural Networks**』 (通称：SNN本) は**Python**でSpiking neural networksの構築と学習を実装することを目標とした技術同人誌です。本サイトはこの本をベースとして作成しています。技術書典7で頒布し、BOOTHで有料で販売してきましたが、無料で公開することとしました。それでも購入していただける方はBOOTHから購入いただければと思います。なお、物理本は完売し、再販の予定はありません。

```{admonition} 『ゼロから作るSpiking Neural Networks』Links
- [pdf](https://compneuro-julia.github.io/_static/pdf/SNN_from_scratch_with_python_ver2_05.pdf) (Ver. 2.05)
- [GitHub](https://github.com/takyamamoto/SNN-from-scratch-with-Python)
- [BOOTH](https://booth.pm/ja/items/1585421)
```



