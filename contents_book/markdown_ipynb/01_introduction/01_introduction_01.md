# 第1章：はじめに
## 本書の目的と構成
### 神経科学におけるモデルの意義
本書では、神経科学における数理モデルの構築とJulia言語での実装を中心的な主題とし、その背景にある計算論的理論と実践的手法を扱う。初めに、神経科学におけるモデルの意義について整理しておこう。

神経科学の最終的な目的は、端的に言えば「脳神経系を理解すること」にある。この目標に向けて、自然科学に共通するように、神経科学もまた実験と理論という二つの柱に支えられて発展してきた。実験は観察や計測を通じて実データを得る行為であり、理論はそれらのデータを抽象化・統合し、予測や仮説の創出へと昇華させる枠組みである。特にモデルは、次のような多面的な役割を担っている \citep{Blohm2020-vc}：
（1）仮説の駆動と明示化、（2）複雑な知見の整理と統合、（3）観察結果の再現や予測、（4）仮想実験の実行、（5）科学的コミュニケーションの明確化、（6）臨床や技術への応用可能性の提供である。このように、モデルは単なる計算装置ではなく、科学的思考そのものの外化であり、神経科学の進展に不可欠な知的道具である。

とはいえ、「脳を理解する」とは具体的に何を意味するのか。この問いに対する明確な答えは一様ではない。神経科学の実験は多様であり，

実験対象：線虫 (c. elegans)，ハエ (drosophila)，鳥類，コウモリ，哺乳類，甲殻類，rodent，無脊椎動物（タコ），ウマ，ブタ，（サル，ネコ，マウス），ヒト
ミクロとマクロ：分子生物学的，心理学実験
培養系と生体，in vivo, in vitro
構造と機能：解剖学，電気生理学・イメージング

本書は構造も少々取り扱うが，主には機能を取り扱う．

本書では、こうした理解の枠組みを与えるために、**Marrの三つのレベル**（Marr’s Three Levels）** \citep{Marr1982-wk} およびDayanとAbbottによる3レベルを紹介する．

対応するモデル


解釈 (interpretive), 説明 (descriptive)  and mechanistic approaches


Interpretive model（解釈的モデル, why）：行動や最適性の観点から機構を説明する（例：ベイズ推論、強化学習）

Descriptive model（記述的モデル, how）：データの構造や統計的特徴をそのまま記述する（例：PSTH、tuning curves）

Mechanistic model（機構的モデル, what）：要素間の因果的な関係を定式化する（例：LIFモデル、STDP則）

抽象的な意味的構成から，生体現象に即した具体的なモデル





---

https://pubmed.ncbi.nlm.nih.gov/38252109/

https://www.sciencedirect.com/science/article/pii/S0959438819300728?via%3Dihub

表を参考に，統合した階層を書く．

https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP279549

Dayan, P., & Abbott, L. F. (2005). Theoretical neuroscience. MIT Press.

Dayanらの序文

Theoretical analysis and computational modeling are important tools for characterizing what nervous systems do, determining how they function, and understanding why they operate in particular ways. Neuroscience encompasses approaches ranging from molecular and cellular studies to human psychophysics and psychology. Theoretical neuroscience encourages crosstalk among these subdisciplines by constructing compact representations of what has been learned, building bridges between different levels of description, and identifying unifying concepts and principles. In this book, we present the basic methods used for these purposes and discuss examples in which theoretical approaches have yielded insight into nervous system function. The questions what, how, and why are addressed by descriptive, mechanistic, and interpretive models, each of which we discuss in the following chapters. Descriptive models summarize large amounts of experimental data compactly yet accurately, thereby characterizing what neurons and neural circuits do. These models may be based loosely on biophysical, anatomical, and physiological findings, but their primary purpose is to describe phenomena, not to explain them. Mechanistic models, on the other hand, address the question of how nervous systems operate on the basis of known anatomy, physiology, and circuitry. Such models often form a bridge between descriptive models couched at different levels. Interpretive models use computational and information-theoretic principles to explore the behavioral and cognitive significance of various aspects of nervous system function, addressing the question of why nervous systems operate as they do.

情報処理タスクを実行するあらゆる機械を理解するために必要な3つのレベル



。これは、脳における情報処理過程を以下の三段階で整理する視点である：

計算理論（computational theory）：対象とする現象において、何をどのように計算するのか、入力と出力の対応関係を定義する。

表現とアルゴリズム（representation and algorithm）：計算理論で定められた変換を、どのような内部表現と手続きにより実現するかを明示する。

実装（implementation）：それらの手続きを、神経回路やハードウェアといった物理基盤上でどのように具現化するかを示す。

本書の立場は、これら三つのレベルを分断されたものではなく、相互に往還可能な理解のスケールと見なすことにある。とりわけ第(3)の実装レベルに重きを置き、モデルを読者自身の手で計算機上に構築・実行し、理論的仮説を数値的に再現・検証する力を養うことを目標とする。その意味で本書は、数式をコードに変換するための「実践的な翻訳辞典」としても機能する。

また、本書で取り上げるモデルの多くは機械学習と関係している。これは神経科学と機械学習が長年にわたり相互作用してきた歴史を反映している。神経科学から機械学習への影響には、ニューラルネットワークの構造や記憶・注意といった機能的モデルがあり、逆に機械学習の発展が神経科学にもたらしたものとして、強化学習に基づく意思決定理論や、ベイズ的知覚理論（いわゆる「ベイズ脳仮説」）などが挙げられる \citep{Hassabis2017-zm}。

筆者の立場は、神経科学は機械学習の素材や工学的応用のためにあるのではなく、脳そのものの理解という自律した目的を持つ学問であるというものである。そのため本書では、「機械学習から神経科学への応用」という観点、すなわちアルゴリズム的知見を手がかりに神経過程を理解するという方向性を重視する。この視点はBlohmらが述べるように、現象に即して問いを明確化し、モデルの仮定と評価基準を明示的に定めるという「設計としての建設的モデル化（modeling as design）」の理念にも通じる。



### 本書の構成
本書は、計算論的神経科学および神経モデルの理解を深めるために、Julia言語を用いて数理モデルを実装しながら学習を進める形式を採用している。第1章では、Julia言語の基本的な使用法に加え、本書全体で用いる数学的記法について解説する。あわせて、神経科学における「学習」と「予測」という枠組みのもと、本書全体の立場を概説する。

第2章から第5章では、発火率モデルを用いた神経回路網の構成とその学習則について段階的に説明する。第2章では、神経細胞の基本的な生理学を導入し、発火率モデルを用いた神経活動の定式化と、Hebb則やOja則などの局所学習則に基づく単純なネットワークの構築について扱う。第3章では、同じく局所学習則でありながら、ネットワーク全体のエネルギーを最小化する観点に基づくエネルギーベースモデル（例えばHopfieldネットワークやボルツマンマシン）について解説する。第4章では、誤差逆伝播法（Backpropagation）に基づく現代的なニューラルネットワークを扱い、そこで発生する貢献度分配問題（credit assignment problem）を取り上げる。第5章では、再帰型ニューラルネットワーク（RNN）を導入し、時間方向での貢献度分配（経時的貢献度分配）の問題とその学習方法を解説する。

第6章と第7章では、スパイキングニューラルネットワーク（SNN）を取り上げる。第6章では、これまでのネットワークレベルの議論から個々の神経細胞とシナプスの動態に立ち戻り、スパイク発生とシナプス伝達の生物物理学的モデルを取り扱う。第7章では、SNNにおけるネットワーク構築と学習について、発火率モデルで扱った学習則や誤差伝播法との接続も含めて解説する。

第8章以降は、応用的・発展的トピックを各論的に扱う。第8章では、リザバーコンピューティングという枠組みに基づき、複雑な動的表現を活用する発火率モデルおよびスパイキングモデルについて紹介する。第9章では、神経回路網がベイズ推論を実現する可能性について、確率的計算の関係から考察する。第10章では、運動学習における最適制御問題に対して、脳がどのような計算を行っているかをモデルベースで探る。第11章では、強化学習の基本的枠組みと、大脳基底核との関係について説明する。第12章は補足的な章として、ネットワーク構造、神経形態学、グリア細胞など、本書で取り上げきれなかった関連トピックについて解説する。
