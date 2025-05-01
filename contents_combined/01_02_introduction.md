## Julia言語の使用法
### Julia言語の特徴
Julia言語は

本書を執筆するにあたり，なぜJulia言語を選択したかというのにはいくつか理由がある．

JuliaはJIT（Just-In-Time）コンパイルを用いており

JITコンパイラ

実行速度が高速であること．
ライセンスフリーであり，無料で使用できること．
線型代数演算が簡便に書けること．
Unicodeを使用できるため，疑似コードに近いコードを書けること．

他の言語の候補として，MATLAB, Pythonが挙げられた．MATLABは神経科学分野で根強く使用される言語であり，線型代数計算の記述が簡便である．なお，線型代数演算の記法に関してはJuliaはMATLABを参考に構築されたため，ほぼ同様に記述することができる．また，MATLABを使用するには有償ライセンスが必要である．ただし，互換性を持ったフリーソフトウェアであるOctaveが存在することは明記しておく．

Pythonは機械学習等の豊富なライブラリと書きやすさから広く利用されている言語である．ただし，numpyを用いないと高速な処理を書けない場合が多く，ナイーブな実装では実行速度が低下してしまう問題がある．線型代数計算も簡便に書くことができず，数式をコードに変換する際の手間が増えるという問題がある．

多重ディスパッチ（multiple dispatch）があることはJulia言語の大きな特徴である．

### Julia言語のインストール方法

Julia (\url{https://julialang.org/}) に

juliaup (\url{https://github.com/JuliaLang/juliaup}) でバージョン管理

また，2025年3月以降，Google Colab (\url{https://colab.google/}) においてPythonやRに並んでJuliaを選択して使用することが可能となっている．

### 使用するライブラリ

REPL
で`]` を入力することで，パッケージ管理モードに移行する．

本書で使用するJuliaライブラリは以下の通りである．

- IJulia: 開発環境
- PyPlot: 描画用ライブラリ
- LinearAlgebra: 高度な線形代数演算
- Random: 

Pythonではnumpyで完結するところをライブラリをいくつも読み込む必要がある点は欠点ではある．

描画用のライブラリには `PyPlot.jl` を使用した．`PyPlot` はPythonライブラリである `matplotlib` に依存したライブラリである．Juliaで完結させたい場合は `Plot.jl` や `Makie.jl` を使用することが推奨されるが，`PyPlot` (`matplotlib`) の方が高機能であるため，

Pythonがない場合は

```julia
julia> ENV["PYTHON"] = ""
julia> ]
pkg> build PyCall
```

Pythonを既にインストールしている場合は，

```julia
julia> ENV["PYTHON"] = raw"C:\Users\TakutoYamamoto\AppData\Local\Programs\Python\Python312\python.exe"
julia> ]
pkg> build PyCall
```

Windowsの場合
例としてPythonの実行ファイル (python.exe) への完全なパスを


### 開発環境

インタプリタ型言語である

vscode

筆者は（Pythonユーザーでもあるため）Jupyter Labを使用している．

JuliaのみでJupyter Labを使用するには

```julia
using IJulia
jupyterlab(detached=true)
```

とすればよい．ただし，この際にCondaを入れることになるため，別途Pythonをインストールしておく方が推奨される．

p.33

`Pluto.jl` を用いることも可能である

### Julia言語の基本構文

https://docs.julialang.org/en/v1/manual/noteworthy-differences/

### 命名規則
この節では，本書で用いるJuliaの変数名や関数名等に関する基本的な取り決めをまとめる．

#### 変数名
- `nt`: 時間ステップ数 (number of time steps)
- `t`, `tt`: 時間ステップのインデント