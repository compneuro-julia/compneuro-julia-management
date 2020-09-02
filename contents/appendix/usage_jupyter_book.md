# Jupyter bookの使い方 (Julia言語版)
このサイトの構築方法について (2020/09/02)。

## Julia言語のサイトを構築する方法
Julia言語でサイトを構築するには[**Documenter.jl**](https://github.com/JuliaDocs/Documenter.jl)や[**Franklin.jl**](https://github.com/tlienart/Franklin.jl), [**Weave.jl**](https://github.com/JunoLab/Weave.jl)などのパッケージを用いるのが一般的である。これらはMarkdown (.mdや.jmd)などのファイルをHTMLに変換できるが、Jupyter notebookからHTMLに変換するのはやや手間である (と自分は思っているが、今後改善される可能性は十分にある)。詳細は次項で説明するが、このサイトを構築している[**Jupyter book**](https://jupyterbook.org/intro.html)はJuliaで完結してはいないものの、より簡単にJupyter notebookをHTMLに変換できる。また、LaTeX形式やpdf形式に変換することもできる (これはWeave.jlでも同じことができるということは述べておく)。

- [Julia の Documenter.jl でホームページを作成する準備． - Qiita](https://qiita.com/SatoshiTerasaki/items/b0ac17088f3b2c374099)
- [Weave.jl で Markdown + Julia の文章をHTMLに変換して自分のホームページで公開しよう - Qiita](https://qiita.com/SatoshiTerasaki/items/3a913f897b2ef4b82979)
- [Weave.jlを使ってJuliaのノートブックを作成する - システムとモデリング](http://otepipi.hatenablog.com/entry/2019/03/30/221635)

## Jupyter bookの導入
[Jupyter book](https://jupyterbook.org/intro.html)は[Sphinx](https://www.sphinx-doc.org/ja/master/)を用いてMarkdownやJupyter notebookからサイトを生成するツールである (以前は[Jekyll](https://jekyllrb.com/)が用いられていた)。Sphinx自体はPythonで書かれた文章作成ツールだが、導入においてPythonコードを編集することは基本的にない。

また、重要な点として、**Jupyter notebookのカーネルはPythonでなくてもよい**ということが挙げられる (Python, R, Julia, Ruby, Go, Scala, Perlなどなど)。カーネルとして用いることのできる言語の一覧は[Jupyter kernels · jupyter/jupyter Wiki](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)を参照。Python以外の言語を用いている例としては[Other Programming Languages](https://myst-nb.readthedocs.io/en/latest/examples/coconut-lang.html)がある。ここではCoconutというPython製の関数型言語が用いられている。

なお、同じサイト内で異なる言語のカーネルを持つJupyter notebookを元としたページも作成できる (例えばあるページはPython, 次のページはJuliaといったように)。同じページ内で複数の言語を用いるのはまだできないようだが。

とはいえ、Pythonカーネルでやるのが一番楽である。以下にJuliaカーネルを用いる場合の手順を記すが、Pythonカーネルの場合はJuliaのinstallの手順を飛ばせばよい。

### 1. Anacondaをinstall
[Anaconda](https://www.anaconda.com/products/individual) または [miniconda](https://docs.conda.io/en/latest/miniconda.html)をinstallする (使用しているOSに応じてinstallerを選択する)。

### 2. jupyter-notebookをinstall
pip (またはconda)で[Jupyter notebook](https://jupyter.org/)をinstallする。

```
$ pip install jupyter
```

installしたTerminal (Windowsならanaconda prompt)でJupyter notebookを立ち上げられることを確認する。

```
$ jupyter notebook
```

### 3. Juliaをinstallし、Ijuliaをinstall
[Julia](https://julialang.org/)をinstallする。次にJuliaを立ち上げて、REPLで`]`を入力してpkg modeにし、

```
pkg> add IJulia
```

により[Ijulia.jl](https://github.com/JuliaLang/IJulia.jl)をinstallする。この段階で、Jupyter notebookを立ち上げて新規にnotebookを作成する際にJuliaカーネルを選択できるはずである。うまくいっていない場合はJuliaのPATHが通っていないなどが考えられる。

### 4. jupyter bookをinstall
pip (またはconda)で`jupyter-book` をinstallする。

```
$ pip install jupyter-book
```

### 5. テンプレートの作成
[Overview and installation](https://jupyterbook.org/start/overview.html)を参照。Terminal (Windowsならanaconda prompt)で次のようにしてテンプレートを作成する。

```
$ cd hogehoge
$ jupyter-book create mybookname
```

ただし、`mybookname`はサイトの構成ファイルを保存するディレクトリ名であり、変更可能である。`jupyter-book`は`jb`と省略できるので、

```
$ jb create mybookname
```

としてもよい。実行後は次のようなファイルが`mybookname`の下に生成される。

```
mybookname/
├── _config.yml
├── _toc.yml
├── content.md
├── intro.md
├── markdown.md
├── notebooks.ipynb
└── references.bib
```

このうち、Markdownファイル (.md)とJupyter notebookファイル (.ipynb)および参考文献を記述するためのBiBTeXファイル(.bib)が実際のサイトの元ファイルとなる。`_config.yml`はサイトの情報を指定するファイルであり、`_toc.yml`はサイトの構成を指定するファイルである。また、ディレクトリとファイルの構成例として`quantecon-mini-example` ([Github pages](https://executablebooks.github.io/quantecon-mini-example/docs/index.html), [Github](https://github.com/executablebooks/quantecon-mini-example)) が用意されている。

Jupyter bookの機能を確認する場合は先に [8. サイトのbuild](#build)を参照。

### 6. `_config.yml`にサイトの情報を記述
[Configure book settings](https://jupyterbook.org/customize/config.html)を参照。サイトの名前、著者、ロゴ、リポジトリへのリンク、colabで立ち上げるボタンの追加、など様々なことが設定できる。

### 7. `_toc.yml`にサイトの構成を記述
[Table of Contents structure](https://jupyterbook.org/customize/toc.html)を参照。各ファイルをどのような構成でサイトに変換するか、ということを指定する。

### 8. サイトのbuild
コンテンツを含むディレクトリを`./mybookname`としたとき

```
jb build mybookname
```

によりbuildする。このとき、Jupyter notebookはbuild時に実行される。Ijulia.jlが入っていなかったり、実行不可能であればエラーが生じる。build完了後、`./build/html`というディレクトリの下に他の依存ファイルを含めたサイトのHTMLが生成される。

### 9. GitHub pagesでサイトを公開する
`./build/html`の中身を同じリポジトリの`gh-pages`ブランチにcommitするか、別のリポジトリを用意してcommitする。本サイトの場合はコンテンツの管理は<https://github.com/compneuro-julia/compneuro-julia-management>で、サイトのホスティングは<https://github.com/compneuro-julia/compneuro-julia.github.io>で行っている。

なお、GitHub pagesはデフォルトではJekyllで処理されるので、`.nojekyll`という名前の空ファイルを作成しておく (空ファイルだとuploadされない場合もあるので注意)。

- [GitHub Pagesで普通の静的ホスティングをしたいときは .nojekyll ファイルを置く - Qiita](https://qiita.com/sky_y/items/b96ae52c90457bcd7846)

## MyST Markdown形式について
Jupyter bookは通常のMarkdown記法に加え、**MyST** (Markedly Structured Text)と呼ばれる形式のMarkdown記法に対応している。例えば

````
```{note}
これはノートです。
```
````

と記述すれば次のように変換される。

```{note}
これはノートです。
```

これはMarkdownファイルにも、Jupyter notebookのMarkdown cellにも記述できる。詳細は[MyST Markdown Overview](https://jupyterbook.org/content/myst.html)や[MyST Cheat Sheet](https://jupyterbook.org/reference/cheatsheet.html)などを参照 (特に後者はMySTでどのようなことができるか一目でわかる)。

### 数式について
通常のMarkdownやJupyter notebookと同様に行える。次のように入力すれば

```tex
$$
F(\omega) = \cfrac{1}{\sqrt{2\pi}}\int_{-\infty}^{+\infty}f(t)e^{i\omega t}dt
$$
```

以下のように出力される。

$$
F(\omega) = \cfrac{1}{\sqrt{2\pi}}\int_{-\infty}^{+\infty}f(t)e^{i\omega t}dt
$$

### 入力・出力を隠す
Jupyter notebookのcellのtagを変更してcellの入力や出力を隠すことができる。詳細は[Hiding cell contents](https://myst-nb.readthedocs.io/en/latest/use/hiding.html)。



## Syntax highlightingについて
Sphinxはコードの構文解析を[Pygments](https://pygments.org/)によって行っている (cf. [sphinx/highlighting.py](https://github.com/sphinx-doc/sphinx/blob/3.x/sphinx/highlighting.py))。これまでは全てPythonLexerでsyntax highlightされていたが、Jupyter book v0.8.0でMyST-NBのv0.10に対応し、Jupyter notebookのカーネルの言語に応じてcode cellがsyntax highlightされるようになった。
-  [Change Log - Jupyter book v0.8.0 2020-09-01](https://jupyterbook.org/reference/_changelog.html#v0-8-0-2020-09-01)

Syntax highlightの色などはカスタムCSSの追加して調整している (Qiitaの配色が見やすいので参考にした)。具体的な調整については<https://compneuro-julia.github.io/_static/custom.css>および次項を参照。

## カスタムCSSの追加
本の構成が

```
mybook/
├── _config.yml
├── _toc.yml
└── page1.md
```

のようであった場合、

```
mybook/
├── _config.yml
├── _toc.yml
├── page1.md
└── _static
    └── myfile.css
```

のように `_static`ディレクトリを作成し、その下にCSSやJavascriptなどのファイルを置いておけばbuildするときに自動で読み込まれる。` _static`ディレクトリには他に画像ファイルやpdfファイルなどを置くことも可能である。