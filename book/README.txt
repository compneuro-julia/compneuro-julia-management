build.batを実行する

ipynbをtexに変換
bib file copyする
biber main
main.texをコンパイルする


-----
md. -> .texの注意．
・数式はmdでは$$~$$で挟む

fig, ax = subplots(figsize=())
fig, axes = subplots(figsize=())

fig, axes = subplots(3, 1, figsize=(5,3.5), sharex="all", height_ratios=[2, 2, 1])
fig.align_labels()
fig.tight_layout()

\begin{equation}
\end{equation}