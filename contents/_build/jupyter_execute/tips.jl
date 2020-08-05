# JuliaのTips集
このページはJuliaでの実装におけるTips (詰まったところの解決策)をまとめたものである。体系的にまとまってはいない。

## 1. 関数名の!記号
単なる**慣習**として関数への入力を変更する場合に!を付ける。

関数内で配列を変更する場合には注意が必要である。以下に入力された配列を同じサイズの要素1の配列で置き換える、ということを目的として書かれた2つの関数がある。違いは`v`の後に`[:]`としているかどうかである。

function wrong!(A::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

実行すると`wrong!`の場合には入力された配列が変更されていないことがわかる。

using Random
v = rand(2, 2)
print("v : ", v)

wrong!(v)
print("\nwrong : ", v)

right!(v)
print("\nright : ", v)

## 2. 配列の1次元化
配列を一次元化(flatten)する方法。まずは3次元配列を作成する。

B = rand(2, 2, 2)

用意されている`flatten`を素直に用いると次のようになる。

import Base.Iterators: flatten
collect(flatten(B))

単に`B[:]`とするだけでもよい。

B[:]

## 3. 行列の行・列ごとの正規化
シミュレーションにおいてニューロン間の重み行列を行あるいは列ごとに正規化 (weight normalization)する場合がある。これは各ニューロンへの入力の大きさを同じにする働きや重みの発散を防ぐ役割がある。以下では行ごとの和を1にする。

W = rand(3,3)

Wnormed = W ./ sum(W, dims=1)

print(sum(Wnormed, dims=1))

## 4. 行列の結合 (concatenate)
行列の結合はMATLABに近い形式で行うことができる。まず、2つの行列A, Bを用意する。

A = [1 2; 3 4]

B = [4 5 6; 7 8 9]

### 4.1 水平結合 (Horizontal concatenation)
`hcat`を使うやり方と、`[ ]`を使うやり方がある。

H1 = hcat(A,B)

H2 = [A B]

なお、MATLABのように次のようにすると正しく結合はされない。

H3 = [A, B]

### 4.2 垂直結合 (Vertical concatenation)

V1 = vcat(A, B')

V2 = [A; B']

[V2 [A;B']]