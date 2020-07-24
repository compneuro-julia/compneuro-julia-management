# JuliaのTips集
このページはJuliaでの実装におけるTips (詰まったところの解決策)をまとめたものである。体系的にまとまってはいない。

## 関数名の!記号
単なる**慣習**として関数への入力を変更する場合に!を付ける。

関数内で配列を変更する場合には注意が必要である。以下に入力された配列を同じサイズの要素1の配列で置き換える、ということを目的として書かれた2つの関数がある。違いは`v`の後に`[:]`としているかどうかである。

function wrong!(A::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

実行すると`wrong!`の場合には入力された配列が変更されていないことがわかる (なのでこの場合には!は付けるべきではない)。

using Random
v = rand(2, 2)
print("v : ", v)

wrong!(v)
print("\nwrong : ", v)

right!(v)
print("\nright : ", v)

## 配列の1次元化
配列を一次元化(flatten)する方法。まずは3次元配列を作成する。

B = rand(2, 2, 2)

用意されている`flatten`を素直に用いると次のようになる。

import Base.Iterators: flatten
collect(flatten(B))

単に`B[:]`とするだけでもよい。

B[:]

## 行列の正規化

C = rand(3,3)

D = C ./ sum(C, dims=1)

print(sum(D, dims=1))