# JuliaのTips集

## 配列の1次元化

using Random

a = rand(2, 2, 2)

import Base.Iterators: flatten
collect(flatten(a))

単に次のようにするだけでもよい。

a[:]

## 行列の正規化

A = rand(3,3)

B = A ./ sum(A, dims=1)

print(sum(B, dims=1))