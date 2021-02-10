# JuliaのTips集
このページはJuliaでの実装におけるTips (詰まったところの解決策)をまとめたものである．体系的にまとまってはいない．

## 1. 関数名の!記号
単なる**慣習**として関数への入力を変更する場合に!を付ける．

関数内で配列を変更する場合には注意が必要である．以下に入力された配列を同じサイズの要素1の配列で置き換える，ということを目的として書かれた2つの関数がある．違いは`v`の後に`[:]`としているかどうかである．

function wrong!(a::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

実行すると`wrong!`の場合には入力された配列が変更されていないことがわかる．

using Random
v = rand(2, 2)
println("v : ", v)

wrong!(v)
println("wrong : ", v)

right!(v)
println("right : ", v)

## 2. 配列の1次元化
配列を一次元化(flatten)する方法．まずは3次元配列を作成する．

B = rand(2, 2, 2)

用意されている`flatten`を素直に用いると次のようになる．

import Base.Iterators: flatten
collect(flatten(B))

ただし，単に`B[:]`とするだけでもよい．

B[:]

## 3. 行列の行・列ごとの正規化
シミュレーションにおいてニューロン間の重み行列を行あるいは列ごとに正規化 (weight normalization)する場合がある．これは各ニューロンへの入力の大きさを同じにする働きや重みの発散を防ぐ役割がある．以下では行ごとの和を1にする．

W = rand(3,3)

Wnormed = W ./ sum(W, dims=1)

println(sum(Wnormed, dims=1))

## 4. 行列の結合 (concatenate)
行列の結合はMATLABに近い形式で行うことができる．まず，2つの行列A, Bを用意する．

A = [1 2; 3 4]

B = [4 5 6; 7 8 9]

### 4.1 水平結合 (Horizontal concatenation)
`hcat`を使うやり方と，`[ ]`を使うやり方がある．

H1 = hcat(A,B)

H2 = [A B]

なお，MATLABのように次のようにすると正しく結合はされない．

H3 = [A, B]

### 4.2 垂直結合 (Vertical concatenation)

V1 = vcat(A, B')

V2 = [A; B']

[V2 [A;B']]

## 5. 配列に新しい軸を追加
要はnumpyでの`A[None, :]`や`A[np.newaxis, :]`のようなことがしたい場合．やや面倒だが，`reshape`を使うか，`[CartesianIndex()]`を用いる．

v = rand(3)

newaxis = [CartesianIndex()]
v1 = v[newaxis, :]

## 6. Array{Array{Float64, x},1}をArray{Float64, x+1}に変換
numpyでは`array([matrix for i in range()])`などを用いると，1次元配列のリストを2次元配列に変換できた．Juliaでも同様にする場合は`hcat(...)`や`cat(...)`を用いる．

A1 = [i*rand(3) for i=1:5]

println("Type : ", typeof(A1))
println("Size : ", size(A1))

A2 = hcat(A1...)'

println("Type : ", typeof(A2))
println("Size : ", size(A2))

以下は多次元配列の場合．`cat(...)`で配列を結合し，`permitedims`で転置する．

B1 = [i*rand(3, 4, 5) for i=1:6]

println("Type : ", typeof(B1))
println("Size : ", size(B1))

B2 = permutedims(cat(B1..., dims=4), [4, 1, 2, 3])

println("Type : ", typeof(B2))
println("Size : ", size(B2))

## 7. 二項分布 (bernoulli distribution)のサンプリング

p = 0.7
N = 100

using BenchmarkTools

@benchmark floor.(p .+ rand(N))

println(sum(floor.(p .+ rand(N)) .== 1.0) / N) 

@benchmark 1.0f0 * (p .≥ rand(N))

println(sum(1.0f0 * (p .≥ rand(N)) .== 1.0) / N) 

## 8. Roth's column lemma

Roth's column lemmaは，例えば，$A, B, C$が与えられていて，$X$を未知とするときの方程式 $AXB = C$を考えると，この方程式は

$$
(B^\top \otimes A)\text{vec}(X) = \text{vec}(AXB)=\text{vec}(C)
$$

の形に書き下すことができる，というものである．$\text{vec}(\cdot)$はvec作用素（行列を列ベクトル化する作用素）である．`vec(X) = vcat(X...)`で実現できる．Roth's column lemmaを用いれば，$AXB = C$の解は

$$
X = \text{vec}^{-1}\left((B^\top \otimes A)^{-1}\text{vec}(C)\right)
$$

として得られる．ただし，$\text{vec}(\cdot)^{-1}$は列ベクトルを行列に戻す作用素(inverse of the vectorization operator)である．`reshape()`で実現できる．2つの作用素をまとめると，

$$
\begin{align}
\text{vec} &: R^{m\times n}\to R^{mn}\\
\text{vec}^{−1} &: R^{mn}\to R^{m×n}
\end{align}
$$

であり，$\text{vec}^{−1}\left(\text{vec}(X)\right)=X\ (\text{for all}\ X\in R^{m\times n})，\text{vec}\left(\text{vec}^{−1}(x)\right)=x\ (\text{for all}\ x \in R^{mn})$となる．

using LinearAlgebra, Kronecker, Random

m = 4
A = randn(m, m)
B = randn(m, m)
C = convert(Array{Float64}, reshape(1:16, (m, m)))

vec(X) = vcat(X...)

X = reshape((B' ⊗ A)^-1 * vec(C), (m, m))

A * X * B