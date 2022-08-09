using LinearAlgebra

a = [1, 2, 3]

A = []

W = rand(3,3)

Wnormed = W ./ sum(W, dims=1)

println(sum(Wnormed, dims=1))

A = [1 2; 3 4]

B = [4 5 6; 7 8 9]

H1 = hcat(A,B)

H2 = [A B]

H3 = [A, B]

V1 = vcat(A, B')

V2 = [A; B']

[V2 [A;B']]

v = rand(3)

newaxis = [CartesianIndex()]
v1 = v[newaxis, :]

I

I(3)

A = rand(2,2)
b = rand(2)

x = inv(A) * b

x = A \ b

using LinearAlgebra, Kronecker, Random

m = 4
A = randn(m, m)
B = randn(m, m)
C = convert(Array{Float64}, reshape(1:16, (m, m)))

X = reshape((B' ⊗ A) \ vec(C), (m, m))

A * X * B

B = rand(2, 2, 2)

import Base.Iterators: flatten
collect(flatten(B))

B[:]

a = rand(2,3,5)
b = reshape(a, (:, 5))

A1 = [i*rand(3) for i=1:5]

println("Type : ", typeof(A1))
println("Size : ", size(A1))

A2 = hcat(A1...)'

println("Type : ", typeof(A2))
println("Size : ", size(A2))

B1 = [i*rand(3, 4, 5) for i=1:6]

println("Type : ", typeof(B1))
println("Size : ", size(B1))

B2 = permutedims(cat(B1..., dims=4), [4, 1, 2, 3])

println("Type : ", typeof(B2))
println("Size : ", size(B2))
