# Julia��Tips�W

## �z���1������

using Random

a = rand(2, 2, 2)

import Base.Iterators: flatten
collect(flatten(a))

�P�Ɏ��̂悤�ɂ��邾���ł��悢�B

a[:]

## �s��̐��K��

A = rand(3,3)

B = A ./ sum(A, dims=1)

print(sum(B, dims=1))