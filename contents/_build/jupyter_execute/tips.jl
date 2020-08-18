# Julia��Tips�W
���̃y�[�W��Julia�ł̎����ɂ�����Tips (�l�܂����Ƃ���̉�����)���܂Ƃ߂����̂ł���B�̌n�I�ɂ܂Ƃ܂��Ă͂��Ȃ��B

## 1. �֐�����!�L��
�P�Ȃ�**���K**�Ƃ��Ċ֐��ւ̓��͂�ύX����ꍇ��!��t����B

�֐����Ŕz���ύX����ꍇ�ɂ͒��ӂ��K�v�ł���B�ȉ��ɓ��͂��ꂽ�z��𓯂��T�C�Y�̗v�f1�̔z��Œu��������A�Ƃ������Ƃ�ړI�Ƃ��ď����ꂽ2�̊֐�������B�Ⴂ��`v`�̌��`[:]`�Ƃ��Ă��邩�ǂ����ł���B

function wrong!(a::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

���s�����`wrong!`�̏ꍇ�ɂ͓��͂��ꂽ�z�񂪕ύX����Ă��Ȃ����Ƃ��킩��B

using Random
v = rand(2, 2)
println("v : ", v)

wrong!(v)
println("wrong : ", v)

right!(v)
println("right : ", v)

## 2. �z���1������
�z����ꎟ����(flatten)������@�B�܂���3�����z����쐬����B

B = rand(2, 2, 2)

�p�ӂ���Ă���`flatten`��f���ɗp����Ǝ��̂悤�ɂȂ�B

import Base.Iterators: flatten
collect(flatten(B))

�������A�P��`B[:]`�Ƃ��邾���ł��悢�B

B[:]

## 3. �s��̍s�E�񂲂Ƃ̐��K��
�V�~�����[�V�����ɂ����ăj���[�����Ԃ̏d�ݍs����s���邢�͗񂲂Ƃɐ��K�� (weight normalization)����ꍇ������B����͊e�j���[�����ւ̓��͂̑傫���𓯂��ɂ��铭����d�݂̔��U��h������������B�ȉ��ł͍s���Ƃ̘a��1�ɂ���B

W = rand(3,3)

Wnormed = W ./ sum(W, dims=1)

println(sum(Wnormed, dims=1))

## 4. �s��̌��� (concatenate)
�s��̌�����MATLAB�ɋ߂��`���ōs�����Ƃ��ł���B�܂��A2�̍s��A, B��p�ӂ���B

A = [1 2; 3 4]

B = [4 5 6; 7 8 9]

### 4.1 �������� (Horizontal concatenation)
`hcat`���g�������ƁA`[ ]`���g������������B

H1 = hcat(A,B)

H2 = [A B]

�Ȃ��AMATLAB�̂悤�Ɏ��̂悤�ɂ���Ɛ����������͂���Ȃ��B

H3 = [A, B]

### 4.2 �������� (Vertical concatenation)

V1 = vcat(A, B')

V2 = [A; B']

[V2 [A;B']]

## 5. �z��ɐV��������ǉ�
�v��numpy�ł�`A[None, :]`��`A[np.newaxis, :]`�̂悤�Ȃ��Ƃ��������ꍇ�B���ʓ|�����A`reshape`���g�����A`[CartesianIndex()]`��p����B

v = rand(3)

newaxis = [CartesianIndex()]
v1 = v[newaxis, :]

## 6. Array{Array{Float64, x},1}��Array{Float64, x+1}�ɕϊ�
numpy�ł�`array([matrix for i in range()])`�Ȃǂ�p����ƁA1�����z��̃��X�g��2�����z��ɕϊ��ł����BJulia�ł����l�ɂ���ꍇ��`hcat(...)`��`cat(...)`��p����B

A1 = [i*rand(3) for i=1:5]

println("Type : ", typeof(A1))
println("Size : ", size(A1))

A2 = hcat(A1...)'

println("Type : ", typeof(A2))
println("Size : ", size(A2))

�ȉ��͑������z��̏ꍇ�B`cat(...)`�Ŕz����������A`permitedims`�œ]�u����B

B1 = [i*rand(3, 4, 5) for i=1:6]

println("Type : ", typeof(B1))
println("Size : ", size(B1))

B2 = permutedims(cat(B1..., dims=4), [4, 1, 2, 3])

println("Type : ", typeof(B2))
println("Size : ", size(B2))