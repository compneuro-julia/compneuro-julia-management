# Julia��Tips�W
���̃y�[�W��Julia�ł̎����ɂ�����Tips (�l�܂����Ƃ���̉�����)���܂Ƃ߂����̂ł���B�̌n�I�ɂ܂Ƃ܂��Ă͂��Ȃ��B

## 1. �֐�����!�L��
�P�Ȃ�**���K**�Ƃ��Ċ֐��ւ̓��͂�ύX����ꍇ��!��t����B

�֐����Ŕz���ύX����ꍇ�ɂ͒��ӂ��K�v�ł���B�ȉ��ɓ��͂��ꂽ�z��𓯂��T�C�Y�̗v�f1�̔z��Œu��������A�Ƃ������Ƃ�ړI�Ƃ��ď����ꂽ2�̊֐�������B�Ⴂ��`v`�̌��`[:]`�Ƃ��Ă��邩�ǂ����ł���B

function wrong!(A::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

���s�����`wrong!`�̏ꍇ�ɂ͓��͂��ꂽ�z�񂪕ύX����Ă��Ȃ����Ƃ��킩��B

using Random
v = rand(2, 2)
print("v : ", v)

wrong!(v)
print("\nwrong : ", v)

right!(v)
print("\nright : ", v)

## 2. �z���1������
�z����ꎟ����(flatten)������@�B�܂���3�����z����쐬����B

B = rand(2, 2, 2)

�p�ӂ���Ă���`flatten`��f���ɗp����Ǝ��̂悤�ɂȂ�B

import Base.Iterators: flatten
collect(flatten(B))

�P��`B[:]`�Ƃ��邾���ł��悢�B

B[:]

## 3. �s��̍s�E�񂲂Ƃ̐��K��
�V�~�����[�V�����ɂ����ăj���[�����Ԃ̏d�ݍs����s���邢�͗񂲂Ƃɐ��K�� (weight normalization)����ꍇ������B����͊e�j���[�����ւ̓��͂̑傫���𓯂��ɂ��铭����d�݂̔��U��h������������B�ȉ��ł͍s���Ƃ̘a��1�ɂ���B

W = rand(3,3)

Wnormed = W ./ sum(W, dims=1)

print(sum(Wnormed, dims=1))

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