# Julia��Tips�W
���̃y�[�W��Julia�ł̎����ɂ�����Tips (�l�܂����Ƃ���̉�����)���܂Ƃ߂����̂ł���B�̌n�I�ɂ܂Ƃ܂��Ă͂��Ȃ��B

## �֐�����!�L��
�P�Ȃ�**���K**�Ƃ��Ċ֐��ւ̓��͂�ύX����ꍇ��!��t����B

�֐����Ŕz���ύX����ꍇ�ɂ͒��ӂ��K�v�ł���B�ȉ��ɓ��͂��ꂽ�z��𓯂��T�C�Y�̗v�f1�̔z��Œu��������A�Ƃ������Ƃ�ړI�Ƃ��ď����ꂽ2�̊֐�������B�Ⴂ��`v`�̌��`[:]`�Ƃ��Ă��邩�ǂ����ł���B

function wrong!(A::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

���s�����`wrong!`�̏ꍇ�ɂ͓��͂��ꂽ�z�񂪕ύX����Ă��Ȃ����Ƃ��킩�� (�Ȃ̂ł��̏ꍇ�ɂ�!�͕t����ׂ��ł͂Ȃ�)�B

using Random
v = rand(2, 2)
print("v : ", v)

wrong!(v)
print("\nwrong : ", v)

right!(v)
print("\nright : ", v)

## �z���1������
�z����ꎟ����(flatten)������@�B�܂���3�����z����쐬����B

B = rand(2, 2, 2)

�p�ӂ���Ă���`flatten`��f���ɗp����Ǝ��̂悤�ɂȂ�B

import Base.Iterators: flatten
collect(flatten(B))

�P��`B[:]`�Ƃ��邾���ł��悢�B

B[:]

## �s��̐��K��

C = rand(3,3)

D = C ./ sum(C, dims=1)

print(sum(D, dims=1))