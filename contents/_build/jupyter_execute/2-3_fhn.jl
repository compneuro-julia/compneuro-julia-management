# 2.3 FitzHugh-Nagumo���f��

## 2.3.1 FitzHugh-Nagumo���f���̒�`

$$
\begin{align*} \frac{dv}{dt} &= c\left(v-\frac{v^3}{3}-u+I\right)\\ 
\frac{du}{dt} &= v-bu+a \end{align*}
$$

������$a,b,c$�͒萔�ł���A$a=0.7, b=0.8, c=10$���悭�g����B$v$�͖��d�ʂŁA$u$�͉񕜕ϐ�(recovery variable)�ł���B $I$�͊O���h���d���ɑΉ�����B

�܂��K�v�ȃp�b�P�[�W��ǂݍ��ށB

using Base: @kwdef
using Parameters: @unpack # or using UnPack

�ύX���Ȃ��萔��ێ����� `struct` �� `FHNParameter` ��, �ϐ���ێ����� `mutable struct` �� `FHN` ���쐬����B

@kwdef struct FHNParameter{FT}
    a::FT = 0.7
    b::FT = 0.8
    c::FT = 10.0
end

@kwdef mutable struct FHN{FT}
    param::FHNParameter = FHNParameter{FT}()
    N::UInt16
    v::Vector{FT} = fill(-1.0, N)
    u::Vector{FT} = zeros(N)
end

���ɕϐ����X�V����֐�`update!`�������B�\���o�[�Ƃ��Ă͗z�IEuler�@�܂���4����Runge-Kutta�@��p����B�ȉ��ł�Euler�@��p���Ă���BJulia�ł�for���[�v��p����1�̃j���[�������ƂɃp�����[�^���X�V��������x�N�g����p������������ł���B

function update!(variable::FHN, param::FHNParameter, I::Vector, dt)
    @unpack N, v, u = variable
    @unpack a, b, c = param
    @inbounds for i = 1:N
        v[i] += dt * c * (-u[i] + v[i] - v[i]^3 / 3 + I[i])
        u[i] += dt * (v[i] - b*u[i] + a)
    end
end

## 2.3.2 FitzHugh-Nagumo���f���̃V�~�����[�V�����̎��s
�������̒萔��ݒ肵�ăV�~�����[�V���������s����B

T = 50 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(0.35f0*ones(nt), 1, N)  # injection current

# �L�^�p
varr = zeros(Float32, nt, N)
uarr = zeros(Float32, nt, N)
gatearr = zeros(Float32, nt, 3, N)

# model�̒�`
neurons = FHN{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[i, :], dt)
    varr[i, :] = neurons.v
    uarr[i, :] = neurons.u
end

���ʂ�`�悷��B

using PyPlot

subplot(2, 1, 1)
plot(t, varr[:, 1], label=false, color="black"); ylabel("v")
subplot(2, 1, 2)
plot(t, uarr[:, 1], label=false); ylabel("u"); xlabel("Times (ms)")

���Ή񐔂����߂�B

spike = (varr[1:nt-1, :] .< 0) .& (varr[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
println("Num. of spikes : ", num_spikes[1])

## 2.3.3 ���}�̕`��

margin = 1.0
vmax, vmin = maximum(varr) + margin, minimum(varr) - margin
umax, umin = maximum(uarr) + margin, minimum(uarr) - margin
 
vrange = vmin:0.1:vmax
urange = umin:0.1:umax
U = [i for i in urange, j in 1:length(vrange)]
V = [j for i in 1:length(urange), j in vrange]

a = 0.7; b = 0.8; c = 10.0; I = 0.34
dV = c * (-U + V - V .^3 / 3 .+ I)
dU = V - b*U .+ a

figure(figsize=(4,3))
streamplot(V, U, dV, dU, density=[0.8, 0.8], linewidth=2) 
contour(V, U, dU, levels=[0])
contour(V, U, dV, levels=[0])
plot(varr, uarr); xlim(vmin, vmax); ylim(umin, umax)
tight_layout()