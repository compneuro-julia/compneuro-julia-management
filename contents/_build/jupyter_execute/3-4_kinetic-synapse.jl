# 3.4 ���͊w���f��
## 3.4.1 �`���l�����Ԃ̓��͊w�I�\��
�w���֐��^�V�i�v�X�ƃ��f���̐U�镑���͂قړ��ꂾ��, ���̍\���������قȂ郂�f���Ƃ���**���͊w���f��** (Kinetic model, �܂���Markov kinetic model)������ ([Destexhe et al., 1994](https://link.springer.com/article/10.1007/BF00961734); [Destexhe et al., 2002](http://cns.iaf.cnrs-gif.fr/files/handbook98.pdf))�B���͊w���f����HH���f���̃Q�[�g�ϐ��̎��Ɨގ��������ŕ\�����B���̃��f���ł̓`���l�����J�������(Open)�ƕ������(Close), ����ѐ_�o�`�B����(neurotransmitter)�̕��o���(T)��2�̗v�f�Ɋւ����Ԃ�����B�܂�, ��$\to$�J�̔������x��$\alpha$, �J$\to$�̔������x��$\beta$�Ƃ���B���̂Ƃ��A������\����ԑJ�ڂ̎��͎��̂悤�ɂȂ�B

$$
\begin{equation}
\text{Close}+\text{T}  \underset{\beta}{\overset{\alpha}{\rightleftharpoons}}\text{Open}    
\end{equation}
$$

������, �V�i�v�X���Ԃ�$r$�Ƃ����

$$
\begin{equation}
\frac{dr}{dt}=\alpha T (1-r) - \beta r
\end{equation}
$$

�ƂȂ�B������, T�̓V�i�v�X�O�זE�����΂����Ƃ��ɃC���p���X�I��1������������Ƃ���B�܂�, $\alpha, \beta$�͑��x�Ȃ̂�, ���萔�̋t���ł��邱�Ƃɒ��ӂ��悤�B $\alpha=2000 \text{ms}^{-1}$, $\beta=200 \text{ms}^{-1}$�Ƃ����, �V�i�v�X���Ԃ͎��̂悤�ɂȂ�B

using PyPlot

dt = 1e-4 # �^�C���X�e�b�v (sec)
�� = 1/5e-4; �� = 1/5e-3
T = 0.05 # �V�~�����[�V�������� (sec)
nt = Int(T/dt) # �V�~�����[�V�����̑��X�e�b�v

r = zeros(nt)

for t in 1:nt-1
    spike = ifelse(t == 1, 1, 0)
    r[t+1] = r[t] + dt * (��*spike*(1-r[t]) - ��*r[t])
end

time = (1:nt)*dt
figure(figsize=(4, 3))
plot(time, r)
xlabel("Time (s)"); ylabel("Post-synaptic current (pA)")
tight_layout()

## 3.4.2 Hodgkin-Huxley���f���ɂ�����V�i�v�X���f��
����܂Ŗ����I�ɃX�p�C�N�̔������\�����ꂽ���f����p���Ă������AHH���f���ł͒P�Ȃ閌�d�ʂ̕ϐ�������݂̂ł���B�����ł͑O�q�������͊w�I���f����p����HH���f���ɂ�����V�i�v�X���Ԃ̋L�q���s�� ([Destexhe et al., 1994](https://www.mitpressjournals.org/doi/10.1162/neco.1994.6.1.14); [Batista et al., 2014](https://www.sciencedirect.com/science/article/pii/S0378437114004592))�B

$r_{j}$��$j$�Ԗڂ̃j���[������pre-synaptic dynamics�Ƃ���ƁA$r_{j}$�͎����ɏ]���B

$$
\frac{\mathrm{d} r_{j}}{\mathrm{d} t}=\left(\frac{1}{\tau_{r}}-\frac{1}{\tau_{d}}\right) \frac{1-r_{j}}{1+\exp \left(-V_{j}+V_{0}\right)}-\frac{r_{j}}{\tau_{d}}
$$

�������A���萔 $\tau_r=0.5, \tau_d = 8$ (ms), ���]�d�� $V_0 = -20$ (mV)�Ƃ���B�O�߂Ŋ���$r$�̕`��͍s�������A�p���X�g����������ꍇ�̋������m�F����B

using Base: @kwdef
using Parameters: @unpack # or using UnPack

@kwdef struct HHParameter{FT}
    Cm::FT = 1.0 # ���e��(uF/cm^2)
    gNa::FT = 120.0 # Na+ �̍ő�R���_�N�^���X(mS/cm^2)
    gK::FT = 36.0 # K+ �̍ő�R���_�N�^���X(mS/cm^2)
    gL::FT = 0.3 # �R��C�I���̍ő�R���_�N�^���X(mS/cm^2)
    ENa::FT = 50.0 # Na+ �̕��t�d��(mV)
    EK::FT = -77.0 # K+ �̕��t�d��(mV)
    EL::FT = -54.387 #�R��C�I���̕��t�d��(mV)
    tr::FT = 0.5 # ms
    td::FT = 8.0 # ms
    invtr::FT = 1.0 / tr
    invtd::FT = 1.0 / td
    v0::FT = -20.0 # mV
end

@kwdef mutable struct HH{FT}
    param::HHParameter = HHParameter{FT}()
    N::Int32
    v::Vector{FT} = fill(-65.0, N)
    m::Vector{FT} = fill(0.05, N)
    h::Vector{FT} = fill(0.6, N)
    n::Vector{FT} = fill(0.32, N)
    r::Vector{FT} = zeros(N)
end

function updateHH!(variable::HH, param::HHParameter, I::Vector, dt)
    @unpack N, v, m, h, n, r = variable
    @unpack Cm, gNa, gK, gL, ENa, EK, EL, tr, td, invtr, invtd, v0= param
    @inbounds for i = 1:N
        m[i] += dt * ((0.1(v[i]+40.0)/(1.0 - exp(-0.1(v[i]+40.0))))*(1.0 - m[i]) - 4.0exp(-(v[i]+65.0) / 18.0)*m[i])
        h[i] += dt * ((0.07exp(-0.05(v[i]+65.0)))*(1.0 - h[i]) - 1.0/(1.0 + exp(-0.1(v[i]+35.0)))*h[i])
        n[i] += dt * ((0.01(v[i]+55.0)/(1.0 - exp(-0.1(v[i]+55.0))))*(1.0 - n[i]) - (0.125exp(-0.0125(v[i]+65)))*n[i])
        v[i] += dt / Cm * (I[i] - gNa * m[i]^3 * h[i] * (v[i] - ENa) - gK * n[i]^4 * (v[i] - EK) - gL * (v[i] - EL))
        r[i] += dt * ((invtr - invtd) * (1.0 - r[i])/(1.0 + exp(-v[i] + v0)) - r[i] * invtd)
    end
end

�V�~�����[�V���������s����B

T = 50 # ms
dt = 0.01f0 # ms
nt = Int32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(5f0 * ((t .> 10) - (t .> 15)), 1, N)  # injection current

# �L�^�p
varr = zeros(Float32, nt, N)
rarr = zeros(Float32, nt, N)

# model�̒�`
neurons = HH{Float32}(N=N)

# simulation
@time for i = 1:nt
    updateHH!(neurons, neurons.param, I[i, :], dt)
    varr[i, :] = neurons.v
    rarr[i, :] = neurons.r
end

�`�悵�Ă݂�B

figure(figsize=(5,5))
subplot(3, 1, 1)
plot(t, varr[:, 1]); ylabel("Membrane\n potential (mV)")
subplot(3, 1, 2)
plot(t, rarr[:, 1]); ylabel("Pre-synaptic\n dynamics")
subplot(3, 1, 3)
plot(t, I[:, 1]); xlabel("Times (ms)"); ylabel("Injection\n current (nA)")
tight_layout()