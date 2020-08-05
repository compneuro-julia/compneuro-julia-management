# 2.1 Hodgkin-Huxley���f��

## 2.1.1 Hodgkin-Huxley���f���ɂ����閌�̓�����H���f��
**Hodgkin-Huxley���f��** (HH ���f��)��, A.L. Hodgkin��A.F. Huxley�ɂ����1952�N�ɍl�Ă��ꂽ�j���[�����̖�������\�����f���ł��� ([Hodgkin & Huxley, 1952](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/))�BHodgkin��̓����C�J�̋���_�o�����ɑ΂���**�d�ʌŒ�@**(voltage-clamp)��p�����������s��, �������瓾��ꂽ�ϑ����ʂ����Ƀ��f�����\�z�����B

HH���f���ɂ͓����ȓd�C��H���f��������, **���̕��񓙉���H���f��** (parallel conductance model)�ƌĂ΂�Ă���B���̕��񓙉���H���f���ł�, �j���[�����̍זE�����R���f���T, �זE���ɖ��܂��Ă���C�I���`���l�����ϒ�R (���I�ɕω������R) �Ƃ��Ēu��������B

**�C�I���`���l��** (ion channel)�͓���̃C�I��(�Ⴆ�΃i�g���E���C�I����J���E���C�I���Ȃ�)��I��I�ɒʂ����A���̂̈��ł���B���ꂼ��̃C�I���̎�ނɂ�����, �قȂ�C�I���`���l�������� (�����C�I���ł������̎�ނ̃C�I���`���l��������)�B�܂�, �C�I���`���l���ɂ̓C�I���̎�ނɉ����ĈقȂ�**�R���_�N�^���X**(��R�̋t���œd���́u����₷���v���Ӗ�����)��**���t�d��**(equilibrium potential)������BHH���f���ł�, �i�g���E��(Na$^{+}$)�`���l��, �J���E��(K$^{+}$)�`���l��, �R��d��(leak current)�̃C�I���`���l�������肷��B�R��d���̃C�I���`���l���͓�������ł��Ȃ������`���l����, ������d�����R��o���`���l�����Ӗ�����B�Ȃ�, ���݂ł͘R��d���̑�����Cl$^{-}$�C�I��(chloride ion)�ɂ�邱�Ƃ��������Ă���B

```{figure} ./_static/images/chapter2/parallel_conductance_model.JPG
---
width: 300px
name: parallel_conductance_model
---
Hodgkin-Huxley���f���̖��̓�����H���f��
```

����ł�, ������H���f����p���ēd�ʕω��̎��𗧂ĂĂ݂悤�B��}�ɂ�����, $C_m$�͍זE���̃L���p�V�^���X(���e��), $I_{m}(t)$�͍זE���𗬂��d��(�O������̓��͓d��), $I_\text{Cap}(t)$�͖��̃R���f���T�𗬂��d��, $I_\text{Na}(t)$�y�� $I_K(t)$�͂��ꂼ��i�g���E���`���l���ƃJ���E���`���l����ʂ��Ė����痬�o����d��, $I_\text{L}(t)$�͘R��d���ł���B���̂Ƃ�, 

$$
I_{m}(t)=I_\text{Cap}(t)+I_\text{Na}(t)+I_\text{K}(t)+I_\text{L}(t)    
$$

�Ƃ�����������Ă���B

���d�ʂ�$V(t)$�Ƃ����, Kirchhoff�̑��@�� (Kirchhoff's Voltage Law)���, 

$$
\underbrace{C_m\frac {dV(t)}{dt}}_{I_\text{Cap} (t)}=I_{m}(t)-I_\text{Na}(t)-I_\text{K}(t)-I_\text{L}(t)
$$

�ƂȂ�BHodgkin��̓`���l���d��$I_\text{Na}, I_K, I_\text{L}$���]�����������I�ɋ��߂��B

$$
\begin{aligned}
I_\text{Na}(t) &= g_{\text{Na}}\cdot m^{3}h(V-E_{\text{Na}})\\
I_\text{K}(t) &= g_{\text{K}}\cdot n^{4}(V-E_{\text{K}})\\
I_\text{L}(t) &= g_{\text{L}}(V-E_{\text{L}})
\end{aligned}
$$

������, $g_{\text{Na}}, g_{\text{K}}$�͂��ꂼ��Na$^+$, K$^+$�̍ő�R���_�N�^���X�ł���B$g_{\text{L}}$�̓I�[���̖@���ɏ]���R���_�N�^���X��, L�R���_�N�^���X�͎��ԓI�ɕω��͂��Ȃ��Ɖ��肷��B�܂�, $m$��Na$^+$�R���_�N�^���X�̊������p�����[�^, $h$��Na$^+$�R���_�N�^���X�̕s�������p�����[�^, $n$��K$^+$�R���_�N�^���X�̊������p�����[�^�ł���, �Q�[�g�̊J�m����\���Ă���B�����, HH���f���̏�Ԃ�$V, m, h, n$��4�ϐ��ŕ\�����B�����̕ϐ��͈ȉ���$x$��$m, n, h$�ɒu��������3�̔����������ɏ]���B 

$$
\frac{dx}{dt}=\alpha_{x}(V)(1-x)-\beta_{x}(V)x
$$

������, $V$�̊֐��ł���$\alpha_{x}(V),\ \beta_{x}(V)$��$m, h, n$�ɂ���ĈقȂ�, ����6�̎��ɏ]���B

$$
\begin{array}{ll}
\alpha_{m}(V)=\dfrac {0.1(25-V)}{\exp \left[(25-V)/10\right]-1}, &\beta_{m}(V)=4\exp {(-V/18)}\\
\alpha_{h}(V)=0.07\exp {(-V/20)}, & \beta_{h}(V)={\dfrac{1}{\exp {\left[(30-V)/10 \right]}+1}}\\
\alpha_{n}(V)={\dfrac {0.01(10-V)}{\exp {\left[(10-V)/10\right]}-1}},& \beta_{n}(V)=0.125\exp {(-V/80)} 
\end{array}
$$

�Ȃ��A���̎���6.3���̏������ɂ����ăC�J�̋��厲���̊������瓾���f�[�^��p���ē����ꂽ���̂ł��邱�Ƃɒ��ӂ��悤�B

````{margin} 
```{note}
HH���f���̍\�z�Ɋւ�����j�ɂ��Ă�([Schwiening, 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3424716/))���Q�ƁB
```
````

## 2.1.2 Hodgkin-Huxley ���f���̒�`
����܂łɐ�����������p����HH���f������������B�܂��K�v�ȃp�b�P�[�W��ǂݍ��ށB

using Base: @kwdef
using Parameters: @unpack # or using UnPack

�ύX���Ȃ��萔��ێ����� `struct` �� `HHParameter` ��, �ϐ���ێ����� `mutable struct` �� `HH` ���쐬����B`v, m, h, n` ��HH model��4�ϐ�����, `r` ��pre-synaptic dynamics��\���ϐ��ł���B�ڍׂ�3�͂ŉ������B �萔�͎��̂悤�ɐݒ肷��B 

\begin{align*} 
C_m=1.0, g_{\text{Na}}=120, g_{\text{K}}=36, g_{\text{L}}=0.3\\
E_{\text{Na}}=50.0, E_{\text{K}}=-77, E_{\text{L}}=-54.387 
\end{align*}


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

���ɕϐ����X�V����֐�`updateHH!`�������B�\���o�[�Ƃ��Ă͗z�IEuler�@�܂���4����Runge-Kutta�@��p����B�ȉ��ł�Euler�@��p���Ă���BJulia�ł�for���[�v��p����1�̃j���[�������ƂɃp�����[�^���X�V��������x�N�g����p������������ł���B

function updateHH!(variable::HH, param::HHParameter, I::Vector, dt)
    @unpack N, v, m, h, n, r = variable
    @unpack Cm, gNa, gK, gL, ENa, EK, EL, tr, td, invtr, invtd, v0 = param
    @inbounds for i = 1:N
        m[i] += dt * ((0.1(v[i]+40.0)/(1.0 - exp(-0.1(v[i]+40.0))))*(1.0 - m[i]) - 4.0exp(-(v[i]+65.0) / 18.0)*m[i])
        h[i] += dt * ((0.07exp(-0.05(v[i]+65.0)))*(1.0 - h[i]) - 1.0/(1.0 + exp(-0.1(v[i]+35.0)))*h[i])
        n[i] += dt * ((0.01(v[i]+55.0)/(1.0 - exp(-0.1(v[i]+55.0))))*(1.0 - n[i]) - (0.125exp(-0.0125(v[i]+65)))*n[i])
        v[i] += dt / Cm * (I[i] - gNa * m[i]^3 * h[i] * (v[i] - ENa) - gK * n[i]^4 * (v[i] - EK) - gL * (v[i] - EL))
        r[i] += dt * ((invtr - invtd) * (1.0 - r[i])/(1.0 + exp(-v[i] + v0)) - r[i] * invtd)
    end
end

## 2.1.3 Hodgkin-Huxley���f���̃V�~�����[�V�����̎��s
�������̒萔��ݒ肵�ăV�~�����[�V���������s����B

T = 450 # ms
dt = 0.01f0 # ms
nt = Int32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(10f0 * ((t .> 50) - (t .> 200)) + 35f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# �L�^�p
varr = zeros(Float32, nt, N)
gatearr = zeros(Float32, nt, 3, N)

# model�̒�`
neurons = HH{Float32}(N=N)

# simulation
@time for i = 1:nt
    updateHH!(neurons, neurons.param, I[i, :], dt)
    varr[i, :] = neurons.v
    gatearr[i, 1, :] = neurons.m
    gatearr[i, 2, :] = neurons.h
    gatearr[i, 3, :] = neurons.n
end

���ʂ�\�����邽�߂� `Plots`��ǂݍ��ށB

using Plots

�j���[�����̖��d�� `v`, �Q�[�g�ϐ� `m, h, n`, �h���d�� `I`�̕`�������B���͓d���̒P�ʂ� $\mu\text{A/cm}^2$�ł���B

p1 = plot(t, varr[:, 1], label="", color="black")
p2 = plot(t, gatearr[:, :, 1], label = ["m" "h" "n"])
p3 = plot(t, I[:, 1], label="", color="black")
plot(p1, p2, p3, 
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["V (mV)" "Gating Value" "Current"],
    layout = grid(3, 1, heights=[0.4, 0.35, 0.25]), size=(600,500))

�����ŗp���邽�߂ɔ��Ή񐔂����߂�B`bitwise and`��p����Ɗy�ł���B

spike = (varr[1:nt-1, :] .< 0) .& (varr[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
println("Num. of spikes : ", num_spikes[1])

50ms����200ms�܂ł�11��, 250ms����400ms�܂ł�16�񔭉΂��Ă���̂Ŕ��Ή񐔂͌v27��ł���A���̌��ʂ͐������B

## 2.1.4 Frequency-current (F-I) curve
���̍��ł�Hodgkin-Huxley���f���ɂ�������͓d���ɑ΂��锭�Η��̕ω����ǂ̂悤�ɂȂ邩�𒲂ׂ�B���̃R�[�h�̂悤�ɓ��͓d�������X�ɑ����������Ƃ��̔��Η������Ă݂悤�B

T = 1000 # ms
dt = 0.01f0 # ms
nt = Int32(T/dt) # number of timesteps

N = 100 # �j���[�����̐�

# ���͎h��
maxcurrent = 30
t = Array{Float32}(1:nt)*dt
I = Array{Float32}(range(1,maxcurrent,length=N)) # injection current

# model�̒�`
neurons = HH{Float32}(N=N)

# �L�^�p
varr_fi = zeros(Float32, nt, N)

# simulation
for i = 1:nt
    updateHH!(neurons, neurons.param, I[:], dt)
    varr_fi[i, :] = neurons.v
end

���Η����v�Z���Č��ʂ�`�悷��B

spike = (varr_fi[1:nt-1, :] .< 0) .& (varr_fi[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
rate = num_spikes/T*1e3

plot(I[:], rate[1, :],
    xlabel="Input current",
    ylabel="Firing rate (Hz)", legend=false, size=(400,300))

���̂悤�ȋȐ���**frequency-current (F-I) curve** (�܂��� neuronal input/output (I/O) curve)�ƌĂԁB

## 2.1.5  �}���ナ�o�E���h (Postinhibitory rebound; PIR)
�j���[�����͓d�����������邱�ƂŖ��d�ʂ��ω���, ���d�ʂ��������臒l�𒴂���Ɣ��΂��N����, �Ƃ����̂̓j���[�����̊����d�ʔ����ɂ��Ă̓T�^�I�Ȑ����ł���B����ł�HH���f���̖��d��臒l�͂ǂ̂��炢�̒l�ɂȂ�̂��낤���B�����́u**���d��臒l�͈��ł͂Ȃ�**�v�ł���B������������ۂƂ��� **�}���ナ�o�E���h** (Postinhibitory rebound; PIR)������B���̎������锭�΂�**���o�E���h����** (rebound spikes) 
�ƌĂԁB�}���ナ�o�E���h�͉ߕ��ɐ��̓d���̈�����~�߂��ۂɖ��d�ʂ��Î~���d�ʂɉ񕜂���݂̂Ȃ炸, ����ɒE���ɂ����Ĕ��΂�����Ƃ������ۂł���B���̌��ۂ�������v���Ƃ���

1. **�A�m�[�_���u���C�N** (anodal break, �܂���anode break excitation; ABE)
2. �x��T�^�J���V�E���d�� (slow T-type calcium current)

������ ([Chik et al., 2004](https://pubmed.ncbi.nlm.nih.gov/15324089/))�BHH ���f���͂��̂����A�m�[�_���u���C�N���Č��ł��邽��, �V�~�����[�V�����ɂ��ǂ̂悤�Ȍ��ۂ��m�F���Ă݂悤�B����͓��͓d����ύX���邾���ōs����B

T = 450 # ms
dt = 0.01f0 # ms
nt = Int32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(10f0 * (-(t .> 50) + (t .> 200)) + 20f0 * (-(t .> 250) + (t .> 400)), 1, N)  # injection current

# model�̒�`
neurons = HH{Float32}(N=N)

# �L�^�p
varr2 = zeros(Float32, nt, N)
gatearr2 = zeros(Float32, nt, 3, N)

# simulation
@time for i = 1:nt
    updateHH!(neurons, neurons.param, I[i, :], dt)
    varr2[i, :] = neurons.v
    gatearr2[i, 1, :] = neurons.m
    gatearr2[i, 2, :] = neurons.h
    gatearr2[i, 3, :] = neurons.n
end

���ʂ͎��̂悤�ɂȂ�B

p1 = plot(t, varr2[:, 1], label="", color="black")
p2 = plot(t, gatearr2[:, :, 1], label = ["m" "h" "n"])
p3 = plot(t, I[:, 1], label="", color="black")
plot(p1, p2, p3, 
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["V (mV)" "Gating Value" "Injection\n current"],
    layout = grid(3, 1, heights=[0.4, 0.35, 0.25]), size=(600,500))

�Ȃ����̂悤�Ȃ��Ƃ��N���邩, �Ƃ����Ɖߕ��ɂ̏�Ԃ���Î~���d�ʂւƖ߂�ۂ�Na$^+$�`���l���������� (Na$^+$�`���l���̊������p�����[�^$m$��������, �s�������p�����[�^$h$������)��, ���d�ʂ��E���ɂ��邱�ƂōēxNa$^+$�`���l��������������, �Ƃ����|�W�e�B�u�t�B�[�h�o�b�N�ߒ�(**���ȍĐ��I�ߒ�**)�ɓ˓����邽�߂ł��� (�������, ���̉ߒ��͒ʏ�̊����d�ʔ����̃��J�j�Y���ł���)�B ���̍�, ���΂ɕK�v��臒l�����d�ʂ̒ቺ�ɉ����ĉ�������, �Ƃ������Ƃ��ł���B

���̂悤�ɖ��d��臒l�͈��ł͂Ȃ��B������, ���̌�̐߂ŏЉ�郂�f���͊ȗ����̂��߂�if����p��, ���d��臒l�𒴂������甭��, �Ƃ������̂�����B���ۂɂ͈Ⴄ�Ƃ������Ƃ𓪂̕Ћ��Ɏc���Ȃ���ǂݐi�߂邱�Ƃ𐄏�����B

```{Note}
PIR�Ɋ֘A���錻�ۂƂ��ė}���㑣�� (Postinhibitory facilitation; PIF)������B����͗}�����͂̌�ɋ������͂�������̎��ԓ��œ���Ɣ��΂��N����Ƃ������ۂł��� ([Dolda et al., 2006](http://www.brain.riken.jp/en/summer/prev/2006/files/j_rinzel04.pdf), [Dodla, 2014](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4614-7320-6_152-1))�B
```