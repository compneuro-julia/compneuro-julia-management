# 2.4 Leaky integrate-and-fire ���f��
## 2.4.1 LIF���f���̒�`
�����w�I�ȃC�I���`���l���̋����͍l������, ���͓d���𖌓d�ʂ�臒l�ɒB����܂Ŏ��ԓI�ɐϕ�����Ƃ������f����**Integrate-and-fire (IF, �ϕ�����)���f��** �Ƃ����B�����, IF���f���ɂ����Ė��d�ʂ̘R��(leak)[^leak]���l���������f���� **Leaky integrate-and-fire (LIF, �R��ϕ�����) ���f��** �ƌĂԁB�����ł�LIF���f���݂̂���舵���B

�j���[�����̖��d�ʂ�$V_m(t)$, �Î~���d�ʂ�$V_\text{rest}$, ���͓d��[^isyn]��$I(t)$, ����R��$R_m$, ���d�ʂ̎��萔��$\tau_m\ (=R_m \cdot C_m)$�Ƃ����, ���͎��̂悤�ɂȂ�[^vrest]�B

$$
\begin{equation}
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_mI(t)
\end{equation}
$$

������, $V_m$��臒l(threshold)[^theta]$V_{\text{th}}$�𒴂����, �E���ɂ��N����, ���d�ʂ̓s�[�N�d�� $V_{\text{peak}}$�܂ŏ㏸����B���Ό�͍ĕ��ɂ��N����, ���d�ʂ̓��Z�b�g�d�� $V_{\text{reset}}$�܂Œቺ����Ɖ��肷��[^reset]�B���Ό�, ���̊���$\tau_{\text{ref}}$ �̊Ԃ͖��d�ʂ��ω����Ȃ�[^ref], �Ƃ���B����� **�s����(refractory time period)** �ƌĂԁB

�ȏ�𓥂܂���LIF���f�����������Ă݂悤�B�܂��K�v�ȃp�b�P�[�W��ǂݍ��ށB

[^leak]: ���̘R��̓C�I���̊g�U�Ȃǂɂ����́B 
[^isyn]: �V�i�v�X���͂ɂ��d�����ǂ��Ȃ邩�́A��O�́u�V�i�v�X�`�B�̃��f���v�ň����B
[^vrest]: $(V_{m}(t)-V_\text{rest})$�̕����͖��d�ʂ̊��Î~���d�ʂƂ������Ƃɂ���, �P��$V_m(t)$�����̏ꍇ������B �܂�, �E�ӂ�$RI(t)$�̕����͒P��$I(t)$�Ƃ���邱�Ƃ�����B �����\�L����, ���̏ꍇ��$I(t)$�̓V�i�v�X�d���ɔ�Ⴗ���, �ƂȂ��Ă���(�P�ʂ�mV)�B 
[^theta]: th����n�܂�̂ŕ���$\theta$���g���邱�Ƃ�����B
[^reset]: ���Z�b�g�d�ʂ͐Î~���d�ʂƓ����ꍇ�������, �ߕ��ɂ��l�����ĐÎ~���d�ʂ���߂ɐݒ肷�邱�Ƃ�����B
[^ref]: �����ɂ���Ă͕s�����̊Ԃ͖��d�ʂ̕ω��͋��e���邪���΂͐����Ȃ��悤�ɂ��邱�Ƃ�����B

using Base: @kwdef
using Parameters: @unpack # or using UnPack

HH���f���Ɠ��l�ɕύX���Ȃ��萔��ێ����� `struct` �� `LIFParameter` ��, �ϐ���ێ����� `mutable struct` �� `LIF` ���쐬����B

@kwdef struct LIFParameter{FT}
    tref::FT   = 2 # �s���� (ms)
    tc_m::FT   = 10 # �����萔 (ms)
    vrest::FT  = -60 # �Î~���d�� (mV) 
    vreset::FT = -65 # ���Z�b�g�d�� (mV) 
    vthr::FT   = -40 # 臒l�d�� (mV)
    vpeak::FT  = 30 #�@�s�[�N�d�� (mV)
end

@kwdef mutable struct LIF{FT}
    param::LIFParameter = LIFParameter{FT}()
    N::UInt32 #�j���[�����̐�
    v::Vector{FT} = fill(-65.0, N) # ���d�� (mV)
    v_::Vector{FT} = fill(-65.0, N) # ���Γd�ʂ��L�^����ϐ�
    fire::Vector{Bool} = zeros(Bool, N) # ����
    tlast::Vector{FT} = zeros(N) # �Ō�̔��Ύ��� (ms)
    tcount::FT = 0 # ���ԃJ�E���g
end

���ɕϐ����X�V����֐�`update!`�������B

function update!(variable::LIF, param::LIFParameter, I::Vector, dt)
    @unpack N, v, v_, fire, tlast, tcount = variable
    @unpack tref, tc_m, vrest, vreset, vthr, vpeak = param
    
    @inbounds for i = 1:N
        #v[i] += dt * ((vrest - v[i] + I[i]) / tc_m) # �s�������l�����Ȃ��ꍇ�̍X�V��
        v[i] += dt * ((dt*tcount) > (tlast[i] + tref))*((vrest - v[i] + I[i]) / tc_m)
        #v[i] += dt * ifelse(dt*tcount[1] > tlast[i] + tref, (vrest - v[i] + I[i]) / tc_m, 0)
    end
    @inbounds for i = 1:N
        fire[i] = v[i] >= vthr
        v_[i] = ifelse(fire[i], vpeak, v[i]) #���Ύ��̓d�ʂ��܂߂ċL�^���邽�߂̕ϐ� (�����Ă��悢)
        v[i] = ifelse(fire[i], vreset, v[i])        
        tlast[i] = ifelse(fire[i], dt*tcount, tlast[i]) # ���Ύ����̍X�V
    end
end

�������̏����ɂ��ĉ�����Ă����B�܂��A��Ԗڂ�for���[�v����`v[i]`��`((dt*tcount) > (tlast[i] + tref))`�͍Ō�Ƀj���[���������΂�������`tlast[i]`�ɕs����`tref`�𑫂��������������݂̎���`dt*tcount[1]`���傫����Ζ��d�ʂ̍X�V�������A��������΍X�V���Ȃ��B��Ԗڂ�for���[�v�ɂ�����`fire[i]`�̓j���[�����̖��d�ʂ�臒l�d��`vthr`�𒴂�����`True`�ƂȂ�B`v[i]`�Ȃǂ̍X�V���ɂ���`ifelse(a, b, c)`��a��`True`�̎���b��Ԃ��A`False`�̎���c��Ԃ��֐��ł���A`v[i] = ifelse(fire[i], vreset, v[i])`��`fire[i]`��`True`�Ȃ�`v[i]`�����Z�b�g�d��`vreset`�Ƃ��A�����łȂ���΂��̂܂܂̒l��Ԃ��Ƃ��������ł���B���l�ɂ���`tlast[i]`�͔��΂����Ƃ��ɂ��̎������L�^����ϐ��ƂȂ��Ă���B�Ȃ��A`v_[i] = ifelse(fire[i], vpeak, v[i])`�͎��ۂ̃V�~�����[�V�����ɂ����ĈӖ����Ȃ��Ȃ��B�P�ɔ��Ύ��̓d��`vpeak`���܂߂ċL�^����ƕ`�掞�̌��h�����ǂ��Ƃ��������ł���B

������`struct`�Ɗ֐���p���ăV�~�����[�V���������s����B`I` ��HH���f���̂Ƃ��Ɠ����悤�ɋ�`�g����͂���B����`I`�͓��͓d���ł͂Ȃ����͓d���ɔ�Ⴗ��ʂƂȂ��Ă��邪�A����͖���R���悶����̒l�ł���ƍl����Ƃ悢�B

## 2.4.2 LIF���f���̃V�~�����[�V�����̎��s
�������̒萔��ݒ肵�ăV�~�����[�V���������s����B

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(25f0 * ((t .> 50) - (t .> 200)) + 50f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# �L�^�p
varr = zeros(Float32, nt, N)

# model�̒�`
neurons = LIF{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[i, :], dt)
    neurons.tcount += 1
    varr[i, :] = neurons.v_
end

`Plots`��ǂݍ��݁A���Ύ��d�ʂ��܂ޖ��d��`v_`�Ɠ��͓d��`I`��`�悷��B

using Plots

p1 = plot(t, varr[:, 1], color="black")
p2 = plot(t, I[:, 1], color="black")
plot(p1, p2, 
    xlabel = ["" "Times (ms)"], 
    ylabel= ["V (mV)" "Current"],
    layout = grid(2, 1, heights=[0.7, 0.3]), legend=false, size=(600,300))

## 2.4.3 LIF���f����F-I curve
### ���l�I�v�Z�ɂ��F-I curve�̕`��
���̍��ڂł�LIF���f���ɂ�������͓d���ɑ΂��锭�Η��̕ω� (F-I curve)��`�悷��B���@��HH���f���̏ꍇ�Ɠ��l�����A����͔��΂������ǂ��������f�����̕ϐ��Ƃ��Ė����I�ɋL�^����Ă���̂ŏ��������Ȃ��B

T = 1000 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps

N = 100 # �j���[�����̐�

# ���͎h��
mincurrent = 15
maxcurrent = 40
t = Array{Float32}(1:nt)*dt
I = Array{Float32}(range(mincurrent,maxcurrent,length=N)) # injection current

# model�̒�`
neurons = LIF{Float32}(N=N)

# �L�^�p
firearr = zeros(Bool, nt, N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[:], dt)
    neurons.tcount += 1
    firearr[i, :] = neurons.fire
end

���Η����v�Z���A�`�悷��B

num_spikes = sum(firearr, dims=1)
rate = num_spikes/T*1e3

plot(I[:], rate[1, :],
    xlabel="Input current",
    ylabel="Firing rate (Hz)", legend=false, size=(400,300))

����ɓd�������߂�Ɣ��Η��͖O�a(saturation)����B�Ȃ��A�s�������Ȃ��A���Ȃ킿0�̏ꍇ��臒l�t�߈ȊO��ReLU�֐��̂悤�ȋ���������B

### ��͓I�v�Z�ɂ��F-I curve�̕`��
�����܂ł͐��l�I�ȃV�~�����[�V�����ɂ��F-I curve�����߂��B�ȉ��ł͉�͓I��F-I curve�̎������߂悤�B��̓I�ɂ́A��肩�����I�ȓ��͓d����$I$�Ƃ����Ƃ���LIF�j���[�����̔��Η�(firing rate)��

$$
\begin{equation}
\text{rate}\approx \left(\tau_m \ln \frac{R_mI}{R_mI�{V_\text{rest}-V_{\text{th}}}\right)^{-1}
\end{equation}
$$

�Ƌߎ��ł��邱�Ƃ������B�܂��A$t=t_1$�ɃX�p�C�N���������Ƃ���B���̂Ƃ�, ���d�ʂ̓��Z�b�g�����̂�$V_m(t_1)=V_\text{rest}$�ł���(���Z�b�g�d�ʂƐÎ~���d�ʂ������Ɖ��肷��)�B$[t_1, t]$�ɂ����閌�d�ʂ�LIF�̎���ϕ����邱�Ƃœ�����B

$$
\begin{equation}
\tau_m \frac{dV_{m}(t)}{dt}=-(V_{m}(t)-V_\text{rest})+R_m I
\end{equation}
$$

�̎���ϕ������, 

$$
\begin{aligned}
\int_{t_1}^{t} \frac{\tau_m dV_m}{R_mI�{V_\text{rest}-V_m}&=\int_{t_1}^{t} dt\\
\ln \left(1-\frac{V_m(t)-V_\text{rest}}{R_mI}\right)&=-\frac{t-t_1}{\tau_m} \quad (\because V_m(t_1)=V_\text{rest})\\
\therefore\ \ V_m(t) &=V_\text{rest} + R_mI\left[1-\exp\left(-\frac{t-t_1}{\tau_m}\right)\right] 
\end{aligned}
$$

�ƂȂ�B$t>t_1$�ɂ����鏉�߂̃X�p�C�N��$t=t_2$�ɐ������Ƃ����, ���̂Ƃ��̖��d�ʂ�$V_m(t_2)=V_{\text{th}}$�ł��� (���ۂɂ�臒l�ȏ�ƂȂ��Ă���ꍇ������܂����ߎ�����)�B$t=t_2$����̎��ɑ������

$$
\begin{align}
V_{\text{th}}&=V_\text{rest} + R_mI\left[1-\exp\left(-\frac{t_2-t_1}{\tau}\right)\right] \\
\therefore\ \ T&= t_2-t_1 = \tau_m \ln \frac{R_mI}{R_mI�{V_\text{rest}-V_{\text{th}}}
\end{align}
$$

�ƂȂ�B������$T$��2�̃X�p�C�N�̎��ԊԊu (spike interval)�ł���B$t_1\leq t<t_2$�ɂ�����X�p�C�N��$t=t_1$����1�Ȃ̂�, ���Η���$1/T$�ƂȂ�B�����

$$
\text{rate}\approx \frac{1}{T}=\left(\tau_m \ln \frac{R_mI}{R_mI�{V_\text{rest}-V_{\text{th}}}\right)^{-1}
$$

�ƂȂ�B�s����$\tau_{\text{ref}}$���l�������, �����I�ɓ��͂�����ꍇ�͒P����$\tau_{\text{ref}}$�������΂��x���̂Ŕ��Η���$1/(\tau_{\text{ref}}+T)$�ƂȂ�B

����ł͂��̎��Ɋ�Â���F-I curve��`�悵�Ă݂悤�B

tc_m = 10 # �����萔 (ms)
tref = 2 # �s���� (ms)

R = 1.0 #����R 
vrest = -60.0 # �Î~���d�� (mV) 
vthr = -40.0 # 臒l�d�� (mV)
rate = zeros(N)

for i = 1:N
    z = R*I[i] / (R*I[i] + vrest - vthr)
    if z > 0
        rate[i] = 1 / (tref + tc_m * log(z)) * 1e3
    else
        rate[i] = 0
    end
end 

`log`�̒��g��0�ɂȂ��Error��������̂�if���ŏꍇ���������Ă���B�Ȃ��A`1e3`���悶�Ă���̂�1/ms����Hz�ɕϊ����邽�߂ł���B���ʂ͎��̂悤�ɂȂ�B���l�I�Ȍv�Z���ʂƂقڈ�v���Ă��邱�Ƃ��킩��B

plot(I[:], rate[:],
    xlabel="Input current",
    ylabel="Firing rate (Hz)", legend=false, size=(400,300))