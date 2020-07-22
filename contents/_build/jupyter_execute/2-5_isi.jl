# 2.5 Inter-spike interval ���f��
����܂ŏЉ�����f���ł́A���͂ɑ΂��閌�d�ʂȂǂ̎��ԕω��Ɋ�Â����΂��N���邩�ǂ����A�Ƃ������Ƃ��l���Ă����B���̐߂ł́A���΂�������܂ł̉ߒ����l�������A���΂̎��ԊԊu(**inter-spike interval, ISI**)�̓��v�ɂ�錻�ۘ_�I���f�����l����B�����**Inter-spike interval (ISI)** ���f���ƌĂԁBISI���f����**�_�ߒ�(point process)** �Ƃ������v�I���f���Ɋ�Â��Ă���A�e���f���ɂ�ISI���]�����z�̖��̂����Ă���B

���̐߂ł́A�g�p�p�x�̍��� **�|�A�\���ߒ� (Poisson process) ���f��**�A�|�A�\���ߒ����f���ɂ����ĕs�������l������ **�����ԕt���|�A�\���ߒ� (Poisson process with dead time, PPD) ���f��**�A�玿�̒�픭�΂ɂ����ă|�A�\���ߒ����f���������Ă͂܂肪�悢�Ƃ���� **�K���}�ߒ� (Gamma process) ���f��**�ɂ��Đ�������B

�Ȃ��ASNN�ɂ����āAISI���f���͎�ɉ摜���͂̍ۂ�**�A���l����X�p�C�N��ւ̃G���R�[�h**�ɗp������B����Ɍ��炸���͂Ƃ��ėp�����邱�Ƃ������B

````{margin}
```{note}
���̐߂� ([����, "�X�p�C�N���v���f������"](https://www.neuralengine.org/res/book/index.html); [Pachitariu, Probabilistic models for spike trains of single neurons"](http://www.gatsby.ucl.ac.uk/~marius/papers/SpikTrainStats.pdf)) ����ɎQ�l�ɂ����B
```
````

## 2.5.1 �|�A�\���ߒ����f��
### �_�ߒ��ƃ|�A�\���ߒ�
���Ԃɉ����ĕω�����m���ϐ��̂��Ƃ�**�m���ߒ�(stochastic process)** �Ƃ����B����Ɋm���ߒ��̒��ŁA�A�����Ԏ���ɂ����ė��U�I�ɐ��N����_���ۂ̌n���**�_�ߒ�(point process)** �Ƃ����B�X�p�C�N�͗��U�I�ɋN����̂ŁA�_�ߒ���p���ă��f�������ł���Ƃ����b�ł���B

�|�A�\���ߒ� (Poisson process)�͓_�ߒ���1�ł���B�|�A�\���ߒ����f���̓X�p�C�N�̔������|�A�\���ߒ��Ń��f�����������̂ŁA���̃��f���ɂ���Đ�����X�p�C�N���|�A�\���X�p�C�N(Poisson spike)�ƌĂԁB�|�A�\���ߒ��ł́A����$t$�܂łɋN�������_�̐�$N(t)$�̓|�A�\�����z�ɏ]���B���Ȃ킿�A�_���N����m�������x$\lambda$�̃|�A�\�����z�ɏ]���ꍇ, ����$t$�܂łɎ��ۂ�$n$��N����m����$P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$�ƂȂ�B 

�|�A�\���ߒ��ɂ����ē_���N����񐔂��|�A�\�����z�ɏ]�����Ƃ́A�|�A�\���ߒ��Ƃ������̗̂R���ƂȂ��Ă���B������`�Ƃ���ꍇ������΁A����4�����𖞂����_�ߒ����|�A�\���ߒ��Ƃ���Ƃ�����`������B

- ����0�ɂ����鏉���̓_�̐���0 : $P[N(0)=0]=1$ 
- $[t, t+\Delta t)$�ɓ_��1������m�� : $P[N(t+\Delta t)-N(t)=1]=\lambda(t)\Delta t+o(\Delta t)$
- ��������$\Delta t$�̊Ԃɓ_��2�ȏ㐶���Ȃ� : $P[N(t+\Delta t)-N(t)=2]=o(\Delta t)$
- �C�ӂ̎��_$t_1 < t_2 < \cdots< t_n$�ɑ΂��āC���� $N(t_2)-N(t_1), N(t_3)-N(t_2), \cdots, N(t_n)-N(t_{n�|1})$�݂͌��ɓƗ��ł���D

������, $o(\cdot)$��Landau�̋L��(Landau��small o)�ł���, $o(x)$��$x\to 0$�̂Ƃ��A$o(x)/x\to 0$�ƂȂ�����ȗʂ�\���B�|�A�\���ߒ��ɏ]���ăX�p�C�N��������Ƃ���ꍇ�A����2�̋��x�֐�$\lambda(t)$��**���Η�**���Ӗ����� (�܂������ɂ����ėL�p)�B����3�͕s������菬�����^�C���X�e�b�v�ɂ����ẮA1�̃^�C���X�e�b�v�ɂ�����1�����X�p�C�N�͐����Ȃ��Ƃ������Ƃ�\���B����4�̓X�p�C�N�͓Ɨ��ɔ�������A�Ƃ������Ƃ��Ӗ�����B�܂��A�����̏�������$N(t)$�̕��z�͋��x�ꐔ$\lambda(t)$�̃|�A�\�����z�ɏ]�����Ƃ�������B

���x�֐�(�_���X�p�C�N�̏ꍇ�A���Η�)��$\lambda(t)=\lambda$ (�萔)�ƂȂ�ꍇ�͓_�̎��ԊԊu(�_���X�p�C�N�̏ꍇ�AISI)�̊m���ϐ�$T$�����x�ꐔ$\lambda$�� **�w�����z**�ɏ]���B�Ȃ��A�w�����z�̊m�����x�֐��͊m���ϐ���$T$�Ƃ���Ƃ��A

$$
f(t;\lambda )=\left\{{\begin{array}{ll}\lambda e^{-\lambda t}&(t\geq 0)\\0&(t<0)\end{array}}\right.
$$

�ƂȂ�B���̂��Ƃ�4������Chapman-Kolmogorov�̎��ɂ�苁�߂��邪�A��₱�����̂�, $P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$���瓱�o�ł��邱�Ƃ��ȒP�Ɏ����B�w�����z�̗ݐϕ��z�֐���$F(t; \lambda)$�Ƃ���ƁA

$$
F(t; \lambda) = P(T<t)=1-P(T>t)=1-P(N(t)=0)=1-e^{-\lambda t}
$$

�ƂȂ�B�����

$$
f(t; \lambda)=\frac{dF(t; \lambda)}{dt}=\lambda e^{-\lambda t}
$$

�����藧�B

### ���|�A�\���ߒ�
��������|�A�\���ߒ��ɂ��X�p�C�N�̃V�~�����[�V��������������B�������@�ɂ�ISI���w�����z�ɏ]�����Ƃ𗘗p�������̂ƁA�|�A�\���ߒ��̏���2�𗘗p�������̂�2�ʂ肪����B�����͌�҂��y�Ōv�Z�ʂ����Ȃ����A��̃K���}�ߒ��̂��߂ɑO�҂̎������ɍs���B

#### ISI�̗ݐςɂ�蔭�Ύ��������߂��@
ISI���w�����z�ɏ]�����Ƃ𗘗p���ă|�A�\���ߒ����f���̎������s���B�܂�ISI���w�����z�ɏ]�������Ƃ���B����ISI��ݐς��邱�ƂŔ��Ύ����𓾂�B�Ō�ɔ��Ύ��Ԃ𐮐��l�Ɋۂ߂�index�Ƃ��邱�Ƃ�$\{0, 1\}$�̃X�p�C�N�񂪓�����BISI�̎擾�ɂ�`Random.randexp()`��p����B���̊֐��� scale 1�̎w�����z�ɏ]��������Ԃ��B����scale�͎w�����z�̊m�����x�֐���$f(t; \frac{1}{\beta}) = \frac{1}{\beta} e^{-t/\beta}$�Ƃ����ۂ�$\beta = 1/\lambda$�ł���(���̎��A���ς�$\beta$�ƂȂ�)�B����Ĕ��Η���`fr`(1/s), �P�ʎ��Ԃ�`dt`(s)�Ƃ����Ƃ���ISI�� `isi = 1/(fr*dt) * randexp()`�Ƃ��ē��邱�Ƃ��ł���B

�܂��K�v�ȃp�b�P�[�W��ǂݍ��ށB

using Random
using Plots

������seed�l��ݒ肵�A�K�v�Ȓ萔���`�������`isi`���v�Z����B`isi`��ݐς��邱�ƂŃX�p�C�N�̐������������L�^����z��`spike_time`���쐬����B�쐬��A`spike_time`��`�悷��B

Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps

n_neurons = 10 # �j���[�����̐�
fr = 30 # �|�A�\���X�p�C�N�̔��Η�(Hz)

isi = 1/(fr*dt*1e-3) * randexp(Int32(nt*1.5/fr), n_neurons)
spike_time = cumsum(isi, dims=1) # ISI��ݐ�

# raster plot
p = plot(xlabel ="Time (ms)", ylabel="Neuron index", xlim =(0, T+10), legend=false, size=(500, 200))
for i=1:n_neurons
    scatter!(p, spike_time[:, i], i*ones(Int32(nt*1.5/fr)), shape=:vline, color="black")
end
display(p)

���̐}�͊e�j���[���������΂������Ƃ�_�ŕ\���Ă���, **���X�^�[�v���b�g** (raster plot) �Ƃ����B

`spike_time`�̂悤�ɔ��Ύ����ŋL�^���Ă���������������ߖ�ł��邪�A�V�~�����[�V�����ɂ����Ă̓X�p�C�N��$S$�̓^�C���X�e�b�v���Ƃɔ��΂��Ă��邩��\��$\{0,1\}$�z��ŕێ����Ă����Ɗy�Ɉ������Ƃ��ł���B���̂��ߏ璷�ł͂��邪�A���Ύ����̔z���$\{0,1\}$�z��`spikes`�ɕϊ����X�p�C�N�̐��Ɣ��Η����v�Z����B

spike_time[spike_time .> nt - 1] .= 1 # nt�𒴂���ꍇ��1��
spike_time = round.(Int32, spike_time) # float to int
spikes = zeros(Bool, nt, n_neurons) # �X�p�C�N�L�^�ϐ�

for i=1:n_neurons    
    spikes[spike_time[:, i], i] .= 1
end

spikes[1] = 0 # (spike_time=1)�̔��΂��폜
print("Num. of spikes : ", sum(spikes))
print("\nFiring rate : ", sum(spikes)/(n_neurons*T)*1e3, "Hz")

#### $\Delta t$ �Ԃ̔��Ίm���� $\lambda\Delta t$ �ł��邱�Ƃ𗘗p������@
����2�Ԗڂ̃|�A�\���ߒ����f���̎������s���B�������$\lambda$�𔭉Η��Ƃ����ꍇ, ���$[t, t+\Delta t)$�̊ԂɃ|�A�\���X�p�C�N����������m����$\lambda \Delta t$�ƂȂ邱�Ƃ𗘗p����B����̓|�A�\���ߒ��̏��������A�|�A�\�����z���瓱���邱�Ƃ��ȒP�Ɏ����Ă����B���ۂ��N����m�������x$\lambda$�̃|�A�\�����z�ɏ]���ꍇ, ����$t$�܂łɎ��ۂ�$n$��N����m����$P[N(t)=n]=\dfrac{(\lambda t)^{n}}{n !} e^{-\lambda t}$�ƂȂ�B�����, ��������$\Delta t$�ɂ����Ď��ۂ�$1$��N����m����

$$
P[N(\Delta t)=1]=\dfrac{\lambda \Delta t}{1 !} e^{-\lambda \Delta t}\simeq \lambda \Delta t+o(\Delta t)
$$

�ƂȂ�B������, $e^{-\lambda \Delta t}$�ɂ��Ă̓}�N���[�����W�J�ɂ��ߎ����s���Ă���B���̂��Ƃ���, ��l���z$U(0,1)$�ɏ]������$\xi$���擾��, $\xi<\lambda dt$�Ȃ甭��$(y=1)$, ����ȊO�ł�$(y=0)$�ƂȂ�悤�ɂ���΃|�A�\���X�p�C�N�������ł���B

Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps

n_neurons = Int32(10) # �j���[�����̐�
fr = 30 # �|�A�\���X�p�C�N�̔��Η�(Hz)

spikes = rand(nt, n_neurons) .< fr*dt*1e-3

print("Num. of spikes : ", sum(spikes))
print("\nFiring rate : ", sum(spikes)/(n_neurons*T)*1e3, "Hz")

function rasterplot(spikes, plotsize=(500, 200))
    # input spike -> time, #neurons
    spike_inds = Tuple.(findall(x -> x > 0, spikes))
    spike_time = first.(spike_inds)
    neuron_inds = last.(spike_inds)
    
    # raster plot
    scatter(spike_time, neuron_inds,
        xlabel ="Time (ms)", ylabel="Neuron index",
        shape=:vline, color="black",
        legend=false, size=plotsize)
end

rasterplot(spikes)

�Ȃ��A�����ł͑S���Ԃɂ����锭�΂��܂Ƃ߂Čv�Z���Ă��邪�A�^�C���X�e�b�v���Ƃɔ��΂̗L�����v�Z���邱�Ƃ��ł���B�O�҂͔��Ώ���ێ����邽�߂̃��������K�v�����A�v�Z���Ԃ͒Z���Ȃ�B��҂̓������̐ߖ�ɂȂ邪�A�v�Z���Ԃ͒����Ȃ�B���̂��߁A�����2�̕��@�̓������ƌv�Z���Ԃ̃g���[�h�I�t�ƂȂ�B�܂��A���ɂ͔��Ώ���a�s��(sparse matrix)�̌`���ŕێ����Ă����ƃ������̐ߖ�ɂȂ�Ǝv����B

### ����|�A�\���ߒ�
����܂ł̎����͔��Η�$\lambda$�����ł���Ƃ���A���|�A�\���ߒ� (homogeneous poisson process)�ł��������A��������͔��Η�$\lambda(t)$�����ԕω�����Ƃ���**����|�A�\���ߒ�** (inhomogeneous poisson process)�ɂ��čl����B�Ƃ͂����A���|�A�\���ߒ��ɂ����锭�Η����A���Ԃɂ��Ă̊֐��Œu�������邾���Ŏ����ł���B�ȉ���$\lambda(t)=\sin^2(\alpha t)$(������$\alpha$�͒萔)�Ƃ����ꍇ�̎����ł���B

Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps

n_neurons = Int32(10) # �j���[�����̐�

t = Array{Float32}(1:nt)*dt
fr = 30(sin.(1e-2t)).^2 # �|�A�\���X�p�C�N�̔��Η�(Hz)

spikes = rand(nt, n_neurons) .< fr*dt*1e-3

p1 = plot(t, fr, ylabel ="Firing rate (Hz)", legend=false)
p2 = rasterplot(spikes)
plot(p1, p2, xlim=(0, T+10), layout = grid(2, 1, heights=[0.5, 0.5]), size=(500,300))

�オ���Η�$\lambda(t)$�̎��ԕω�, �������X�^�[�v���b�g�ł���B

## 2.5.2 �����ԕt���|�A�\���ߒ����f�� (Poisson process with dead time, PPD)
�|�A�\���ߒ��͊ȈՓI�ŗL�p�����A�s�������l�����Ă��Ȃ��B���̂��߁A���ɂ͐����I���e�𒴂����o�[�X�g���΂��N����ꍇ������[^burst]�B�����ŁA�|�A�\���ߒ��ɂ����ĕs�����̂悤�ȃC�x���g�̐��N���N����Ȃ� **������(dead time)** [^deadtime]���l������**�����ԕt���|�A�\���ߒ� (Poisson process with dead time, PPD)** (�܂���dead time modified Poisson process)�Ƃ������f���𓱓�����B

�����ɂ����Ă�LIF�j���[�����̎��Ɠ����悤�ȕs�����̏���������B�܂�A���݂��s�������ǂ����𔻒f���A�s�����Ȃ甭�΂������Ȃ��悤�ɂ���B

[^burst]: �����̃j���[��������̔��΂̏d�ˍ��킹(superposition)�ł���ƍl���邱�Ƃ��ł���B
[^deadtime]: �Ⴆ�΁A�K�C�K�[�E�J�E���^�[(Geiger counter)�Ȃǂ̕��ː��̌��o��ɂ͕��ː��̓��B���@��̕����I�����Ƃ��Č��o�ł��Ȃ�����(�܂莀����)������B���̂��ߕ��ː��̓��B�����|�A�\�����z�ɏ]���Ƃ����ꍇ�A���ː����葕�u�̃��f���Ƃ���PPD���p������B

Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps
tref = 5f0 # �s���� (ms)

n_neurons = Int32(10) # �j���[�����̐�
fr = 30 # �|�A�\���X�p�C�N�̔��Η�(Hz)

tlast = zeros(n_neurons) # ���Ύ����̋L�^�ϐ�
spikes = zeros(nt, n_neurons)

# simulation
@time for i=1:nt
    fire = rand(n_neurons) .< fr*dt*1e-3
    spikes[i, :] = ((dt*i) .> (tlast .+ tref)) .* fire
    tlast[:] = tlast .* (1 .- fire) + dt*i * fire # ���Ύ����̍X�V
end

print("Num. of spikes : ", sum(spikes))
print("\nFiring rate : ", sum(spikes)/(n_neurons*T)*1e3, "Hz")

`struct`��`function`���`���Ă��ǂ����A�����ł�for���[�v���ɒ��ڏ������������Bfor���[�v���Ɋւ��Ă͈ȉ��̂悤�Ƀj���[�������Ƃɏ������Ă��ǂ� (���x�ɑ傫�ȍ��͂Ȃ�)�B

```julia
# simulation
@time for i=1:nt
    fire = rand(n_neurons) .< fr*dt*1e-3
    for j=1:n_neurons
        spikes[i, j] = ifelse(dt*i > tlast[j] + tref, fire[j], 0)
        tlast[j] = ifelse(fire[j], dt*i*fire[j], tlast[j]) # ���Ύ����̍X�V
    end
end
```

�܂��A�s���������邽�߂ɔ��Η��͐ݒ�l��30Hz�����Ⴍ�Ȃ��Ă��邱�Ƃ�������B���Ƀ��X�^�[�v���b�g��`�悷��B

rasterplot(spikes)

�ʏ��Poisson spike�ƍ��͂��܂芴�����Ȃ����A���p�x���΂̏ꍇ�ɒʏ�̃��f���Ƃ̈Ⴂ�����ĂƂȂ�B

## 2.5.3 �K���}�ߒ����f��
�K���}�ߒ�(gamma process)�͓_�̎��ԊԊu���K���}���z�ɏ]���Ƃ��郂�f���ł���B�K���}�ߒ��̓|�A�\���ߒ������玿�ɂ������픭�΂ւ̓��Ă͂܂肪�ǂ��Ƃ���Ă��� ([Shinomoto, et al., 2003](https://pubmed.ncbi.nlm.nih.gov/14629869/); [Maimon & Assad,2009](https://pubmed.ncbi.nlm.nih.gov/19447097/))�B

���ԊԊu�̊m���ϐ���$T$�Ƃ����ꍇ�A�K���}���z�̊m�����x�֐���

$$
\begin{equation}
f(t;k,\theta) =  t^{k-1}\frac{e^{-t/\theta}}{\theta^k\Gamma(k)}
\end{equation}
$$

�ƕ\�����B�������A$t > 0$�ł���A 2�̕ꐔ��$k, \theta > 0$�ł���B�܂��A$\Gamma (\cdot)$�̓K���}�֐��ł���A

$$
\begin{equation}
\Gamma (k)=\int _{0}^{\infty }x^{k-1}e^{-x}\,dx
\end{equation}
$$

�ƒ�`�����B�K���}���z�̕��ς�$k\theta$�����A���Η���ISI�̕��ς̋t���Ȃ̂ŁA$\lambda=1/k\theta$�ƂȂ�B�܂��A$k=1$�̂Ƃ��A�K���}���z�͎w�����z�ƂȂ�B�����$k$���������̂Ƃ��A�K���}���z�̓A�[�������z�ƂȂ�B

�K���}�ߒ����f���̎����̓|�A�\���ߒ����f����ISI��ݐς����@�Ɠ��l�ɏ������Ƃ��ł���B���������̎��A[Distributions.jl](https://github.com/JuliaStats/Distributions.jl)��p����B��{�I�ɂ�`randexp(shape)`��`rand(Gamma(a,b), shape)`�ɒu��������΂悢 (������񑽏��̏C���͕K�v�Ƃ���)�B

using Distributions 

�X�p�C�N��𐶐�����֐��������B���X�璷�Ȃ̂��C�ɂȂ�_�ł���B

function GammaSpike(T, dt, n_neurons, fr, k)
    nt = Int32(T/dt) # number of timesteps
    theta = 1/(k*(fr*dt*1e-3)) # fr = 1/(k*theta)

    isi = rand(Gamma(k, theta), Int32(round(nt*1.5/fr)), n_neurons)
    spike_time = cumsum(isi, dims=1) # ISI��ݐ�

    spike_time[spike_time .> nt - 1] .= 1 # nt�𒴂���ꍇ��1��
    spike_time = round.(Int32, spike_time) # float to int
    spikes = zeros(Bool, nt, n_neurons) # �X�p�C�N�L�^�ϐ�

    for i=1:n_neurons    
        spikes[spike_time[:, i], i] .= 1
    end

    spikes[1] = 0 # (spike_time=1)�̔��΂��폜
    return spikes
end

`GammaSpike` �֐���p���� $k=1, 12$ �̏ꍇ�̃V�~�����[�V���������s����B�Ȃ��A$k=1$�̂Ƃ��̓|�A�\���ߒ��Ɉ�v���邱�Ƃɒ��ӂ��悤�B

Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps

n_neurons = 10 # �j���[�����̐�
fr = 30 # �K���}�X�p�C�N�̔��Η�(Hz)

# k=1�̂Ƃ��̓|�A�\���ߒ��Ɉ�v
spikes1 = GammaSpike(T, dt, n_neurons, fr, 1)
spikes2 = GammaSpike(T, dt, n_neurons, fr, 12)

print("Num. of spikes : ", sum(spikes1), ", ",sum(spikes2))
print("\nFiring rate : ", sum(spikes1)/(n_neurons*T)*1e3, "Hz, ", sum(spikes2)/(n_neurons*T)*1e3, "Hz")

ISI�̕��z��`�悷�邽�߂̊֐����`����B

function GammaISIplot(dt, fr, k, n=1000)
    theta = 1/(k*(fr*dt*1e-3)) # fr = 1/(k*theta)
    isi = rand(Gamma(k, theta), n)
    gamma_pdf = pdf.(Gamma(k, theta), minimum(isi):maximum(isi))

    p = plot(legend=false, xlabel="ISI (ms)", ylabel="Density")
    histogram!(p, isi, normed=true)
    plot!(p, minimum(isi):maximum(isi), gamma_pdf, color="black")
end

���ʂ�A������B��i��ISI�̕��z�A���i�̓��X�^�[�v���b�g�ł���B����$k=1$�̏ꍇ���|�A�\���ߒ����f���̃X�p�C�N��Ɣ�r���悤 (�����O�ςɂȂ��Ă��邱�Ƃ�������)�B�E��$k=12$�Ƃ����ꍇ�ł���B

p1 = GammaISIplot(dt, fr, 1)
p2 = GammaISIplot(dt, fr, 12)
p3 = rasterplot(spikes1)
p4 = rasterplot(spikes2)
plot(p1, p2, p3, p4,
    layout = grid(2, 2, widths=[0.5, 0.5], heights=[0.5, 0.5]), legend = false, size=(600, 300))

�Ȃ��A�O�q�����悤�ɃK���}�ߒ����f���̕����|�A�\���ߒ����f�������玿�j���[�����̃��f���Ƃ��Ă͗D��Ă��邪�A���͉摜�̃G���R�[�f�B���O���K���}�ߒ����f���ɂ��邱�Ƃ�SNN�̔F�����x�����シ�邩�ǂ����͂܂��\���Ɍ�������Ă��Ȃ��B�܂��A([Deger, et al., 2012](https://pubmed.ncbi.nlm.nih.gov/21964584/))�ł�PPD��K���}�ߒ��̏d�ˍ��킹�ɂ��X�p�C�N��𐶐�����A���S���Y�����l�Ă��Ă���B