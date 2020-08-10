# 2.5 Izhikevich ���f��
## 2.5.1 Izhikevich ���f���̒�`
**Izhikevich ���f��** (�܂���**Simple model**)��([Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm))�ōl�Ă��ꂽ���f���ł���BHH���f���̂悤�Ȑ����w�I�Ȓm���Ɋ�Â������f���͎��ۂ̃j���[�����̔��Γ������悭�Č��ł��邪�A�������G�����邽�߁A���w�I�ȉ�͂�����A�v�Z�ʂ������邽�߂ɑ�K�͂ȃV�~�����[�V����������ƂȂ�[^hh]�B�����ŁA�����w�I�Ȑ������ɂ͖ڂ��Ԃ�A���̓��ł̃j���[�����̔��Γ������Č����郂�f�������߂�ꂽ�B���̓��������̂�Izhikevich ���f���ł��� (�ȉ��ł�Iz���f���ƕ\�L����)�BIz���f���� 2�ϐ������Ȃ�[^ber]�ȑf�Ȕ�������������, �l�X�ȃj���[�����̊�����͕킷�邱�Ƃ��ł���B�莮���ɂ͎��2��ނ���B�܂��A([Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm))�Œ�Ă��ꂽ�̂������ł���B

$$
\begin{align}
\frac{dv(t)}{dt}&=0.04v(t)^2 + 5v(t)+140-u(t)+I(t) \\
\frac{du(t)}{dt}&=a(bv(t)-u(t))
\end{align} 
$$

�����ŁA$v$��$u$���ϐ��ł���, $v$�͖��d��(membrane potential;�P�ʂ�mV), $u$�͉񕜓d��(recovery current; �P�ʂ�pA)[^rec]�ł���B�܂��A$a$�͉񕜎��萔(recovery time constant; �P�ʂ�ms$^{-1}$)�̋t�� (���ꂪ�傫����$u$�����ɖ߂鎞�Ԃ��Z���Ȃ�), $b$��$u$��$v$�ɑ΂��銴��(���x����,  resonance; �P�ʂ�pA/mV)�ł���B

���̎��͊ȕւ����A�����w�I�ȈӖ��Â���������ɂ����B���P���ꂽ���Ƃ���["Dynamical Systems in Neuroscience" (Izhikevich, 2007)](https://mitpress.mit.edu/books/dynamical-systems-neuroscience)��Chapter 8�ŏЉ��Ă���̂������ł���B

$$
\begin{align}
C\frac{dv(t)}{dt}&=k\left(v(t)-v_r\right)\left(v(t)-v_t\right)-u(t)+I(t) \\
\frac{du(t)}{dt}&=a\left\{b\left(v(t)-v_{r}\right)-u(t)\right\}
\end{align} 
$$

�����ŁA$C$�͖��e��(membrane capacitance; �P�ʂ�pF), $v_r$�͐Î~���d��(resting membrane potential; �P�ʂ�mV), $v_t$��臒l�d��(instantaneous threshold potential; �P�ʂ�mV), $k$�̓j���[�����̃Q�C���Ɋւ��萔�ŁA�������Ɣ��΂��₷���Ȃ� (�P�ʂ�pA/mV)�B�Ȍ�͂�����̎���p����B

Iz���f����**臒l�̎�舵��**��LIF���f���ƈقȂ�AHH���f���ɋ߂��BLIF���f���ł�臒l�𒴂������ɖ��d�ʂ��s�[�N�d�ʂ܂ŏ㏸���� (���̉ߒ��͖����Ă��悢)�A�����Ė��d�ʂ����Z�b�g����BIz���f����臒l��$v_t$����, ���d�ʂ̃��Z�b�g��臒l�𒴂������Ŕ��f�����A���d��$v$���s�[�N�d��$v_{\text{peak}}$�ɂȂ����Ƃ��i�܂��͒��������j�ɍs���B���̂���Iz���f���̎��ۂ�臒l�͖��d�ʂ̋������ω�����(���Ώ�ԂɈڍs����)�A�܂蕪��(bifurcation) ��������_�ł���A�p�����[�^��臒l$v_t$�Ƃ̊Ԃɂ͍��ق�����B

���āA���d�ʂ��s�[�N�d��$v_{\text{peak}}$�ɒB�����Ƃ� (���Ȃ킿 `if` $v \geq v_{\text{peak}}$)�A$u, v$�����̂悤�Ƀ��Z�b�g����[^burst]�B

$$
\begin{align} 
u&\leftarrow u+d\\
v&\leftarrow v_{\text{reset}}
\end{align}
$$

�Ƃ���B������, $v_{\text{reset}}$�͉ߕ��ɂ��l�����ĐÎ~���d��$v_r$�����������l�Ƃ���B�܂��A$d$�̓X�p�C�N���Β��Ɋ���������鐳���̊O�����d���̍��v��\���A���Ό�̖��d�ʂ̋����ɉe������ (�P�ʂ�pA)�B

�ȏ�𓥂܂���, �V�~�����[�V�������s���B�܂��A�K�v�ȃp�b�P�[�W��ǂݍ��ށB

[^hh]: ����Ɋւ��Ă͕K�������������Ȃ��B�v�Z�@�̔��B�ɂ��HH���f���ő傫�ȃ��f�����V�~�����[�V�������邱�Ƃ��\�ł���B
[^ber]: ���l�v�Z�������ł͊ȈՓI�����Aif�������邽�߂ɉ�͂�����͓̂���Ȃ�B([Bernardo, et al., 2008](https://www.springer.com/gp/book/9781846280399))��ǂނƂ����炵���B
[^rec]: �����ł́u�񕜁v�Ƃ����̂͒E���ɂ�����̖��d�ʂ��Î~���d�ʂւƖ߂�A�Ƃ����Ӗ��ł��� (�΋`���activation�Ŗ��d�ʂ̏㏸���Ӗ�����)�B
$u$��$v$�̓��֐��ɂ�����$v$�̏㏸��}������悤��$-u$�œ����Ă��邽�߁A$u$�Ƃ��Ă�K$^+$�`���l���d����Na$^+$�`���l���̕s���������ԂȂǂ��l������B
[^burst]: �o�[�X�g����(bursting)�̋�����\�����邽�߂ɂ́A�����񕜕ϐ�(fast recovery variable)�ƒx���񕜕ϐ�(slow recovery variable)��2���K�v�ƂȂ�(�]���Ė��d�ʂ����킹�đS����3�ϐ��K�v)�B����ŁAIz���f���ł�LIF���f���̂悤��if���ɂ�郊�Z�b�g��p���Ă��邽�߁A�����񕜕ϐ����K�v�Ȃ��A�x���񕜕ϐ�$u$�݂̂Ńo�[�X�g���΂�\���ł���B


using Base: @kwdef
using Parameters: @unpack # or using UnPack

�ύX���Ȃ��萔��ێ�����`struct`��`IZParameter`�ƁA�ϐ���ێ�����`mutable struct`��`IZ`���쐬����B2�̒莮���Ńp�����[�^�̒l���قȂ�̂Œ��ӂ��邱�ƁB

@kwdef struct IZParameter{FT}
    C::FT = 100  # ���e�� (pF)
    a::FT = 0.03 # �񕜎��萔�̋t�� (1/ms)
    b::FT = -2 # u �� v �ɑ΂��鋤�x���� (pA/mV)
    d::FT = 100 # ���΂Ŋ���������鐳���̊O�����d�� (pA)
    k::FT = 0.7 # �Q�C�� (pA/mV)
    vthr::FT = -40 # 臒l�d�� (mV)
    vrest::FT = -60 # �Î~���d�� (mV)
    vreset::FT = -50 # ���Z�b�g�d�� (mV)
    vpeak::FT = 35 #�@�s�[�N�d�� (mV)
end

@kwdef mutable struct IZ{FT}
    param::IZParameter = IZParameter{FT}()
    N::UInt32
    v::Vector{FT} = fill(param.vrest, N)
    u::Vector{FT} = zeros(N)
    fire::Vector{Bool} = zeros(Bool, N)
end

���ɕϐ����X�V����֐�`update!`�������BLIF�̏ꍇ�ƈقȂ�A`v[i] >= vpeak`�ł��邱�Ƃɒ��ӂ��� (`v[i] >= vthr`�ł͂Ȃ�)�B

function update!(variable::IZ, param::IZParameter, I::Vector, dt)
    @unpack N, v, u, fire = variable
    @unpack C, a, b, d, k, vthr, vrest, vreset, vpeak = param
    @inbounds for i = 1:N
        v[i] += dt/C * (k*(v[i]-vrest)*(v[i]-vthr) - u[i] + I[i])
        u[i] += dt * (a * (b * (v[i]-vrest) - u[i]))
    end
    @inbounds for i = 1:N
        fire[i] = v[i] >= vpeak
        v[i] = ifelse(fire[i], vreset, v[i])
        u[i] += ifelse(fire[i], d, 0)
    end
end

## 2.5.2 Izhikevich ���f���̃V�~�����[�V�����̎��s
�������̒萔��ݒ肵�ăV�~�����[�V���������s����B

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # �j���[�����̐�

# ���͎h��
t = Array{Float32}(1:nt)*dt
I = repeat(150f0 * ((t .> 50) - (t .> 200)) + 300f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# �L�^�p
varr = zeros(Float32, nt, N)
uarr = zeros(Float32, nt, N)

# model�̒�`
neurons = IZ{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[i, :], dt)
    varr[i, :] = neurons.v
    uarr[i, :] = neurons.u
end

`Plots`��ǂݍ��݁A���d��`v`, �񕜕ϐ�`u`, ���͓d��`I`��`�悷��B

using Plots

p1 = plot(t, varr[:, 1])
p2 = plot(t, uarr[:, 1])
p3 = plot(t, I[:, 1])
plot(p1, p2, p3, 
    title= ["Regular Spiking (RS) Neurons" "" ""],
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["Membrane\n potential (mV)" "Recovery\n current (pA)" "Injection\n current (pA)"],
    layout = grid(3, 1, heights=[0.5, 0.25, 0.25]), legend = false, size=(500, 400))

## 2.5.3 �l�X�Ȕ��΃p�^�[���̃V�~�����[�V����
���ɗl�X�Ȕ��΃p�^�[����͕킷��悤��Iz���f���̒萔��ω������Ă݂悤�BIntrinsically Bursting (IB)�j���[������Chattering (CH) �j���[����(�܂��� fast rhythmic bursting (FRB) �j���[����)�̃V�~�����[�V�������s���B��{�I�ɂ͒萔��ς��邾���ł���B

```{note}
�{���ŗp���Ă��鎮�ɂ����锭�΃p�^�[���ɑ΂���p�����[�^��([Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm))�ł͓����Ȃ����A["Dynamical Systems in Neuroscience" (Izhikevich, 2007)](https://mitpress.mit.edu/books/dynamical-systems-neuroscience)�ɂ͋L�ڂ�����B���̔��΃p�^�[���Ɋւ��Ă͂��̖{���Q�Ƃ̂��ƁB
```

# �L�^�p
varr_ib = zeros(Float32, nt, N)
varr_ch = zeros(Float32, nt, N)

I = repeat(500f0 * ((t .> 50) - (t .> 200)) + 700f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# IB neurons
neurons_ib = IZ{Float32}(N=N, 
    param=IZParameter{Float32}(C = 150, a = 0.01, b = 5, k =1.2, d = 130, vrest = -75, vreset = -56, vthr = -45, vpeak = 50))

# CH neurons
neurons_ch = IZ{Float32}(N=N, 
    param=IZParameter{Float32}(C = 50, a = 0.03, b = 1, k =1.5, d = 150, vrest = -60, vreset = -40, vthr = -40, vpeak = 35))

# simulation
@time for i = 1:nt
    update!(neurons_ib, neurons_ib.param, I[i, :], dt)
    update!(neurons_ch, neurons_ch.param, I[i, :], dt)
    varr_ib[i, :] = neurons_ib.v
    varr_ch[i, :] = neurons_ch.v
end

����܂łƈقȂ�A���f���̒�`����`param`��ݒ肵�Ă��邱�Ƃɒ��ӂ��悤�B�Ō�ɖ��d�ʕω���`�悷��B

p1 = plot(t, varr_ib[:, 1])
p2 = plot(t, varr_ch[:, 1])
p3 = plot(t, I[:, 1])
p4 = plot(t, I[:, 1])
plot(p1, p2, p3, p4,
    title = ["IB Neurons" "CH neurons" "" ""],
    xlabel = ["" "" "Times (ms)" "Times (ms)"], 
    ylabel= ["Membrane\n potential (mV)" "" "Injection\n current (pA)" ""],
    layout = grid(2, 2, heights=[0.7, 0.3], widths=[0.5, 0.5]), legend = false, size=(600, 300))

## 2.5.4 �����_���l�b�g���[�N�̃V�~�����[�V����
1000��Iz�j���[����(������800��, �}����200��)�ɂ�郉���_���l�b�g���[�N�̃V�~�����[�V�������s���B�����([Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm))�ɂ�����MATLAB�R�[�h��������Ă���A�����Julia�ɈڐA�������̂ł���B���̃V�~�����[�V�����ł�RS(regular spiking)�j���[�������������זE�AFS(fast spiking)�j���[������}�����זE�̃��f���Ƃ��ėp���Ă���B

# Excitatory neurons    Inhibitory neurons
Ne = 800;               Ni = 200
re = rand(Ne,1);        ri = rand(Ni,1)
a = [0.02*ones(Ne,1);   0.02 .+ 0.08*ri]
b = [0.2*ones(Ne,1);    0.25 .- 0.05*ri]
c = [-65 .+ 15*re.^2;   -65*ones(Ni,1)]
d = [8 .- 6*re.^2;      2*ones(Ni,1)]
S = [0.5*rand(Ne+Ni,Ne) -rand(Ne+Ni,Ni)] # synaptic weight
v = -65*ones(Ne+Ni,1)   # Initial values of v
u = b .* v              # Initial values of u
firings = []            # spike timings

for t=1:1000 # simulation of 1000 ms
    I=[5*randn(Ne,1); 2*randn(Ni,1)] # thalamic input
    fired = findall(v[:, 1] .>= 30) # indices of spikes
    firings = t==1 ? [t .+ 0*fired fired] : [firings; [t .+ 0*fired fired]]
    v[fired]=c[fired]
    u[fired]=u[fired]+d[fired]
    I = I + sum(S[:,fired], dims=2)
    v = v .+0.5*(0.04*v.^2+5*v .+140 -u+I) # step 0.5 ms for numerical stability
    v = v .+0.5*(0.04*v.^2+5*v .+140 -u+I) 
    u = u+a.*(b.*v-u)
end

���d�ʂ̍X�V�̍ہA`v`��2��ɕ����čX�V���Ă��邪�A����͐��l�I�Ȉ��萫�����߂邽�߂ł���B�v�Z�ʂ͏オ�邪�A�O�q�������f���ɂ����Ă����l�̏������s������������B

�V�~�����[�V�����̎��s��A�l�b�g���[�N���\������j���[�����̔��΂�`�悷��B�����**���X�^�[�v���b�g** (raster plot)�Ƃ����B���̐}�͉��������ԁA�c�����j���[�����̔ԍ��ƂȂ��Ă���A�e�j���[���������΂������Ƃ�_�ŕ\���Ă���B

scatter(firings[:,1], firings[:,2], markersize=1, markercolor="black", 
    xlabel="Time (ms)", ylabel="# neuron", xlim=(0, 1000), ylim=(0, 1000), legend=false)

���߂�400ms���炢�܂ł�100ms���Ƃ�10Hz��$\alpha$�g��