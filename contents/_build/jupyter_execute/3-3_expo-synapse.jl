# 3.3 �w���֐��^�V�i�v�X���f��
�V�i�v�X�̃��f���͕������邪, �ǂ��p������̂�**�w���֐��^�V�i�v�X���f��**(exponential synapse model)�ł���B���̃��f���͐����w�I�ȉߒ��𖳎��������ۘ_�I���f���ł��邱�Ƃɒ��ӂ��悤�B�w���֐��^�V�i�v�X���f���ɂ�2�̎��, **�P��w���֐��^���f��** (single exponential model)��**��d�w���֐��^���f��** (double exponential model)������B

�����̐����̑O�Ƀ��f���̋����������B���}��2��ނ̃��f���ɂ�����$t=0$�ŃX�p�C�N�������Ă���̃V�i�v�X��d���̕ω��������Ă���B������, ���ۂ̃V�i�v�X��d���͂����**�V�i�v�X���x** (Synaptic strength)[^synstr]���悶�đ��a����������̂ƂȂ�B

[^synstr]: �V�i�v�X���x�Ƃ����͕̂֋X��̌ď̂�, ���ۂɂ͐_�o�`�B�����̎�ނ�, ���̎�e�̂̐��ȂǕ����̗v���ɂ���Č��肳��Ă���. �܂�, ���̃V�i�v�X���x�̓V�i�v�X�d�݂Ƃ������Ƃ�����B����͂ǂ��炩�ƌ����΋@�B�w�K�̕\���Ɉ�������ꂽ���̂ł���B���̂���, ���̃T�C�g�ł͏d�݂Ƃ�������g���B

using PyPlot

dt = 5e-5 # �^�C���X�e�b�v (sec)
td = 2e-2 # synaptic decay time (sec)
tr = 2e-3 # synaptic rise time (sec)
T = 0.1 # �V�~�����[�V�������� (sec)
nt = Int(T/dt) # �V�~�����[�V�����̑��X�e�b�v

# �P��w���֐��^�V�i�v�X
r1 = zeros(nt)

for t in 1:nt-1
    spike = ifelse(t == 1, 1, 0)
    r1[t+1] = r1[t]*(1-dt/td) + spike/td
    #r1[t+1] = r1[t]*exp(-dt/td) + spike/td
end

# ��d�w���֐��^�V�i�v�X
r2 = zeros(nt) 
hr = zeros(nt)

for t in 1:nt-1
    spike = ifelse(t == 1, 1, 0)
    double_r[t] = r
    r2[t+1] = r2[t]*(1-dt/tr) + hr[t]*dt
    hr[t+1] = hr[t]*(1-dt/td) + spike/(tr*td)
    #r2[t+1] = r2[t]*exp(-dt/tr) + hr[t]*dt
    #hr[t+1] = hr[t]*exp(-dt/td) + spike/(tr*td)
end   

time = (1:nt)*dt
figure(figsize=(4, 3))
plot(time, r1, linestyle="dashed", label="single exponential")
plot(time, r2, label="double exponential")
xlabel("Time (s)"); ylabel("Post-synaptic current (pA)")
legend()
tight_layout()

2��ނ̎w���֐��^�V�i�v�X�̓��ԁB�j���͒P��w���֐��^�V�i�v�X��, �����͓�d�w���֐��^�V�i�v�X�ł���B

## 3.3.1 �P��w���֐��^���f��(Single exponential model)
�V�i�v�X�O�j���[�����ɂ����ăX�p�C�N�������Ă���̃V�i�v�X��d���̕ω��͂����悻�w���֐��I�Ɍ�������, �Ƃ����̂��P��w���֐��^���f���ł��� [^comp]. ���͎��̂悤�ɂȂ�B

[^comp]: ��w���Ԃ̐Ò�1�R���p�[�g�����g���f���Ɠ������ł���B

$$
\begin{equation}
f(t)=\frac{1}{\tau_{s}}\exp\left(-\frac{t}{\tau_s}\right)    
\end{equation}
$$

���̊֐������ԓI�ȃt�B���^�[�Ƃ���, �ߋ��̑S�ẴX�p�C�N�ɂ��Ă̑��a�����B

$$
\begin{equation}
r(t)=\sum_{t_{k}< t} f\left(t-t_{k}\right)
\end{equation}
$$

������${r(t)}$�͑O�߂ɂ�����V�i�v�X����($s_{\text{syn}}$)��, $t_{k}$�͂���j���[������$k$�Ԗڂ̃X�p�C�N�̔��������ł���B${t_{k}<t}$�̈Ӗ��͌��݂̎���$t$�܂łɔ��������X�p�C�N�ɂ��Ă̘a�����Ƃ����Ӗ��ł���B�Ȃ��A�X�p�C�N�������Ă���, ������x�̎��Ԃ��o�߂�����͂��̃X�p�C�N�̉e���͂Ȃ��ƌ��Ȃ���̂�, ���̎��Ԃ܂ł̑��a�����̂��悢�B

�ʂ̕\�L�@�Ƃ��ăX�p�C�N��ɑ΂����ݍ��݂��s���Ƃ������̂�����B��ݍ��݉��Z�q��$*$�Ƃ�, �V�i�v�X�O�זE�̃X�p�C�N���$S(t)=\sum_{t_{k}< t} \delta\left(t-t_{k}\right)$�Ƃ��� (������, $\delta$��Dirac��delta�֐��ɂ�����$\delta(0)=1$�Ƃ����֐�)�B���̂Ƃ�, $r(t)=f*S(t)$�ƕ\�����Ƃ��ł���B��ݍ��݉��Z�q��p����Ɗȗ��ȕ\�L���ł��邪�A������͑��Ɠ�����@��p����B

### �����������ɂ��\��
��̎�@�ł̓j���[�����̔��Ύ������L����, ���Ԗ��ɑS�ẴX�p�C�N�ɂ��Ă̘a�����K�v������B������, ��������ꍇ�͎��̓����Ȕ�����������p����B

$$
\begin{equation}
\frac{dr}{dt}=-\frac{r}{\tau_{s}}+\frac{1}{\tau_{s}} \sum_{t_{k}< t} \delta\left(t-t_{k}\right)   
\end{equation}
$$

������$\tau_s$�̓V�i�v�X�̎��萔(synaptic time constant)�ł���B �܂�, $\delta(\cdot)$��Dirac��delta�֐��ł�(������$\delta(0)=1$�ł�). �����Euler�@�ō���������� 

$$
\begin{equation}
r(t+\Delta t)=\left(1-\frac{\Delta t}{\tau_{s}}\right)r(t)+\frac{1}{\tau_{s}}\delta_{t,t_{k}} 
\end{equation}
$$

�ƂȂ�B������$\delta_{t,t_{k}}$��Kronecker��delta�֐���, $t=t_{k}$�̂Ƃ���1, ����ȊO��0�ƂȂ�B�܂������x�Ƃ���$\left(1-\Delta  t/\tau_{d}\right)$�̑����$\exp\left(-\Delta t/\tau_{d}\right)$��p����ꍇ������B

## 3.3.2 ��d�w���֐��^���f��(Double exponential model)
2�d�̎w���֐��ɂ��V�i�v�X��d���̗����オ����l������̂�, ��d�w���֐��^���f��(Double exponential model)�ł���[^comp2]�B$t=0$�ɃV�i�v�X�O�זE�����΂����Ƃ��̃V�i�v�X��d���̎��ԕω��̊֐��͎��̂悤�ɂȂ�B
[^comp2]: ��w���Ԃ̓���1�R���p�[�g�����g���f���Ɠ������ł���B

$$
\begin{equation}
f(t)=A\left[\exp\left(-\frac{t}{\tau_d}\right)-\exp\left(-\frac{t}{\tau_r}\right)\right]    
\end{equation}
$$

������, ${\tau_r}$�͗����オ�莞�萔(synaptic rise time constant), ${\tau_d}$�͌������萔(synaptic decay time constant)�ł���B$\tau_{d}$��$\tau_{s}$�Ɠ������_�o�`�B�����̌������x�����肵�Ă���B$A$�͋K�i���萔�Ŏ��̂悤�ɕ\�����B

$$
\begin{equation}
A=\frac{\tau_d}{\tau_d-\tau_r}\cdot \left(\frac{\tau_r}{\tau_d}\right)^\frac{\tau_r}{\tau_r-\tau_d}    
\end{equation}
$$

�K�i���萔$A$���悶�邱�Ƃōő�l��1�ƂȂ�B������, �V�~�����[�V�����������Ŏ��ۂɋK�i��������ꍇ�͏��Ȃ��B

### $\alpha$�֐�
��L�̎��ɂ�����, $\tau=\tau_{r}=\tau_{d}$�̏ꍇ�� $\boldsymbol{\alpha}$ **�֐�** (alpha function, alpha synapse)�ƌĂ� ([Rall, 1967](https://pubmed.ncbi.nlm.nih.gov/6055351/))�B���Ƃ��Ă͎��̂悤�ɂȂ�B

$$
\begin{equation}
\alpha(t)=\frac{t}{\tau}\exp\left(1-\frac{t}{\tau}\right)    
\end{equation}
$$

���̎��͓�d�w���֐��^�V�i�v�X�̎��ɒP�ɑ�����邾���ł͓��o�ł��Ȃ��B�����̎��̑Ή��ɂ��Ă͌�q����B

### �����������ɂ��\��
������, ��d�w���֐��^�V�i�v�X�̎��ɑΉ�����, �⏕�ϐ�$h$��p���������������𓱓�����B 

$$
\begin{align} 
\frac{dr}{dt}&=-\frac{r}{\tau_{d}}+h\\
\frac{dh}{dt}&=-\frac{h}{\tau_{r}}+\frac{1}{\tau_{r} \tau_{d}} \sum_{t_{k}< t} \delta\left(t-t_{k}\right) 
\end{align} 
$$

�P��w���֐��^�V�i�v�X�̏ꍇ�Ɠ��l��Euler�@�ō���������� 

$$
\begin{align} 
r(t+\Delta t)&=\left(1-\frac{\Delta t}{\tau_{d}}\right)r(t)+h(t)\cdot \Delta t\\ 
h(t+\Delta t)&=\left(1-\frac{\Delta t}{\tau_{r}}\right)h(t)+\frac{1}{\tau_{r}\tau_{d}} \delta_{t,t_{j k}}
\end{align}
$$

�ƂȂ�B

�O�̂���, �����������ƌ��̎�����v���邱�Ƃ��m�F���Ă������B$t=0$�̂Ƃ��ɃV�i�v�X�O�זE�����΂����Ƃ�, ����ȍ~�̔��΂͂Ȃ��Ƃ���B���̂Ƃ�, $h(0)=1/\tau_{r}\tau_{d}, r(0)=0$ �ł���B$h$�ɂ��Ă̔����������̉���

$$
\begin{equation}
h(t)=h(0)\cdot \exp\left(-\frac{t}{\tau_r}\right)    
\end{equation}
$$

�ƂȂ�̂�, �����$r$�ɂ��Ă̎��ɑ������

$$
\begin{equation}
\frac{dr}{dt}=-\frac{r}{\tau_{d}}+h(0)\cdot \exp\left(-\frac{t}{\tau_r}\right) 
\end{equation}
$$

�ƂȂ�B����������ɂ͗��ӂɐϕ����q$\exp({t}/{\tau_d})$�������Ă���ϕ������邩Laplace�ϊ������邩�ł���B�����Laplace�ϊ���p����B�E�ӈꍀ�ڂ��ڍs������ɗ��ӂ�Laplace�ϊ�����ƈȉ��̂悤�ɂȂ�B

$$
\begin{align}
\mathcal{L}\left[\frac{dr}{dt}+r/\tau_{d}\right]&=\mathcal{L}\left[h(0)\cdot \exp\left(-t/\tau_r\right)\right]\\
sF(s)-r(0)+\frac{1}{\tau_{d}}F(s)&=\frac{h(0)}{s+1/\tau_r}\\
F(s)&=\frac{h(0)}{(s+1/\tau_r)(s+1/\tau_d)}
\end{align}
$$

������$r(t)$��Laplace�ϊ���$F(s)$�Ƃ���. �����ŋtLaplace�ϊ����s���Ǝ��̂悤�ɂȂ�B

$$
\begin{align}
r(t)&=\mathcal{L}^{-1}(F(s))\\
&=\mathcal{L}^{-1}\left[\frac{h(0)}{(s+1/\tau_r)(s+1/\tau_d)}\right]\\
&=\mathcal{L}^{-1}\left[\frac{h(0)}{1/\tau_r-1/\tau_d}\left(\frac{1}{s+1/\tau_d}-\frac{1}{s+1/\tau_r}\right)\right]\\
&=\frac{1}{\tau_d-\tau_r}\left[\exp(-t/\tau_d)-\exp(-t/\tau_r)\right]
\end{align}
$$

���̎��̍ő�l$r_{\max}$�����߂Ă������B $r(t)$���������0�ƒu�������̉�$t_{\max}$��������΋��߂���B�v�Z�����, 

$$
\begin{equation}
t_{\max}=\dfrac{\ln(\tau_d/\tau_r)}{1/\tau_r-1/\tau_d},\ \ r_{\max}=\dfrac{1}{\tau_{d}}\cdot \left(\dfrac{\tau_{r}}{\tau_{d}}\right)^{\frac{\tau_{r}}{\tau_d-\tau_{r}}}    
\end{equation}
$$

�ƂȂ�B

�Ȃ�, $\alpha$�֐��̓��o�͋tLaplace�ϊ�������O��$\tau=\tau_d=\tau_r$�Ƃ���΂悭, 

$$
\begin{align}
F_\alpha(s)&=\frac{h(0)}{(s+1/\tau)^2}\\
\alpha(t)&=\frac{t}{\tau^2}\exp\left(-\frac{t}{\tau}\right)
\end{align}
$$
�ƂȂ�B�኱�̌W���̈Ⴂ�͂��邪, �����`�̊֐������o���ꂽ�B 