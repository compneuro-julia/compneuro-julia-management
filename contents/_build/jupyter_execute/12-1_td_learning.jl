# 12.1 TD�w�K

TD�덷

$$
\begin{align}
\delta_{t} = r_{t+1} + \gamma V(s_{t+1}) - V(s_{t})
\end{align}
$$

���l�̍X�V

$$
\begin{align}
V(s_{t}) \leftarrow V(s_{t}) + \alpha \delta_{t}
\end{align}
$$

([Schultz, Dayan & Montague, Science, 1997](https://science.sciencemag.org/content/275/5306/1593))���Q�l�ɃV�~�����[�V�������s���B60���s����B0�b�̎��ɏ����h��(���h��)������B�܂��A6���s�ڂ���40���s�ڂ܂ŏ����h���������Ă���1.2�b��ɕ�V���^������Ƃ���B

using PyPlot

num_trial = 60 # ���s��
T = 3.0f0 # s
dt = 0.1f0 # s
nt = UInt(T/dt) + 1 # number of timesteps
value = zeros(num_trial, nt) 
delta = zeros(num_trial, nt) # TD error

flash_time = UInt(1.1f0/dt)
delay = UInt(1.2f0/dt)
reward_trial = 6:40
reward = zeros(num_trial, nt)
reward[reward_trial, flash_time+delay] .= 1.0

�� = 0.8 # �w�K��
�� = 0.99 # ������

# simulation
for i in 2:num_trial
    for t in 1:nt-1
        delta[i, t] = reward[i, t] + ��*value[i-1, t+1] - value[i-1, t]
        if t > flash_time
            value[i, t] = value[i-1, t] + ��*delta[i, t]
        end
    end
end

���ʂ�`�悷��B

figure(figsize=(8, 3.5))

subplot2grid((3,3), (0,0), rowspan=3)
imshow(value); colorbar()
title("Value"); ylabel("Trial"); xlabel("Time (s)"); xticks(0:Int(1/dt):nt, -1:1:T-1)

subplot2grid((3,3), (0,1), rowspan=3)
imshow(delta); colorbar()
title("TD error"); ylabel("Trial"); xlabel("Time (s)"); xticks(0:Int(1/dt):nt, -1:1:T-1)

subplot2grid((3,3), (0,2))
plot(-1:0.1:2, delta[6, :]); title("No CS + R (Trial #6)"); xticks([])

subplot2grid((3,3), (1,2))
plot(-1:0.1:2, delta[30, :]); title("CS + R (Trial #30)"); ylabel("TD error"); xticks([])

subplot2grid((3,3), (2,2))
plot(-1:0.1:2, delta[41, :]); title("CS + No R (Trial #41)"); xlabel("Time (s)")

tight_layout()

CS�͏����h��(conditioned stimulus), R�͕�V(reward)���Ӗ�����B