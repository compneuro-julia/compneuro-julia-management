num_trial = 60 # 試行回数
T = 3.0f0 # s
dt = 0.1f0 # s
nt = UInt(T/dt) + 1 # number of timesteps
value = zeros(num_trial, nt) 
delta = zeros(num_trial, nt) # TD error

flash_time = UInt(1.1f0/dt)
delay = UInt(1.2f0/dt)
reward_trial = 6:40 # 報酬が貰える試行の区間を設定する
reward = zeros(num_trial, nt)
reward[reward_trial, flash_time+delay] .= 1.0

α = 0.8 # 学習率
γ = 0.99 # 割引率

# simulation
for i in 2:num_trial
    for t in 1:nt-1
        delta[i, t] = reward[i, t] + γ*value[i-1, t+1] - value[i-1, t]
        if t > flash_time
            value[i, t] = value[i-1, t] + α*delta[i, t]
        end
    end
end