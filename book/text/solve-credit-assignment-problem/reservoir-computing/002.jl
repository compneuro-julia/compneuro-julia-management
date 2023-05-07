N = 2000  # ニューロンの数
dt = 5e-5 # タイムステップ(s)
tref = 2e-3 # 不応期(s)
tc_m = 1e-2 #　膜時定数(s)
vreset = -65 # リセット電位(mV) 
vrest = 0 # 静止膜電位(mV)
vthr = -40 # 閾値電位(mV)
vpeak = 30 # ピーク電位(mV)
BIAS = -40 # 入力電流のバイアス(pA)
td = 2e-2; tr = 2e-3 # シナプスの時定数(s)
alpha = dt*0.1  
P = np.eye(N)*alpha
Q = 10; G = 0.04

T = 15 # シミュレーション時間 (s)
tmin = round(5/dt) # 重み更新の開始ステップ
tcrit = round(10/dt) # 重み更新の終了ステップ
step = 50 # 重み更新のステップ間隔
nt = round(T/dt) # シミュレーションステップ数
zx = np.sin(2*np.pi*np.arange(nt)*dt*5) # 教師信号