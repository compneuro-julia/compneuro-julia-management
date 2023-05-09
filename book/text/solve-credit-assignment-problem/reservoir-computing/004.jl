from Models.Neurons import CurrentBasedLIF
from Models.Synapses import DoubleExponentialSynapse

# ニューロンとシナプスの定義 
neurons = CurrentBasedLIF(N=N, dt=dt, tref=tref, tc_m=tc_m,
                          vrest=vrest, vreset=vreset, vthr=vthr, vpeak=vpeak)
neurons.v = vreset + np.random.rand(N)*(vpeak-vreset) # 膜電位の初期化

synapses_out = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)
synapses_rec = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)

# 再帰重みの初期値
p = 0.1 # ネットワークのスパース性
OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(np.sqrt(N)*p)
for i in range(N):
    QS = np.where(np.abs(OMEGA[i,:])>0)[0]
    OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS], axis=0)/len(QS)