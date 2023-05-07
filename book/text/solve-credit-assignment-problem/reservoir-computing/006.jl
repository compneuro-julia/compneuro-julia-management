# 変数の初期値
k = 1 # 出力ニューロンの数
E = (2*np.random.rand(N, k) - 1)*Q
PSC = np.zeros(N).astype(np.float32) # シナプス後電流
JD = np.zeros(N).astype(np.float32) # 再帰入力の重み和
z = np.zeros(k).astype(np.float32) # 出力の初期化
Phi = np.zeros(N).astype(np.float32) #　学習される重みの初期値

# 記録用変数 
REC_v = np.zeros((nt,10)).astype(np.float32) # 膜電位の記録変数
current = np.zeros(nt).astype(np.float32) # 出力の電流の記録変数
tspike = np.zeros((4*nt,2)).astype(np.float32) # スパイク時刻の記録変数
ns = 0 # スパイク数の記録変数 