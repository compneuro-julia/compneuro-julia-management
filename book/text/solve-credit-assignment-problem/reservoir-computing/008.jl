for t in tqdm(range(nt)):
    I = PSC + np.dot(E, z) + BIAS # シナプス電流 
    s = neurons(I) # 中間ニューロンのスパイク
    
    index = np.where(s)[0] # 発火したニューロンのindex
    len_idx = len(index) # 発火したニューロンの数
    if len_idx > 0:
        JD = np.sum(OMEGA[:, index], axis=1)  
        tspike[ns:ns+len_idx,:] = np.vstack((index, 0*index+dt*t)).T
        ns = ns + len_idx # スパイク数の記録

    PSC = synapses_rec(JD*(len_idx>0)) # 再帰的入力電流
    #PSC = np.dot(OMEGA, r) # 遅い
    r = synapses_out(s) # 出力電流(神経伝達物質の放出量)  
    r = np.expand_dims(r,1) # (N,) -> (N, 1)
    z = np.dot(Phi.T, r) # デコードされた出力
    err = z - zx[t] # 誤差

    # FORCE法(RLS)による重み更新
    if t % step == 1:
        if t > tmin:
            if t < tcrit:
                cd = np.dot(P, r)
                Phi = Phi - np.dot(cd, err.T)
                P = P - np.dot(cd, cd.T) / (1.0 + np.dot(r.T, cd)
    
    current[t] = z # デコード結果の記録
    REC_v[t] = neurons.v_[:10] # 膜電位の記録