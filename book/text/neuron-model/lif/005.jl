function update!(variable::LIF, param::LIFParameter, Ie::Vector, dt)
    @unpack N, v, v_, fire, tlast, tcount = variable
    @unpack tref, tc_m, vrest, vreset, vthr, vpeak = param
    
    @inbounds for i = 1:N
        #v[i] += dt * ((vrest - v[i] + Ie[i]) / tc_m) # 不応期を考慮しない場合の更新式
        v[i] += dt * ((dt*tcount) > (tlast[i] + tref))*((vrest - v[i] + Ie[i]) / tc_m)
        #v[i] += dt * ifelse(dt*tcount[1] > tlast[i] + tref, (vrest - v[i] + Ie[i]) / tc_m, 0)
    end
    @inbounds for i = 1:N
        fire[i] = v[i] >= vthr
        v_[i] = ifelse(fire[i], vpeak, v[i]) #発火時の電位も含めて記録するための変数 (除いてもよい)
        v[i] = ifelse(fire[i], vreset, v[i])        
        tlast[i] = ifelse(fire[i], dt*tcount, tlast[i]) # 発火時刻の更新
    end
end