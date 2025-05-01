## 短期的シナプス可塑性

シナプス前活動に応じて**シナプス伝達効率** (synaptic efficacy) が動的に変化する性質を**短期的シナプス可塑性** (Short-term synaptic plasticity; STSP) といい，このような性質を持つシナプスを**動的シナプス** (dynamical synapses)と呼ぶ．シナプス伝達効率が減衰する現象を短期抑圧 (short-term depression; STD)，増強する現象を短期促通(short-term facilitation; STF)という．さらにそれぞれに対応するシナプスを減衰シナプス，増強シナプスという．ここでは{cite:p}`Mongillo2008-kk`および{cite:p}`Orhan2019-rq`で用いられている定式化を使用する．

$$
\begin{align}
\frac{\mathrm{d} x(t)}{\mathrm{d} t}=\frac{1-x(t)}{\tau_{x}}-u(t) x(t) r(t) \Delta t \\
\frac{\mathrm{d} u(t)}{\mathrm{d} t}=\frac{U-u(t)}{\tau_{u}}+U(1-u(t)) r(t) \Delta t
\end{align}
$$

ただし，$x$を利用可能な神経伝達物質の量, $u$を利用されている神経伝達物質の量(the neurotransmitter utilization), $\tau_x$は神経伝達物質の時定数 , $\tau_u$はutilization, $U$はincrement , $\Delta t$を時間幅とする．ここでは$\tau_x=$(200 ms/1,500 ms; facilitating/depressing),  $\tau_u=$(1,500 ms/200 ms; facilitating/depressing), $U=$(0.15/0.45; facilitating/depressing), $\Delta t=$10msとする．