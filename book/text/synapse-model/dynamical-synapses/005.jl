figure(figsize=(8, 5))
subplot2grid((6, 2), (0, 0)); title("Facilitating synapse"); plot(tarray, s, "k"); ylabel("Presynaptic\n spike");
subplot2grid((6, 2), (0, 1)); title("Depressing synapse"); plot(tarray, s, "k");
subplot2grid((6, 2), (1, 0), rowspan=2); plot(tarray, uf); plot(tarray, xf, "tab:red"); ylabel("Synaptic variables"); ylim(0, 1.1); 
subplot2grid((6, 2), (1, 1), rowspan=2); plot(tarray, ud, label=L"$u$"); plot(tarray, xd, "tab:red", label=L"$x$"); ylim(0, 1.1); legend()
subplot2grid((6, 2), (3, 0)); plot(tarray, xuf, "k"); ylabel("Synaptic\n efficacy")
subplot2grid((6, 2), (3, 1)); plot(tarray, xud, "k"); 
subplot2grid((6, 2), (4, 0), rowspan=2); plot(tarray, rf, "k"); xlabel("Time (ms)"); ylabel("Postsynaptic current")
subplot2grid((6, 2), (4, 1), rowspan=2); plot(tarray, rd, "k"); xlabel("Time (ms)")
tight_layout()