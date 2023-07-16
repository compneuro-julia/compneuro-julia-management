figure(figsize=(6, 5))
subplot(3,1,1); plot(time, varr[:, 1], color="black"); ylabel("V (mV)")
subplot(3,1,2); labellist=["m" "h" "n"] 
for i in 1:3
    plot(time, gatearr[:, i, 1], label=labellist[i])
end; 
ylabel("Gating Value"); legend()
subplot(3,1,3); plot(time, Ie[:, 1], color="black"); ylabel(L"Current($\mu$A/cm$^2$)"); xlabel("Times (ms)")
tight_layout()