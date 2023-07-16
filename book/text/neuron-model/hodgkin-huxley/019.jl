figure(figsize=(6, 3))
subplot(2,1,1); plot(time, varr_cs[:, 1], color="black"); ylabel("V (mV)")
subplot(2,1,2); plot(time, Iext_cs[:, 1], color="black"); ylabel(L"Current($\mu$A/cm$^2$)"); xlabel("Times (ms)")
tight_layout()