time = (1:nt)*dt
figure(figsize=(4, 3))
plot(time, r_single, linestyle="dashed", label="single exponential")
plot(time, r_double, label="double exponential")
xlabel("Time (s)"); ylabel("Post-synaptic current (pA)")
legend(); tight_layout()