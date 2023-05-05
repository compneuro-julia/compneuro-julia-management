# Plot results
T = range(0, 1, length=Nt)
figure(figsize=(8, 3), dpi=100)
subplot(1,2,1); title("Two-layer network")
for i in 1:4
    plot(T, A[:,i], label="a"*string(i))
    axhline(s[i], color="gray", linestyle="dashed")
end
xlim(0, 1); xlabel("t"); ylabel("A(t)"); legend(ncol=2)

subplot(1,2,2); title("Three-layer network")
for i in 1:4
    plot(T, B[:,i], label="b"*string(i))
    axhline(s[i], color="gray", linestyle="dashed")
end
xlim(0, 1); xlabel("t"); ylabel("B(t)"); legend(ncol=2)
tight_layout()