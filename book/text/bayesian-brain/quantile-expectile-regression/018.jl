# Classical TD learning
N = 20
c = get_cmap("brg") 
cmap = c(range(0, 0.5, length=N))
x = range(-1, 1, step=1e-2)
θ = range(π/6, π/3, length=N)
α = tan.(θ)
y = α * x'

# Plot
figure(figsize=(5.5, 3))
subplot(1,2,1)
axvline(x=0, color="gray", linestyle="dashed", linewidth=2)
axhline(y=0, color="gray", linestyle="dashed", linewidth=2)
for i in 1:N
    if i == Int(N/2)       
        plot(x, y[i, :], color=cmap[Int(N/2), :], alpha=1, linewidth=3, label="Neutral")
    else
        plot(x, y[i, :], color=cmap[Int(N/2), :], alpha=0.2)
    end
end

ylim(-1,1); xlim(-1,1)
xticks([]); yticks([])
legend(loc="upper left")
title("Classical TD learning")
xlabel("RPE")
ylabel("Firing")

# Distributional TD learning
α_pos = tan.(θ)
α_neg = reverse!(tan.(θ))
 
yd = (α_pos * ((x .> 0) .* x)' + (α_neg) * ((x .≤ 0) .* x)') 

# Plot
ax = subplot(1,2,2)
axvline(x=0, color="gray", linestyle="dashed", linewidth=2)
axhline(y=0, color="gray", linestyle="dashed", linewidth=2)
for i in 1:N
    if i == 1        
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Pessimistic")
    elseif i == Int(N/2)       
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Neutral")
    elseif i == N
        plot(x, yd[i, :], color=cmap[i, :], alpha=1, linewidth=3,
                 label="Optimistic")
    else
        plot(x, yd[i, :], color=cmap[i, :], alpha=0.2)
    end
end
handles, labels = ax.get_legend_handles_labels()
ax.legend(reverse!(handles), reverse!(labels), loc="upper left")
ylim(-1,1); xlim(-1,1); xticks([]); yticks([])
title("Distributional TD learning"); xlabel("RPE"); ylabel("Firing")
tight_layout()