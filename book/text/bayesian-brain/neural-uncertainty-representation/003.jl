mosaic = """
DA
DB
EC
"""

fig, ax = plt.subplot_mosaic(mosaic,figsize=(6, 4),constrained_layout=true)
cm = get_cmap(:turbo) # get color map
ax["D"].set_title("サンプリングに基づく符号化 (SBC)")
ax["D"].plot(samples, 1:Ns, "o-", alpha=0.5)
ax["D"].set_ylim(-1,)
ax["D"].invert_yaxis()
ax["D"].text(-1.55, -0.3, "発火率", ha="center",va="center"); 
ax["D"].text(-2.25, 1.375, "時間", ha="center",va="center",rotation="vertical")
ax["D"].hlines(y=0.3, xmin=-2.1, xmax=-1.0, clip_on=false, color="k", linewidths=1)
ax["D"].vlines(x=-2.1, ymin=0.3, ymax=2.0, clip_on=false, color="k", linewidths=1)
ax["D"].xaxis.tick_top()
ax["D"].xaxis.set_label_position("top") 
ax["D"].axis("off")

ax["E"].set_title(L"$\Downarrow$", fontsize=20)
ax["E"].plot(x, y)
ax["E"].plot(U.x, U.density)
ax["E"].scatter(samples, zeros(Ns), alpha=0.5, marker="o")
ax["E"].spines.right.set_visible(false)
ax["E"].spines.top.set_visible(false)
ax["E"].set_xlabel("Variable")
ax["E"].set_ylabel("確率密度");

ax["A"].set_title("確率的集団符号化 (PPC)")

for i in 1:Np
    ax["A"].scatter(μs[i], ppc_fr[i],color=cm(i/Np),zorder=5)
end
ax["A"].plot(μs, ppc_fr, "--", color="tab:gray")
ax["A"].set_ylabel("発火率"); 
ax["A"].set_xlabel("Neuron #")
for i in 1:Np
    ax["B"].plot(x, tuning_curves[i, :],color=cm(i/Np))
end

ax["B"].set_title(L"$\times$", fontsize=20)
ax["B"].set_xlabel("Stimulus value")

ax["C"].set_title(L"$\Downarrow$", fontsize=20)
ax["C"].plot(x, y)
ax["C"].plot(x, act)
ax["C"].set_xlabel("Variable")
for i in 'A':'C'
    ax[string(i)].spines.right.set_visible(false)
    ax[string(i)].spines.top.set_visible(false)
end

for i in 'A':'E'
    ax[string(i)].set_xlim(-2.1, 2.1)
    ax[string(i)].set_xticks([])
    ax[string(i)].set_yticks([])
end

#fig.tight_layout()