fig, ax = subplots(1, 3, figsize=(6.5, 2), sharex="all", sharey="all")
titles = ["Activation map", "Occupation map", "Firing rate map"]
hists = [activ_hist, occup_hist, rate_hist]
for (i, (t, h)) in enumerate(zip(titles, hists))
    ax[i].set_title(t)
    ims = ax[i].imshow(h, origin="lower", cmap="turbo", interpolation="gaussian", extent=[-50, 50, -50, 50])
    if i == 3
        fig.colorbar(ims, label="Hz", ax=ax[i])
    end
end
tight_layout()