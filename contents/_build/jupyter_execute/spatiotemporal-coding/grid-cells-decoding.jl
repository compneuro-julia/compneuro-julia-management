using PyPlot, MAT, StatsBase, FFTW

# from http://www.ntnu.edu/kavli/research/grid-cell-data
pos = matopen("../_static/datasets/grid_cells_data/10704-07070407_POS.mat")
spk = matopen("../_static/datasets/grid_cells_data/10704-07070407_T2C3.mat")

post = read(pos, "post")[:] #  times at which positions were recorded
posx = read(pos, "posx")[:] #x positions
posy = read(pos, "posy")[:] # y positions
spkt = read(spk, "cellTS")[:] #spike time

println(size(post), size(posx), size(posy), size(spkt))

figure(figsize=(3,3))
plot(posx, posy, color="k", alpha=0.3)
xlim(-50, 50); ylim(-50, 50)
tight_layout()

function nearest_pos(array, value)
    idx = argmin(abs.(array .- value))
    return idx
end

idx = [nearest_pos(post, t) for t in spkt]

print(size(idx))

figure(figsize=(3,3))
plot(posx, posy, color="k", alpha=0.3)
scatter(posx[idx], posy[idx], color="r", s=5)
xlim(-50, 50); ylim(-50, 50)
tight_layout()

activ_hist = fit(Histogram, (posy[idx], posx[idx]), (-50:10:50, -50:10:50)).weights # activation
occup_hist = fit(Histogram, (posy, posx), (-50:10:50, -50:10:50)).weights　# occup position while trajectory 
occup_hist *= 0.02 # one time step is 0.02s 
occup_hist[occup_hist .== 0] .= 1 # avoid devide by zero

rate_hist = activ_hist ./ occup_hist;

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

function correlate_fft(x, y)
    corr = fftshift(real(ifft(fft(x) .* conj(fft(y)))))
    return corr / maximum(corr)
end;

corr_map = correlate_fft(rate_hist, rate_hist);

figure(figsize=(3, 2))
title("Autocorrelation map")
imshow(corr_map, origin="lower", cmap="turbo", interpolation="gaussian", extent=[-50, 50, -50, 50])
colorbar(label="Autocorr.")
tight_layout()
