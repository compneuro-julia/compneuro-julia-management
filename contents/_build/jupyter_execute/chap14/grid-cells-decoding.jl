using MAT

# from http://www.ntnu.edu/kavli/research/grid-cell-data
pos = matopen("../_static/datasets/grid_cells_data/10704-07070407_POS.mat")
spk = matopen("../_static/datasets/grid_cells_data/10704-07070407_T2C3.mat")

using PyPlot

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

using StatsBase

activ_hist = fit(Histogram, (posy[idx], posx[idx]), (-50:10:50, -50:10:50)).weights # activation
occup_hist = fit(Histogram, (posy, posx), (-50:10:50, -50:10:50)).weights　# occup position while trajectory 
occup_hist *= 0.02 # one time step is 0.02s 
occup_hist[occup_hist .== 0] .= 1 # avoid devide by zero

rate_hist = activ_hist ./ occup_hist

println(size(rate_hist))

figure(figsize=(10.5, 3))
subplot(1,3,1)
title("Activation map")
imshow(activ_hist, interpolation="gaussian", extent=[-50, 50, 50, -50])
gca().invert_yaxis()

subplot(1,3,2)
title("Occupation map")
imshow(occup_hist, interpolation="gaussian", extent=[-50, 50, 50, -50])
gca().invert_yaxis()

subplot(1,3,3)
title("Firing rate map")
imshow(rate_hist, cmap="jet", interpolation="gaussian", extent=[-50, 50, 50, -50])
colorbar(label="Hz")
gca().invert_yaxis()

tight_layout()

using PyCall
sc = pyimport("scipy.signal");

corr_map = sc.correlate2d(rate_hist, rate_hist, fillvalue=4)

println(size(corr_map))

figure(figsize=(3, 3))
title("Autocorrelation map")
imshow(corr_map, cmap="jet", interpolation="gaussian")
gca().invert_yaxis()
tight_layout()
