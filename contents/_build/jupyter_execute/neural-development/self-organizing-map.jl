using Random, PyPlot, ProgressMeter

# inputs 
N = 400 # num of inputs
dims = 2  # dims of inputs 
Random.seed!(1234);
σv, σw = 0.2, 0.05
v = [σv*randn(Int(N/2), dims);  1.0 .+ σv*randn(Int(N/2), dims)]
map_width = 10
w = σw*randn(map_width, map_width, dims) .+ 0.5;

function plot_som(v, w)
    scatter(v[:, 1], v[:, 2], s=10)
    plot(w[:, :, 1], w[:, :, 2], "k", alpha=0.8); plot(w[:, :, 1]', w[:, :, 2]', "k", alpha=0.8)
    scatter(w[:, :, 1], w[:, :, 2], s=10) # w[i, j, 1]とw[i, j, 2]の点をプロット
end;

# Gaussian mask for inputs
function gaussian_mask(sizex=9, sizey=9; σ=5)
    x, y = 0:sizex-1, 0:sizey-1
    X, Y = ones(sizey) * x', y * ones(sizex)' 
    x0, y0 = (sizex-1) / 2, (sizey-1) / 2
    mask = exp.(-((X .- x0) .^2 + (Y .- y0) .^2) / (2.0(σ^2)))
    return mask ./ sum(mask)
end;

function SOM!(v, w; α0=1.0, σ0=6, T=500)
    # α0: update rate, σ0 : width, T : training steps
    map_width = size(w)[1]
    N = size(v)[1]
    w_history = [copy(w)] # history of w
    @showprogress for t in 1:T
        α = α0 * (1 - t/T); # update rate
        σ = (σ0 - 1) * (1 - t/T) + 1; # decay from large to small
        wm = ceil(Int, σ)
        h = gaussian_mask(2wm+1, 2wm+1, σ=σ);
        # loop for the N inputs
        for i in 1:N
            dist = sum([(v[i, j] .- w[:, :, j]).^2 for j in 1:dims]) # distance between input and neurons
            win_idx = argmin(dist) # winner index
            idx = [max(1,win_idx[j] - wm):min(map_width, win_idx[j] + wm) for j in 1:2] # neighbor indices
            # update the winner & neighbor neuron
            η = α * h[1:length(idx[1]), 1:length(idx[2])]
            for j in 1:dims
                w[idx..., j] += η .* (v[i, j] .- w[idx..., j])
            end
        end
        append!(w_history, [copy(w)]) # save w
    end
    return w_history
end;

w_history = SOM!(v, w, α0=1.0, σ0=6, T=100);

figure(figsize=(6, 4)) 
idx = [1, 50, 100]
for i in 1:length(idx)
    wh = w_history[idx[i]]
    subplot(2,length(idx),i)
    title("Epoch : "*string(idx[i]))
    plot_som(v, wh); axis("off")
    subplot(2,length(idx),i+length(idx))
    imshow(wh[:, :, 1]); axis("off")
end
tight_layout()

product(sets...) = hcat([collect(x) for x in Iterators.product(sets...)]...)' # Array of Cartesian product of sets 
pol2cart(θ, r) = r*[cos(θ), sin(θ)];

# generate stimulus
Random.seed!(1234);
Nx, Ny, NOD, NOR = 10, 10, 2, 12
dims = 5  # dims of inputs 
l, r = 0.14, 0.2

rx, ry = range(0, 1, length=Nx), range(0, 1, length=Ny)
rOD = range(-l, l, length=NOD)
rORθ = range(-π/2, π/2, length=NOR+1)[1:end-1]

# stimuli
v = product(rx, ry, rOD, rORθ, r)
rORxy = hcat(pol2cart.(2v[:, 4], v[:, 5])...)
v[:, 4], v[:, 5] = rORxy[1, :], rORxy[2, :];
v += (rand(size(v)...) .- 1) * 1e-5;

# initial neurons
map_width = 64
M = map_width^2
w = product(range(0, 1, length=map_width), range(0, 1, length=map_width))
w += (rand(size(w)...) .- 1) * 0.05;
w = [w 2l*(rand(M) .- 0.5) hcat(pol2cart.(4π*(rand(M) .- 0.5), r*rand(M))...)']
w = reshape(w, (map_width, map_width, dims));

SOM!(v, w, α0=1.0, σ0=5, T=50);

function plot_visual_maps(v, w)
    figure(figsize=(9, 8))
    subplot(2,2,1, adjustable="box", aspect=1); title("Retinotopic map")
    plot_som(v, w)

    ax1 = subplot(2,2,2, adjustable="box", aspect=1); title("Ocular dominance (OD) map")
    imshow(w[:, :, 3], cmap="gray", origin="lower") 
    
    ins1 = ax1.inset_axes([1.05,0,0.05,1])
    colorbar(cax=ins1, aspect=40, pad=0.08, shrink=0.6)
    ins1.text(0, -0.15, "Left", ha="center", va="center")
    ins1.text(0, 0.15, "Right", ha="center", va="center")
    
    subplot(2,2,3, adjustable="box", aspect=1); title("Contours of OD and OR")
    ORmap = atan.(w[:, :, 5], w[:, :, 4]); # get angle of polar 
    sizex, sizey = map_width, map_width
    x, y = 0:sizex-1, 0:sizey-1
    X, Y = ones(sizey) * x', y * ones(sizex)';
    contour(X, Y, ORmap, cmap="hsv")
    contour(X, Y, w[:, :, 3], colors="k", levels=1)

    ax2 = subplot(2,2,4, adjustable="box", aspect=1); title("Orientation (OR) angle map")
    imshow(ORmap, cmap="hsv", origin="lower")
    
    cm = get_cmap(:hsv)
    lines, colors = [], []
    for i in 1:9
        θ = (i-1)/8*π
        c, s = cos(θ), sin(θ)
        push!(lines, [(-c/2, 15-1.5i -s/2), (c/2, 15-1.5i + s/2)])
        push!(colors, cm(1/8*(i-1)))
    end
    
    ins2 = ax2.inset_axes([1,0,0.2,1])
    ins2.add_collection(matplotlib.collections.LineCollection(lines, linewidths=3,color=colors))
    ins2.set_aspect("equal")
    ins2.axis("off")
    ins2.set_xlim(-1, 1); ins2.set_ylim(0, 15)
    
    tight_layout()
end;

plot_visual_maps(v, w)
