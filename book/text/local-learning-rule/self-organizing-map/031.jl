function plot_visual_maps(v, w)
    figure(figsize=(7, 6))
    subplot(2,2,1, adjustable="box", aspect=1); title("Retinotopic map")
    plot_som(v, w)

    M, dims = size(w)
    map_width = Int(sqrt(M))
    rw = reshape(w, (map_width, map_width, dims))

    ax1 = subplot(2,2,2, adjustable="box", aspect=1); title("Ocular dominance (OD) map")
    imshow(rw[:, :, 3], cmap="gray", origin="lower") 
    
    ins1 = ax1.inset_axes([1.05,0,0.05,1])
    colorbar(cax=ins1, aspect=40, pad=0.08, shrink=0.6)
    ins1.text(0, -0.15, "Left", ha="center", va="center")
    ins1.text(0, 0.15, "Right", ha="center", va="center")
    
    subplot(2,2,3, adjustable="box", aspect=1); title("Contours of OD and OR")
    ORmap = atan.(rw[:, :, 5], rw[:, :, 4]); # get angle of polar 
    sizex, sizey = map_width, map_width
    x, y = 0:sizex-1, 0:sizey-1
    X, Y = ones(sizey) * x', y * ones(sizex)';
    contour(X, Y, ORmap, cmap="hsv")
    contour(X, Y, rw[:, :, 3], colors="k", levels=1)

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