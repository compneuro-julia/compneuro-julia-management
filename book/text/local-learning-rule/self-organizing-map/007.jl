function plot_som(v, w; vcolor="tab:blue", wcolor="tab:orange")
    M, dims = size(w)
    map_width = Int(sqrt(M))
    rw = reshape(w, (map_width, map_width, dims))
    scatter(v[:, 1], v[:, 2], s=10, color=vcolor)
    plot(rw[:, :, 1], rw[:, :, 2], "k", alpha=0.8); plot(rw[:, :, 1]', rw[:, :, 2]', "k", alpha=0.8)
    scatter(w[:, 1], w[:, 2], s=10, color=wcolor) # w[i, j, 1]とw[i, j, 2]の点をプロット
end;