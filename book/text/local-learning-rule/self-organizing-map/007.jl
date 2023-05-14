function plot_som(v, w; vcolor="tab:blue")
    num_w, dims = size(w)
    num_w_sqrt = Int(sqrt(num_w))
    rw = reshape(w, (num_w_sqrt, num_w_sqrt, dims))
    scatter(v[:, 1], v[:, 2], s=10, color=vcolor)
    plot(rw[:, :, 1], rw[:, :, 2], "k", alpha=0.5); 
    plot(rw[:, :, 1]', rw[:, :, 2]', "k", alpha=0.5)
    scatter(w[:, 1], w[:, 2], s=5, fc="white", ec="k", zorder=99) # w[i, j, 1]とw[i, j, 2]の点をプロット
end;