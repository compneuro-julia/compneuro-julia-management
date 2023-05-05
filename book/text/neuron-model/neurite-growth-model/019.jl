maxwidth, minwidth = 1.5, 0.5 # 描画する神経突起の最大/最小の太さ
days_range = floor.(Int, collect(range(1, 18, history_num+1)))[2:end]

fig, ax = subplots(1, history_num, sharex=true, sharey=true)
for i in 1:history_num
    lines, _ = segments_lines(tree_info_history[i], seg_vec_history[i]);
    linewidths = range(maxwidth, minwidth, length=length(lines));

    line_segments = matplotlib.collections.LineCollection(lines, linewidths=linewidths, color="k")
    ax[i].add_collection(line_segments)  # dendrite
    ax[i].scatter(0, 0, s=50, color="k") # soma
    ax[i].set_xlabel(L"$x\ (\mu m)$"); ax[i].set_ylabel(L"$y\ (\mu m)$")
    ax[i].set_aspect("equal")
    ax[i].set_title(string(days_range[i])*" days")
    ax[i].label_outer()
end