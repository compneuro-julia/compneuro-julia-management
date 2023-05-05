figure(figsize=(6, 6), dpi=100)
ax = PyPlot.axes()
for i in 1:length(tree_info_eg)
    x, y = lines_eg[i][1]
    dx, dy = lines_eg[i][2] .- lines_eg[i][1]
    arrow(x, y, dx, dy,width=0.01,head_width=0.1,head_length=0.1,length_includes_head=true,color="k")
end
scatter(0, 0, s=100, color="k") # root

text(3, 4, L"遠心性位数 $\gamma$"*"\n (centrifugal order)", size=10, color="tab:red", ha="center", va="center")
hlines_y = [0, 0, -1, 2];
for i in 1:4
    hlines(i-1, hlines_y[i], 3, color="gray", linestyle="dashed", linewidth=2, alpha=0.5)
    text(3, (i-1)+0.2, string(i-1), size=10, color="tab:red", ha="center", va="center")
end

arrowprops=Dict("arrowstyle" => "->", "color" => "tab:blue");
annotate("根 (root)", xy=(-0.1, 0), xytext=(-1.0, 0), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("根分節\n (root segments)", xy=(0, 0.5), xytext=(-1.6, 0.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("中間分節\n (internal segments)", xy=(-0.5, 1.5), xytext=(-2.2, 1.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("末端分節\n (terminal segments)", xy=(-1.25, 2.25), xytext=(-2.8, 2.25), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("分岐点\n (branch point)", xy=(1.8, 3), xytext=(0.6, 3), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("成長円錐\n (growth cone)", xy=(1.5, 3.5), xytext=(0.3, 3.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("継続点\n (continuation point)", xy=(1.5, 2.4), xytext=(1.5, 1.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("一組の娘枝\n (daughter branches)", xy=(-0.75, 2.25), xytext=(-1.25, 3), size=10, color="tab:purple", ha="center", va="center", arrowprops=Dict("arrowstyle" => "->", "color" => "tab:purple"))
annotate("", xy=(-1.25, 2.25), xytext=(-1.1, 2.74), ha="center", va="center", arrowprops=Dict("arrowstyle" => "->", "color" => "tab:purple"))

xlim(-4, 3); ylim(-0.3, 4); axis("off")
ax.set_aspect("equal")
tight_layout()