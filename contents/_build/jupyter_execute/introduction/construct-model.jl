using PyPlot
using PyPlot: matplotlib
#rc("axes.spines", top=false, right=false)

rc("font", family="Meiryo")
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5)

fig, ax = subplots(1,1, figsize=(6,2),sharex="all",sharey="all",constrained_layout=true)
ax.text(0.25, 0.5, "実験",fontsize=10,ha="center",va="center", bbox=props);
ax.text(0.5, 0.75, "神経科学の数理モデル",fontsize=10,ha="center",va="center", bbox=props);
ax.text(0.75, 0.5, "工学的モデル",fontsize=10,ha="center",va="center", bbox=props);


