using PyPlot, LinearAlgebra, Random, Distributions
using PyPlot: matplotlib
using KernelDensity
Random.seed!(2)
rc("font", family="Meiryo")

gaussian(x, μ, σ) = 1/(σ*sqrt(2*π))*exp(-0.5*((x-μ)/σ)^2)

x = -4:0.01:4
d = Normal(0, 1); 

props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5)
fig, ax = subplots(1,2, figsize=(6,2),sharex="all",sharey="all",constrained_layout=true)
ax[1].set_title("点推定\n (Point estimation)")
ax[2].set_title("分布推定\n (Distribution estimation)")
ax[2].set_ylabel(L"$p(\theta)$")
ax[1].scatter(0, 0,clip_on=false,zorder=5)
ax[2].plot(x, pdf.(d, x), zorder=5)
ax[1].spines.left.set_visible(false)
for i in 1:2
    ax[i].set_xlabel(L"$\theta^*$")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(0, )
    ax[i].set_xlim(-4,4)
    ax[i].axvline(0, linestyle="dashed", color="tab:gray")
    ax[i].spines.right.set_visible(false)
    ax[i].spines.top.set_visible(false)
end

ax[2].annotate(text="", xy=(0,pdf(d, 1)), xytext=(1,pdf(d, 1)), arrowprops=Dict("arrowstyle" => "<->", "color" => "tab:red"))
ax[2].text(0.55, pdf(d, 1)-0.06, L"$\sigma^*$",ha="center",va="center",color="tab:red");

ax[1].text(-6, -0.25, "神経活動で\n表現する方法",fontsize=11,ha="center",va="center");
ax[1].text(0, -0.25, "単一の細胞の瞬時的な活動",fontsize=10,ha="center",va="center", bbox=props);
ax[2].text(0, -0.25, "単一の細胞の経時的な活動 (例. SBC)\n or\n 複数の細胞の瞬時的な活動 (例. PPC)",fontsize=10,ha="center",va="center", bbox=props);