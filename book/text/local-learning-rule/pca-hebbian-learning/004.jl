y = 0:0.1:2;
θₘ = 1.0
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5)
figure(figsize=(3, 2))
plot(y, 1.5*y, label="Hebb則")
plot(y, ϕ.(y, θₘ), label="BCM則")
xlim(0,);
annotate(text="", xy=(0.8,0), xytext=(1.2,0), arrowprops=Dict("arrowstyle" => "<->", "color" => "tab:purple"))
axvline(θₘ, linestyle="dashed", color="tab:purple")
axhline(0, linestyle="dashed", color="tab:gray")
xticks([]); yticks([]); xlabel(L"$y$ "*"(シナプス後細胞の活動)")
text(0, 3.5, L"$\phi(y, \theta_m)$",ha="center",va="center")
text(2.2, 3, "Hebb則", color="tab:blue",fontsize=10)
text(2.2, 2, "BCM則", color="tab:orange",fontsize=10)
text(0.5, 0.2, L"\theta_m", color="tab:purple",fontsize=11)
text(-0.4, -0.3, "LTD",fontsize=11, color="tab:blue",ha="center",va="center", bbox=props);
text(-0.4, 1.8, "LTP",fontsize=11, color="tab:red",ha="center",va="center", bbox=props);
tight_layout()