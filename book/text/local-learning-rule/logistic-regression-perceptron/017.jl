figure(figsize=(4, 4))
scatter(p1[:, 1], p1[:, 2])
scatter(p2[:, 1], p2[:, 2])
plot(xx, yy, color="k")
xlabel(L"$x_1$"); ylabel(L"$x_2$"); 
tight_layout()