figure(figsize=(4, 4))
scatter(X[y.==0, 1], X[y.==0, 2])
scatter(X[y.==1, 1], X[y.==1, 2])
xlabel(L"$x_1$"); ylabel(L"$x_2$"); 
tight_layout()