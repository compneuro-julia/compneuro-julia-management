@time Y_npca, W_npca = HebbianPCA(X_place; n_components=Ng, η=1e-2, maxiter=5000, func=relu, orthogonal=true);
Y_npca = reshape(Y_npca, (Ng, step, step));