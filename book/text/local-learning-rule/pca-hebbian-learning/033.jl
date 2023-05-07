@time Y_pca, W_pca = HebbianPCA(X_place, n_components=Ng, η=1e-2, maxiter=5000, orthogonal=true)
Y_pca = reshape(Y_pca, (Ng, step, step));