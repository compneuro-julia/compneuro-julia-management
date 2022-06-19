using PyPlot, Random, Distributions, LinearAlgebra, FFTW
rc("axes.spines", top=false, right=false)
rc("font", family="Meiryo")

ϕ(y, θₘ) = y * (y - θₘ);

y = 0:0.1:2;
θₘ = 1.0
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5)
figure(figsize=(3, 2))
plot(y, ϕ.(y, θₘ))
xlim(0,)
annotate(text="", xy=(0.8,1), xytext=(1.2,1), arrowprops=Dict("arrowstyle" => "<->", "color" => "tab:purple"))
axvline(θₘ, linestyle="dashed", color="tab:purple")
axhline(0, linestyle="dashed", color="tab:gray")
xticks([]); xlabel(L"$y$ "*"(シナプス後細胞の活動)")
ylabel(L"$\phi(y, \theta_m)$")
text(0.7, 1.2, L"\theta_m", color="tab:purple",fontsize=11)
text(0.5, 1.8, "LTD",fontsize=11, color="tab:blue",ha="center",va="center", bbox=props);
text(1.5, 1.8, "LTP",fontsize=11, color="tab:red",ha="center",va="center", bbox=props);
tight_layout()

d = MvNormal([0,0], [1.0 0.5; 0.5 1.0]) # multivariate normal distribution
N = 300 # sample size
Random.seed!(0) # set seed
X = rand(d, N);  # generate toy data

U, S, V = svd(X*X')

figure(figsize=(3,3))
scatter(X[1,:], X[2,:], alpha=0.5)
arrow(0, 0, V[1,1], V[2,1], head_width=0.2, color="tab:red", length_includes_head=true, label="PC1")
arrow(0, 0, V[1,2], V[2,2], head_width=0.2, color="tab:orange", length_includes_head=true, label="PC2")
θc = 0:1e-2:2pi
plot(cos.(θc), sin.(θc), "k--", alpha=0.8)
xlabel(L"$X_1$"); ylabel(L"$X_2$")
legend(); tight_layout()

w = randn(2) # initialize weight
w ./= sqrt.(sum(w.^2)) # L2 normalize
initw = copy(w) # save initial weight
η = 1e-3 # learning rate
for _ in 1:200
    y = X' * w  
    w += η * (X * y - y' * y * w) # Oja's rule
end

figure(figsize=(3,3))
scatter(X[1,:], X[2,:], alpha=0.5)
arrow(0,0,initw[1],initw[2], head_width=0.2, color="k", length_includes_head=true, label=L"Init. $w$")
arrow(0,0,w[1],w[2], head_width=0.2, color="tab:red", length_includes_head=true, label=L"Opt. $w$")
plot(cos.(θc), sin.(θc), "k--", alpha=0.8)
xlabel(L"$X_1$"); ylabel(L"$X_2$")
tight_layout()
legend(); tight_layout()

W = randn(2, 2) # initialize weight
W ./= sqrt.(sum(W.^2, dims=2)) # normalize
initW = copy(W) # save initial weight
for _ in 1:200
    Y = W * X
    W += η * (Y * X' - LowerTriangular(Y * Y') * W) # Sanger's rule
end

figure(figsize=(3,3))
scatter(X[1,:], X[2,:], alpha=0.5)
arrow(0, 0, W[1,1], W[1,2], head_width=0.2, color="tab:red", length_includes_head=true, label=L"$w_1$")
arrow(0, 0, W[2,1], W[2,2], head_width=0.2, color="tab:orange", length_includes_head=true, label=L"$w_2$")
plot(cos.(θc), sin.(θc), "k--", alpha=0.8)
xlabel(L"$X_1$"); ylabel(L"$X_2$")
legend(); tight_layout()

function HebbianPCA(X; n_components=10, η=1e-6, maxiter=200, func=identity, orthogonal=true)
    # X : n x m -> Y : n_components x m
    n = size(X)[1]
    X = (X .- mean(X, dims=2)) ./ std(X, dims=2) # normalization
    Y = nothing
    W = randn(n_components, n) # initialize weight
    W ./= sqrt.(sum(W.^2, dims=2)) # normalization
    for _ in 1:maxiter
        Y = func.(W * X)
        if orthogonal
            W .+= η * (Y * X' - LowerTriangular(Y * Y') * W) # Sanger's rule
        else
            W .+= η * (Y * X' - Diagonal(Y * Y') * W) # Oja's rule
        end
    end
    return Y, W
end;

function gaussian2d(center, width, height, step, sigma, scale=1)
    x, y = range(-width/2, width/2, length=step), range(-height/2, height/2, length=step)
    f(x,y) = exp(-((x-center[1])^2 + (y-center[2])^2) / (2.0*scale*(sigma^2)))
    gau = f.(x', y)
    return gau ./ sum(gau)
end

function DoG(center, width=2.2, height=2.2, step=55, sigma=0.12, surround_scale=2)
    g1 = gaussian2d(center, width, height, step, sigma)
    g2 = gaussian2d(center, width, height, step, sigma, surround_scale)
    return g1 - g2
end

sqNp = 32          # 場所細胞の数の平方根: Np=sqNp^2
Ng = 9             # 格子細胞の数
sigma = 0.12       # 場所細胞のtuning curveの幅 [m]
surround_scale = 2 # DoGのσ²の比率
box_width = 2.2    # 箱の横幅 [m]
box_height = 2.2   # 箱の縦幅 [m]
step = 45;         # 空間位置の離散化数

c_eg = zeros(2)
gau_eg = gaussian2d(c_eg, box_width, box_height, step, sigma)
dog_eg = DoG(c_eg, box_width, box_height, step, sigma, surround_scale);

fig, ax = subplots(2,2,figsize=(4,4),sharex="all", sharey="row")
ax[1,1].set_title("Gaussian")
ax[1,1].imshow(gau_eg, cmap="turbo", extent=(-box_width/2, box_width/2, -box_height/2, box_height/2))
ax[1,1].set_ylabel("y [m]")
ax[1,2].set_title("Difference of\n Gaussians (DoG)")
ax[1,2].imshow(dog_eg, cmap="turbo", extent=(-box_width/2, box_width/2, -box_height/2, box_height/2))
x_pos = range(-box_width/2, box_width/2, length=step)
ax[2,1].plot(x_pos, gau_eg[div(step, 2), :]/maximum(gau_eg))
ax[2,1].set_xlabel("x [m]"); ax[2,1].set_ylabel(L"$y=0$"*"の形状 (正規化)")
ax[2,2].plot(x_pos, dog_eg[div(step, 2), :]/maximum(dog_eg))
ax[2,2].set_xlabel("x [m]")
tight_layout()

x_pos = range(-box_width/2, box_width/2, length=sqNp)
y_pos = range(-box_height/2, box_height/2, length=sqNp)
centers = [[i, j] for i in x_pos for j in y_pos]
X_place = hcat([DoG(c, box_width, box_height, step, sigma, surround_scale)[:] for c in centers]...)';

@time Y_pca, W_pca = HebbianPCA(X_place, n_components=Ng, η=1e-5, orthogonal=true)
Y_pca = reshape(Y_pca, (Ng, step, step));

figure(figsize=(3,3.5))
suptitle("次元削減された活動 (PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(Y_pca[i, :, :], cmap="turbo")
    axis("off")
end
tight_layout()

function correlate_fft(x, y)
    corr = fftshift(real(ifft(fft(x) .* conj(fft(y)))))
    return corr / maximum(corr)
end;

corr_pca = [correlate_fft(Y_pca[i, :, :], Y_pca[i, :, :]) for i in 1:Ng];

figure(figsize=(3,3.5))
suptitle("自己相関マップ (PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(corr_pca[i], cmap="turbo")
    axis("off")
end
tight_layout()

relu(x) = max(x, 0)

@time Y_npca, W_npca = HebbianPCA(X_place; n_components=Ng, η=1e-5, maxiter=5000, func=relu, orthogonal=true);
Y_npca = reshape(Y_npca, (Ng, step, step));

figure(figsize=(3,3.5))
suptitle("次元削減された活動 (非負PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(Y_npca[i, :, :], cmap="turbo")
    axis("off")
end
tight_layout()

corr_npca = [correlate_fft(Y_npca[i, :, :], Y_npca[i, :, :]) for i in 1:Ng];

figure(figsize=(3,3.5))
suptitle("自己相関マップ (非負PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(corr_npca[i], cmap="turbo")
    axis("off")
end
tight_layout()
