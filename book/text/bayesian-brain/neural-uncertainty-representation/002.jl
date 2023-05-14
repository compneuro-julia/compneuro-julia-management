Ns = 20 # num. sampling
Np = 10 # num. ppc neurons

μ_dist = -0.5
mixed_gauss = Normal(μ_dist, 0.5) #混合分布
y = pdf.(mixed_gauss, x); # 真の確率密度
samples = rand(mixed_gauss, Ns); # サンプリング
U = kde(samples); # サンプリングのkde

μs = range(-2, 2, length=Np)
tuning_curves = stack([gaussian.(x, μ, 0.5) for μ in μs])';

ppc_fr = exp.(-(μs .- μ_dist).^2/0.5);
act = ppc_fr' * tuning_curves;