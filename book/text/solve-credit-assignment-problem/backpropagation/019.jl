# dataset parameter
θmax = 40.0 # degree, θ∈[-θmax, θmax]
Δθ = 10.0 # degree
stimuli_size = Int(2θmax / Δθ)
w = 15.0 # degree; 1/e width
σ = √2w/(4Δθ);

# training parameter
n_data = 10000
n_traindata = Int(n_data*0.95)
n_batch = 25 # batch size
n_iter_per_epoch = Int(n_traindata/n_batch)
n_epoch = 1000; # number of epoch