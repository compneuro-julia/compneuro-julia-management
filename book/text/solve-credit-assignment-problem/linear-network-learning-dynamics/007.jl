# Set initial values
N₁, N₂, N₃ = 4, 16, 7
Σyx = [ones(4)'; ones(2)' zeros(2)'; zeros(2)' ones(2)'; I(4)];
Σx = I(N₁)
_, s, _ = svd(Σyx);
eps = 1e-2
W₁, W₂ = eps*rand(N₂,N₁), eps*rand(N₃,N₂) # weight for deep
Ws = eps*rand(N₃,N₁) # weight for shallow

#Simulation & training
dt = 0.005
Nt = 1500

# Singular values for shallow, deep
A, B = zeros(Nt, N₁), zeros(Nt, N₁); 