using PyPlot, LinearAlgebra, Random

# Ordinary least squares regression
function OLSRegGradientDescent(X, y, initθ; lr=1e-4, num_iters=10000)
    θ = initθ
    for i in 1:num_iters
        ŷ = X * θ # predictions
        δ = y - ŷ  # error
        θ += lr * X' * δ # Update
    end
    return θ
end;

# Generate Toy datas
N = 500 # sample size
dims = 3 # dimensions
x = sort(randn(N))
y =  x.^2 + 3x + 5x .* randn(N);
X = [ones(N) x x.^2]; # design matrix

# Gradient descent
initθ = zeros(dims) # init variables
θgd = OLSRegGradientDescent(X, y, initθ)
ŷgd = X * θgd # predictions

# Results plot
figure(figsize=(4,3))
title("Linear Regression with Gradient descent")
scatter(x, y, color="gray", s=5) # samples
plot(x, ŷgd, color="tab:red")  # regression line
xlabel("x"); ylabel("y")
tight_layout()

θne = (X' * X) \ X' * y
ŷne = X * θne; # predictions

figure(figsize=(4,3))
title("Linear Regression with Normal equation")
scatter(x, y, color="gray", s=5) # samples
plot(x, ŷne, color="tab:red")  # regression line
xlabel("x"); ylabel("y")
tight_layout()
