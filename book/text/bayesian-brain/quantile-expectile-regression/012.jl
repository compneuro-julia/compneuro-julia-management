τs = [0.01, 0.1, 0.5, 0.9, 0.99]
m = length(τs) 

# Quantile regression
initθ = zeros(dims)
Ŷq = zeros(m, num_test); # memory array
for i in 1:m
    θq = QuantileGradientDescent(X, y, initθ, τs[i], lr=1e-2, num_iters=1e5)
    Ŷq[i, :] = Xtest * θq
end


# Expectile regression
initθ = zeros(dims)
Ŷe = zeros(m, num_test); # memory array
for i in 1:m
    θe = ExpectileGradientDescent(X, y, initθ, τs[i], lr=1e-2, num_iters=1e5)
    Ŷe[i, :] = Xtest * θe
end