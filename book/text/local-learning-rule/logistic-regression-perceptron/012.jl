n_iter = 100
loss = zeros(n_iter);
W = randn(n, m)
b = randn(n)
for t in 1:n_iter
    ŷ = step.(W * X' .+ b)'
    e = y - ŷ
    loss[t] = sum(e.^2) 
    W[:, :] += 0.1*(1/200) * e' * X
    b[:] .+= 0.1*(1/200) * sum(e)
end