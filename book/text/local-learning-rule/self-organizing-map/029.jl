# initial neurons
num_w_sqrt = 64
num_w = num_w_sqrt^2
init_w = product(range(0, 1, length=num_w_sqrt), range(0, 1, length=num_w_sqrt))
init_w += (rand(size(init_w)...) .- 1) * 0.05;
init_w = [init_w 2l*(rand(num_w) .- 0.5) hcat(pol2cart.(4π*(rand(num_w) .- 0.5), r*rand(num_w))...)']
#w = reshape(w, (num_w_sqrt, num_w_sqrt, dims));