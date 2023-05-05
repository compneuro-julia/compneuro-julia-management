# initial neurons
map_width = 64
M = map_width^2
init_w = product(range(0, 1, length=map_width), range(0, 1, length=map_width))
init_w += (rand(size(init_w)...) .- 1) * 0.05;
init_w = [init_w 2l*(rand(M) .- 0.5) hcat(pol2cart.(4π*(rand(M) .- 0.5), r*rand(M))...)']
#w = reshape(w, (map_width, map_width, dims));