# inputs 
dims = 2  # dims of inputs 
Random.seed!(1234);
σv, σw = 0.1, 0.05
N = 300 # num of inputs
num_blobs = 5
map_width = 15
M = map_width^2
init_w = σw*randn(M, dims);