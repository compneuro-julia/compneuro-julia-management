# inputs 
Random.seed!(1234);
σv, σw = 0.1, 0.05
dims = 2  # dims of inputs and neurons
num_v = 300 # num of inputs
num_blobs = 5 # num. cluster of dataset 
num_w_sqrt = 15 # must be int
num_w = num_w_sqrt^2
init_w = σw*randn(num_w, dims);