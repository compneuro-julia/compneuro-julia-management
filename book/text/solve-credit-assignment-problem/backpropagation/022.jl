# model parameter
n_in = stimuli_size^2 + 2 # number of inputs
n_hid = 16   # number of hidden units
n_out = stimuli_size^2   # number of outputs
η = 1e-2  # learning rate
losstype = "binary_crossentropy" # "squared_error"

nn = NN(n_batch, n_in, n_hid, n_out);
optimizer = SGD(η=η); 
#optimizer = Adam();