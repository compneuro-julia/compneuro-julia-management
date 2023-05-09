error_arr = zeros(Float32, n_epoch); # memory array of each epoch error

@showprogress "Training..." for e in 1:n_epoch
    for iter in 1:n_iter_per_epoch
        idx = (iter-1)*n_batch+1:iter*n_batch
        error, _, _ = update!(nn, x_traindata[idx, :], y_traindata[idx, :], true, optimizer, losstype)
        error_arr[e] += sum(error .^ 2)
    end 
    error_arr[e] /= n_traindata
end