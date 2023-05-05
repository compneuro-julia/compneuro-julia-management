# find best matching unit
function find_bmu(v, w)
    N = size(v)[1]
    dims = size(w)[2]
    pos = hcat([[i, j] for i in 1:map_width for j in 1:map_width]...)
    mapped_vpos = zeros(N, dims);
    for i in 1:N
        dist = sum((v[i, :]' .- w).^2, dims=2) # distance between input and neurons
        win_idx = argmin(dist)[1] # winner index
        mapped_vpos[i, :] = pos[:, win_idx]' .- 1
    end
    return mapped_vpos
end