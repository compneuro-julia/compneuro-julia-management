function Umatrix2d(w)
    M = size(w)[1]
    map_width = Int(sqrt(M))
    pos = hcat([[i, j] for i in 1:map_width for j in 1:map_width]...)
    abs_dist_mat = hcat([sum(abs.(pos .- pos[:, i]), dims=1)' for i in 1:M]...)
    adj_indices = [findall(x -> x == 1, abs_dist_mat[i, :]) for i in 1:M] # adjacent indices
    U = [sqrt(sum((w[adj_indices[i], :] .- w[i, :]') .^2) / size(adj_indices[i])[1]) for i in 1:M]
    U = reshape(U, (map_width, map_width));
    return U
end