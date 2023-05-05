# 単位円上に等間隔にならんだクラスターによるtoy datasetを作成する
function make_blobs(num_samples, num_blobs, dims, σ)
    n = Int(num_samples/num_blobs) # number of samples in each 
    x = vcat([σ*randn(n, dims) .+ [cos(i/num_blobs*2π), sin(i/num_blobs*2π)]' for i in 0:num_blobs-1]...)
    y = repeat(1:num_blobs, inner=n)
    return x, y
end