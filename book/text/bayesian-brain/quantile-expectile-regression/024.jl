# Gaussian kernel density estimation
function kde(data, dx=0.1, band_width=1)
    x = minimum(data):dx:maximum(data)
    y = zero(x)
    n = size(data)[1]
    for i in 1:n
        y += exp.(-(((x .- data[i])/band_width).^2)/2)
    end
    y /= (n*band_width*sqrt(2π))
    return x, y
end