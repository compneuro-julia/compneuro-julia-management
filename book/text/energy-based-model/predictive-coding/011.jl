# Gaussian mask for inputs
function gaussian_2d(sizex=16, sizey=16, sigma=5)
    x, y = 0:sizex-1, 0:sizey-1
    x0, y0 = (sizex-1)/2, (sizey-1)/2
    f(x,y) = exp(-((x-x0)^2 + (y-y0)^2) / (2.0*(sigma^2)))
    gau = f.(x', y)
    return gau ./ sum(gau)
end