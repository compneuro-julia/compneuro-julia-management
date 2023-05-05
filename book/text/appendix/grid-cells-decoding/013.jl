function nearest_pos(array, value)
    idx = argmin(abs.(array .- value))
    return idx
end