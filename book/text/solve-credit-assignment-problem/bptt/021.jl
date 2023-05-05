error, ŷ, h = update!(rnn, rnn.param, x, y, false)
println("Error : ", sum(error.^2))