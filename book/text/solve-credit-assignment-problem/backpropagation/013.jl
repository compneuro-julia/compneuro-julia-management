function update!(variable::NN, x::Array, y::Array, training::Bool, optimizer::Optimizer=SGD(), losstype::String="binary_crossentropy")
    @unpack n_batch, params, grads = variable 
    W1, W2, b1, b2 = params["W1"], params["W2"], params["b1"], params["b2"]
    
    # feedforward
    h = sigmoid.(x * W1 .+ b1) # hidden
    ŷ = sigmoid.(h * W2 .+ b2) # output
    error = ŷ - y
    
    if training # backward 
        if losstype == "binary_crossentropy"
            δ2 = error 
        elseif losstype == "squared_error"
            δ2 = error .* ŷ .* (1.0 .- ŷ)
        end
        δ1 = δ2 * W2' .* h .* (1.0 .- h)

        # get gradients
        grads["W1"] = x' * δ1
        grads["W2"] = h' * δ2
        grads["b1"] = sum(δ1, dims=1)
        grads["b2"] = sum(δ2, dims=1)

        # update params
        for key in keys(nn.params) 
            optimizer_update!(params[key], grads[key] / n_batch, optimizer)
        end
    end
    return error, ŷ, h
end