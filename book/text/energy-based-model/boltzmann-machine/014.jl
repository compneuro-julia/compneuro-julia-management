@showprogress "Computing..." for epoch in 1:num_epoch
    for i in 1:num_data
        input = data[i, :]
        h_given_v = sigmoid.(W * input + hbias)
        v = 0.5 * ones(num_v, 1) # init state
        h = 0.5 * ones(num_h, 1) # init state
        sum_v = zeros(num_v, 1)
        sum_h = zeros(num_h, 1)
        outerprod = zeros(num_h, num_v)

        for _ in 1:num_draws
            h = 1.0f0 * (sigmoid.(W * v + hbias) .≥ rand(num_h, 1)) # hidden 
            v = 1.0f0 * (sigmoid.(W' * h + vbias) .≥ rand(num_v, 1)) # visible
            #h = floor.(sigmoid.(W * v + hbias) + rand(num_h, 1)) # hidden 
            #v = floor.(sigmoid.(W' * h + vbias) + rand(num_v, 1)) # visible
            sum_h += h
            sum_v += v
            outerprod += h * v'
        end
            
        sum_h /= num_draws
        sum_v /= num_draws
        outerprod /= num_draws
        
        # update parameters
        W += η * (h_given_v * input' - outerprod)
        hbias += η * (h_given_v - sum_h)
        vbias += η * (input - sum_v)
    end
end