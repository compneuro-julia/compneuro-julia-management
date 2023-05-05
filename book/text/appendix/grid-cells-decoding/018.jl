activ_hist = fit(Histogram, (posy[idx], posx[idx]), (-50:10:50, -50:10:50)).weights # activation
occup_hist = fit(Histogram, (posy, posx), (-50:10:50, -50:10:50)).weights　# occup position while trajectory 
occup_hist *= 0.02 # one time step is 0.02s 
occup_hist[occup_hist .== 0] .= 1 # avoid devide by zero

rate_hist = activ_hist ./ occup_hist;