figure(figsize=(6, 4)) 
idx = [1, 50, 100]
for i in 1:length(idx)
    wh = w_history[idx[i]]
    subplot(2,length(idx),i)
    title("Epoch : "*string(idx[i]))
    
    if i == 1
        ylabel("Weight unfolding\n in data space")
    end
    plot_som(v, wh, vcolor=vcolors);
    subplot(2,length(idx),i+length(idx))
    
    if i == 1
        ylabel("1st dim. weight")
    end
    imshow(reshape(wh[:, 1], (num_w_sqrt, num_w_sqrt)));
end
tight_layout()