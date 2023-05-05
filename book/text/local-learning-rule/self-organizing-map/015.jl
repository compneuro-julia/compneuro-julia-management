figure(figsize=(6, 4)) 
idx = [1, 50, 100]
for i in 1:length(idx)
    wh = w_history[idx[i]]
    subplot(2,length(idx),i)
    title("Epoch : "*string(idx[i]))
    plot_som(v, wh, vcolor=vcolors); axis("off")
    subplot(2,length(idx),i+length(idx))
    imshow(reshape(wh[:, 1], (map_width, map_width))); axis("off")
end
tight_layout()