figure(figsize=(3,3.5))
suptitle("次元削減された活動 (PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(Y_pca[i, :, :], cmap="turbo")
    axis("off")
end
tight_layout()