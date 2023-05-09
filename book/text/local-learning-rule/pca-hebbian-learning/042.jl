figure(figsize=(3,3.5))
suptitle("次元削減された活動 (非負PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(Y_npca[i, :, :], cmap="turbo")
    axis("off")
end
tight_layout()