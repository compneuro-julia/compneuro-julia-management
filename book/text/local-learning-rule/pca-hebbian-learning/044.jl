figure(figsize=(3,3.5))
suptitle("自己相関マップ (非負PCA)")
for i in 1:Ng
    subplot(3,3,i)
    imshow(corr_npca[i], cmap="turbo")
    axis("off")
end
tight_layout()