figure(figsize=(3, 2))
hist(model.r[:], bins=50)
xlim(0, 0.5)
tight_layout()