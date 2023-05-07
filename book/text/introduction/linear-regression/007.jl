figure(figsize=(5,3.5))
title("Linear Regression with Gradient descent")
scatter(x, y, color="gray", s=10) # samples
plot(xtest, ytest, label="actual")  # regression line
plot(xtest, ŷgd, label="predicted")  # regression line
xlabel("x"); ylabel("y"); legend()
tight_layout()