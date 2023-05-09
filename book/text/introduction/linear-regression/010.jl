figure(figsize=(5,3.5))
title("Linear Regression with Normal equation")
scatter(x, y, color="gray", s=10) # samples
plot(xtest, ytest, label="actual")  # regression line
plot(xtest, ŷne, label="predicted")  # regression line
xlabel("x"); ylabel("y"); legend()
tight_layout()