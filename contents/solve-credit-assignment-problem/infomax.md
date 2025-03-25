The **learning rule** for the **InfoMax ICA** algorithm is derived by maximizing the **mutual information** (or equivalently, maximizing the entropy of the output). Below is a step-by-step derivation.

---

## **Step 1: Define the ICA Model**
We assume a **linear mixing model** where the observed signals \(X\) are mixtures of independent sources \( S \):

\[
X = A S
\]

where:
- \( S \) is an unknown vector of independent sources.
- \( A \) is the unknown **mixing matrix**.
- Our goal is to **recover \( S \) from \( X \)** by finding a **demixing matrix** \( W \):

\[
S' = W X
\]

where \( S' \) is an estimate of the true sources \( S \).

---

## **Step 2: InfoMax Principle**
The **InfoMax principle** suggests that maximizing the entropy of a **nonlinear** function of the sources leads to independent components.

Define a **nonlinear activation function** (sigmoid function):

\[
y_i = g(s_i') = g(w_i^T x)
\]

where \( g \) is the **sigmoid (logistic) function**:

\[
g(u) = \frac{1}{1 + e^{-u}}
\]

Since we want to maximize **mutual information**, this is equivalent to **maximizing the likelihood** of the sources.

---

## **Step 3: Log-likelihood Function**
The likelihood of the data given \( W \) is:

\[
p(X | W) = p(S') \left| \det W \right|
\]

Taking the log:

\[
\log p(X | W) = \sum_i \log p(y_i) + \log |\det W|
\]

For **super-Gaussian sources**, we assume \( p(y) \) follows a **sigmoid-like function**, so we approximate:

\[
p(y_i) \propto e^{-H(y_i)}
\]

where \( H(y_i) \) is the entropy of \( y_i \).

Thus, the **log-likelihood function** becomes:

\[
L(W) = \sum_i \sum_n \log g(w_i^T x_n) + \log |\det W|
\]

where the first term comes from maximizing entropy and the second term ensures invertibility.

---

## **Step 4: Gradient Ascent on Log-Likelihood**
To maximize \( L(W) \), we take its derivative with respect to \( W \):

\[
\frac{\partial L}{\partial W} = \sum_n \left[ (1 - 2 Y) X^T \right] + W^{-T}
\]

where:
- \( Y = g(WX) \) is the output after the nonlinearity.
- The term \( (1 - 2 Y) \) comes from the derivative of the sigmoid.

---

## **Step 5: Learning Rule**
Applying a **stochastic gradient ascent update**, we get the **InfoMax learning rule**:

\[
\Delta W \propto (\text{I} + (1 - 2 Y) X^T) W
\]

where \( I \) is the identity matrix.

---

## **Intuition Behind the Learning Rule**
1. **The term \( (1 - 2Y)X^T \) forces the network to decorrelate the sources** by reducing statistical dependencies.
2. **The term \( W \) ensures proper scaling and invertibility.**
3. **Gradient ascent ensures the network adapts \( W \) to maximize the entropy of the output.**

---

### **Final Thoughts**
This derivation follows from **maximum likelihood estimation (MLE)** under **non-Gaussian assumptions**. It is used in EEG, fMRI, and audio signal processing for **blind source separation (BSS)**.

Would you like to see an implementation of this in **Julia**? 🚀