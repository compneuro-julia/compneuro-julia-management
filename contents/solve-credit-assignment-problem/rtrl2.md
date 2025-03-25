Sure! Real-Time Recurrent Learning (RTRL) is an online learning algorithm used to compute the exact gradient of a loss function with respect to the parameters of a recurrent neural network (RNN). It is an alternative to Backpropagation Through Time (BPTT), but unlike BPTT, RTRL updates the gradient at each time step without unfolding the network.

Let's go step by step through the derivation of RTRL.

---

## **1. Define the Recurrent Neural Network (RNN)**
Consider a simple RNN with a hidden state \(\mathbf{h}_t\), an input \(\mathbf{x}_t\), and an output \(\mathbf{y}_t\). The recurrence relation is:

\[
\mathbf{h}_t = f(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t)
\]

where:
- \(\mathbf{h}_t \in \mathbb{R}^n\) is the hidden state at time \(t\),
- \(\mathbf{W} \in \mathbb{R}^{n \times n}\) is the recurrent weight matrix,
- \(\mathbf{U} \in \mathbb{R}^{n \times m}\) is the input weight matrix,
- \(\mathbf{x}_t \in \mathbb{R}^m\) is the input at time \(t\),
- \(f(\cdot)\) is a nonlinear activation function.

The output \(\mathbf{y}_t\) is given by:

\[
\mathbf{y}_t = g(\mathbf{V} \mathbf{h}_t)
\]

where \(\mathbf{V} \in \mathbb{R}^{p \times n}\) is the output weight matrix, and \(g(\cdot)\) is typically a softmax or linear function.

---

## **2. Define the Loss Function**
The loss at time \(t\) is given by:

\[
L_t = \ell(\mathbf{y}_t, \mathbf{\hat{y}}_t)
\]

where \(\mathbf{\hat{y}}_t\) is the target output.

The total loss over a sequence is:

\[
L = \sum_{t=1}^{T} L_t
\]

Our goal is to compute the gradient \(\frac{\partial L}{\partial \mathbf{W}}\) in real time.

---

## **3. Compute the Recursive Gradient Propagation**
To apply gradient descent, we need \(\frac{\partial L_t}{\partial \mathbf{W}}\). Using the chain rule:

\[
\frac{\partial L_t}{\partial \mathbf{W}} = \frac{\partial L_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}
\]

The first term, \(\frac{\partial L_t}{\partial \mathbf{h}_t}\), follows from backpropagation through the output layer:

\[
\mathbf{\delta}_t = \frac{\partial L_t}{\partial \mathbf{h}_t} = \left( \frac{\partial L_t}{\partial \mathbf{y}_t} \right) \frac{\partial \mathbf{y}_t}{\partial \mathbf{h}_t}
\]

where:

\[
\frac{\partial \mathbf{y}_t}{\partial \mathbf{h}_t} = g'(\mathbf{V} \mathbf{h}_t) \mathbf{V}
\]

The second term, \(\frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}\), requires tracking how the hidden state depends on \(\mathbf{W}\) over time.

Using the recurrence relation:

\[
\mathbf{h}_t = f(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t)
\]

we differentiate both sides with respect to \(\mathbf{W}\):

\[
\frac{\partial \mathbf{h}_t}{\partial \mathbf{W}} = f'(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t) \left( \mathbf{W} \frac{\partial \mathbf{h}_{t-1}}{\partial \mathbf{W}} + \frac{\partial (\mathbf{W} \mathbf{h}_{t-1})}{\partial \mathbf{W}} \right)
\]

Since:

\[
\frac{\partial (\mathbf{W} \mathbf{h}_{t-1})}{\partial \mathbf{W}} = \mathbf{h}_{t-1}^T \otimes I
\]

we obtain the recursive update equation:

\[
\frac{\partial \mathbf{h}_t}{\partial \mathbf{W}} = f'(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t) \left( \mathbf{W} \frac{\partial \mathbf{h}_{t-1}}{\partial \mathbf{W}} + \mathbf{h}_{t-1}^T \otimes I \right)
\]

where \(\otimes\) denotes the Kronecker product.

Thus, the **gradient recursion** is:

\[
\mathbf{P}_t = f'(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t) \left( \mathbf{W} \mathbf{P}_{t-1} + \mathbf{h}_{t-1}^T \otimes I \right)
\]

where \(\mathbf{P}_t = \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}\).

---

## **4. Compute the Final Gradient Update**
Finally, the real-time gradient is computed as:

\[
\frac{\partial L_t}{\partial \mathbf{W}} = \mathbf{\delta}_t \mathbf{P}_t
\]

This update is performed **at each time step** in an online fashion, making RTRL different from BPTT.

---

## **5. Summary of RTRL Algorithm**
At each time step \(t\):
1. Compute the hidden state update:

   \[
   \mathbf{h}_t = f(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t)
   \]

2. Compute the output and loss:

   \[
   \mathbf{y}_t = g(\mathbf{V} \mathbf{h}_t), \quad L_t = \ell(\mathbf{y}_t, \mathbf{\hat{y}}_t)
   \]

3. Compute the error term:

   \[
   \mathbf{\delta}_t = \frac{\partial L_t}{\partial \mathbf{h}_t}
   \]

4. Update the gradient propagation matrix:

   \[
   \mathbf{P}_t = f'(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t) \left( \mathbf{W} \mathbf{P}_{t-1} + \mathbf{h}_{t-1}^T \otimes I \right)
   \]

5. Compute the real-time weight gradient:

   \[
   \frac{\partial L_t}{\partial \mathbf{W}} = \mathbf{\delta}_t \mathbf{P}_t
   \]

6. Update the weights:

   \[
   \mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L_t}{\partial \mathbf{W}}
   \]

where \(\eta\) is the learning rate.

---

## **6. Computational Complexity**
The main drawback of RTRL is its high computational cost. Since \(\mathbf{P}_t\) has size \(O(n^3)\), updating it at each time step requires \(O(n^4)\) operations, making RTRL impractical for large networks. This is why Truncated BPTT is often used instead.

However, **modern approximations of RTRL**, such as Unbiased Online Recurrent Optimization (UORO), reduce this cost by using low-rank approximations.

Would you like a specific implementation of RTRL in Julia?

---

```julia
using Random
using LinearAlgebra

# Define activation function and its derivative
function sigmoid(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

function dsigmoid(x)
    s = sigmoid(x)
    return s .* (1 .- s)
end

# Define RTRL-based RNN structure
mutable struct RTRL_RNN
    W::Matrix{Float64}  # Recurrent weight matrix
    U::Matrix{Float64}  # Input weight matrix
    V::Matrix{Float64}  # Output weight matrix
    h::Vector{Float64}  # Hidden state
    P::Array{Float64, 3} # Gradient propagation matrix
end

# Initialize RTRL_RNN
function RTRL_RNN(n::Int, m::Int, p::Int)
    W = randn(n, n) * 0.1
    U = randn(n, m) * 0.1
    V = randn(p, n) * 0.1
    h = zeros(n)
    P = zeros(n, n, n)  # Gradient propagation
    return RTRL_RNN(W, U, V, h, P)
end

# Forward and RTRL gradient update
function forward_rtrl!(rnn::RTRL_RNN, x::Vector{Float64}, target::Vector{Float64}, eta::Float64)
    # Compute hidden state update
    pre_activation = rnn.W * rnn.h + rnn.U * x
    h_new = sigmoid(pre_activation)
    
    # Compute output
    y = rnn.V * h_new
    loss = sum((y - target).^2) / 2
    
    # Compute error term
    delta = rnn.V' * (y - target)
    delta_h = delta .* dsigmoid(pre_activation)
    
    # Update P matrix
    P_new = zeros(size(rnn.P))
    for i in 1:size(rnn.W, 1)
        for j in 1:size(rnn.W, 2)
            P_new[:, i, j] = dsigmoid(pre_activation) .* (rnn.W * rnn.P[:, i, j] + (i == j ? rnn.h : zeros(size(rnn.h))))
        end
    end
    
    # Compute gradient
    dL_dW = zeros(size(rnn.W))
    for i in 1:size(rnn.W, 1)
        for j in 1:size(rnn.W, 2)
            dL_dW[i, j] = sum(delta_h .* P_new[:, i, j])
        end
    end
    
    # Update weights
    rnn.W -= eta * dL_dW
    rnn.U -= eta * (delta_h * x')
    rnn.V -= eta * ((y - target) * h_new')
    
    # Update state
    rnn.h = h_new
    rnn.P = P_new
    
    return loss
end

# Example usage
function train_rtrl()
    Random.seed!(42)
    n, m, p = 5, 3, 2  # Hidden, input, and output dimensions
    rnn = RTRL_RNN(n, m, p)
    
    eta = 0.01
    num_steps = 100
    
    for t in 1:num_steps
        x = rand(m)
        target = rand(p)
        loss = forward_rtrl!(rnn, x, target, eta)
        println("Step ", t, ", Loss: ", loss)
    end
end

train_rtrl()
```