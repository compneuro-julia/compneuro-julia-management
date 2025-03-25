Let's derive Backpropagation Through Time (BPTT) step by step for a simple recurrent neural network (RNN). For concreteness, consider an RNN defined by:

1. **Forward Dynamics:**

   - **Hidden state update:**
     \[
     h_t = f\Big(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\Big)
     \]
     where:
     - \( h_t \) is the hidden state at time \( t \),
     - \( x_t \) is the input at time \( t \),
     - \( W_{hh} \) and \( W_{xh} \) are weight matrices,
     - \( b_h \) is a bias vector,
     - \( f(\cdot) \) is an activation function (like \(\tanh\) or sigmoid).

   - **Output computation:**
     \[
     y_t = g\Big(W_{hy}\, h_t + b_y\Big)
     \]
     where:
     - \( y_t \) is the output at time \( t \),
     - \( W_{hy} \) is the weight matrix mapping hidden states to outputs,
     - \( b_y \) is the output bias,
     - \( g(\cdot) \) is an activation function (for instance, softmax for classification).

2. **Loss Function:**

   Assume we have a loss \( L_t \) at each time step (for example, cross-entropy loss for classification). The total loss over a sequence of length \( T \) is:
   \[
   L = \sum_{t=1}^{T} L_t(y_t, \text{target}_t)
   \]

3. **Unrolling the RNN:**

   In BPTT, we “unroll” the RNN over time. This means we view the RNN as a deep feedforward network with \( T \) layers (one per time step) that share the same weights. For example, the computation graph for time steps \( 1, 2, \dots, T \) is:

   \[
   h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T
   \]

   and at each time \( t \) we have an output \( y_t \) and a corresponding loss \( L_t \).

4. **Gradients with Respect to the Output Weights \( W_{hy} \):**

   Since \( y_t = g(W_{hy}\, h_t + b_y) \), the gradient at time \( t \) is computed directly by applying the chain rule:
   \[
   \frac{\partial L_t}{\partial W_{hy}} = \frac{\partial L_t}{\partial y_t}\cdot \frac{\partial y_t}{\partial W_{hy}}
   \]
   and then summed over time:
   \[
   \frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hy}}.
   \]

5. **Gradients with Respect to the Recurrent Weights \( W_{hh} \) and Input Weights \( W_{xh} \):**

   These are more involved because \( h_t \) depends on previous hidden states. Let’s derive the gradients for \( W_{hh} \) (the derivation for \( W_{xh} \) is analogous).

   **a. Define the Pre-activation:**
   \[
   z_t = W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h,
   \]
   so that
   \[
   h_t = f(z_t).
   \]

   **b. Compute the Error Signal at the Output:**

   First, compute the derivative of the loss at time \( t \) with respect to the output pre-activation (if needed). For now, denote:
   \[
   \delta^y_t = \frac{\partial L_t}{\partial y_t} \cdot g'\big(W_{hy}\, h_t + b_y\big).
   \]

   **c. Backpropagate to the Hidden State:**

   The loss at time \( t \) not only depends on \( h_t \) through \( y_t \) but also indirectly influences all future time steps \( t+1, t+2, \dots, T \) because \( h_t \) is an input to the next steps. Define the total error with respect to the hidden state at time \( t \) as:
   \[
   \delta_t \triangleq \frac{\partial L}{\partial h_t}.
   \]
   This error has two components:
   - The direct error from time \( t \): \( W_{hy}^T\, \delta^y_t \).
   - The error propagated from time \( t+1 \): \( \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T \delta_{t+1} \).

   Since
   \[
   h_{t+1} = f\Big(W_{hh}\, h_{t} + W_{xh}\, x_{t+1} + b_h\Big),
   \]
   we have:
   \[
   \frac{\partial h_{t+1}}{\partial h_t} = W_{hh}^T \odot f'\Big(W_{hh}\, h_{t} + W_{xh}\, x_{t+1} + b_h\Big),
   \]
   where the \( \odot \) denotes elementwise multiplication by the derivative \( f' \). Thus, we write the recursive relationship:
   \[
   \delta_t = W_{hy}^T\, \delta^y_t + W_{hh}^T\, \delta_{t+1} \odot f'(z_t).
   \]
   Often, we combine the derivative of the activation into the definition of the local error:
   \[
   \delta_t = \Big(W_{hy}^T\, \delta^y_t + W_{hh}^T\, \delta_{t+1}\Big) \odot f'(z_t).
   \]
   At the final time step \( T \), there is no future dependency, so:
   \[
   \delta_T = \Big(W_{hy}^T\, \delta^y_T\Big) \odot f'(z_T).
   \]

   **d. Compute the Gradients:**

   Now that we have the error signal \( \delta_t \) at each time step for the hidden state pre-activations, we can compute the gradients with respect to the weights:
   
   - **For \( W_{hh} \):**
     \[
     \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^T.
     \]
     This comes from the fact that at each time step:
     \[
     \frac{\partial z_t}{\partial W_{hh}} = h_{t-1}.
     \]
   
   - **For \( W_{xh} \):**
     \[
     \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^T.
     \]

6. **Summary of the BPTT Algorithm:**

   - **Forward Pass:** Unroll the RNN for \( t = 1, 2, \dots, T \) to compute all \( h_t \) and \( y_t \), and then compute the losses \( L_t \).
   
   - **Backward Pass:** For \( t = T \) down to \( 1 \):
     1. Compute the output error:
        \[
        \delta^y_t = \frac{\partial L_t}{\partial y_t}\, g'\big(W_{hy}\, h_t + b_y\big).
        \]
     2. Compute the hidden error recursively:
        \[
        \delta_t = \begin{cases}
        \Big(W_{hy}^T\, \delta^y_t\Big) \odot f'(z_t) & \text{if } t = T, \\
        \Big(W_{hy}^T\, \delta^y_t + W_{hh}^T\, \delta_{t+1}\Big) \odot f'(z_t) & \text{if } t < T.
        \end{cases}
        \]
   
   - **Gradient Accumulation:**
     \[
     \frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \delta^y_t\, h_t^T,
     \]
     \[
     \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^T,
     \]
     \[
     \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^T.
     \]

7. **Remarks:**

   - **Truncation:** In practice, to combat issues like vanishing or exploding gradients, the backpropagation might be truncated to a fixed number of time steps.
   - **Shared Weights:** Remember that the same weights are used at every time step, which is why we sum the gradients over time.

This completes the step-by-step derivation of Backpropagation Through Time (BPTT) for an RNN.