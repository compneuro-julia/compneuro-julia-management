using LinearAlgebra, Random

struct ActivationFunction
    forward::Function   # function for forward propagation
    backward::Function  # function for back-propagation
end

(f::ActivationFunction)(x) = f.forward(x)

abstract type NeuralNet end
(f::NeuralNet)(x) = forward!(f, x)