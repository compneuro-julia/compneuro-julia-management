確かに、活性化関数とその微分を個別に管理すると冗長に感じることがあります。これを改善するには、活性化関数とその微分を1つの構造体または関数として定義し、微分を内部で処理する仕組みに変更することで、コードを簡潔にできます。

以下に、その改善方法を示します。

---

### 活性化関数を構造体として管理する例

Juliaでは、関数オーバーロードや構造体を活用してコードをより洗練できます。

```julia
using Random

# 活性化関数構造体
struct ActivationFunction
    forward::Function  # 順伝播時の関数
    backward::Function  # 逆伝播時の関数
end

# 定義済みの活性化関数
sigmoid = ActivationFunction(
    x -> 1.0 ./ (1.0 .+ exp.(-x)),
    y -> y .* (1 .- y)  # y はシグモイド適用後の出力
)

tanh_act = ActivationFunction(
    x -> tanh.(x),
    y -> 1 .- y .^ 2  # y はtanh適用後の出力
)

relu = ActivationFunction(
    x -> max.(0, x),
    y -> y .> 0  # y は活性化関数適用後の出力（原則的にはxだが計算簡略化のため）
)

# フォワードパス
function forward_pass(X, weights, biases, activations)
    layer_outputs = [X]
    for i in 1:length(weights)
        z = layer_outputs[end] * weights[i] .+ biases[i]
        a = activations[i].forward(z)
        push!(layer_outputs, a)
    end
    return layer_outputs
end

# バックプロパゲーション
function backpropagation(X, y, weights, biases, activations, learning_rate)
    layer_outputs = forward_pass(X, weights, biases, activations)
    errors = [layer_outputs[end] .- y]  # 出力層の誤差

    for i in length(weights):-1:2
        delta = (errors[1] * weights[i]') .* activations[i].backward(layer_outputs[i])
        unshift!(errors, delta)
    end

    for i in 1:length(weights)
        weights[i] .-= learning_rate * (layer_outputs[i]' * errors[i])
        biases[i] .-= learning_rate * sum(errors[i], dims=1)
    end

    return sum(abs.(errors[end]))
end

# ネットワークの初期化
function initialize_network(layer_sizes)
    weights = [rand(layer_sizes[i], layer_sizes[i+1]) .- 0.5 for i in 1:length(layer_sizes)-1]
    biases = [rand(1, layer_sizes[i+1]) .- 0.5 for i in 1:length(layer_sizes)-1]
    return weights, biases
end

# メイン関数
function train_neural_network(X, y, layer_sizes, activations, epochs, learning_rate)
    weights, biases = initialize_network(layer_sizes)

    for epoch in 1:epochs
        error = backpropagation(X, y, weights, biases, activations, learning_rate)
        if epoch % 100 == 0
            println("Epoch $epoch, Error: $error")
        end
    end

    return weights, biases
end

# サンプルデータ (XOR問題)
X = [0 0; 0 1; 1 0; 1 1]
y = [0; 1; 1; 0]
y = reshape(y, :, 1)

# パラメータ
layer_sizes = [2, 4, 3, 1]
activations = [relu, tanh_act, sigmoid]  # 各層ごとの活性化関数
epochs = 5000
learning_rate = 0.1

# 学習
println("Training Neural Network...")
trained_weights, trained_biases = train_neural_network(X, y, layer_sizes, activations, epochs, learning_rate)

# テスト
layer_outputs = forward_pass(X, trained_weights, trained_biases, activations)
predictions = layer_outputs[end]
println("Predictions:")
println(predictions)
```

---

### 改善点と利点

1. **簡潔さ**:
   - 活性化関数とその微分を1つの構造体`ActivationFunction`にまとめたため、リストの管理が簡単。

2. **再利用性**:
   - 活性化関数を簡単に追加可能（新しい`ActivationFunction`を定義するだけ）。

3. **柔軟性**:
   - フォワードパスとバックプロパゲーションで一貫して同じインターフェースを使用。

4. **読みやすさ**:
   - 活性化関数の処理がコード内で分散せず、明確。

---

### 使用例

- **すべての層でReLUを使用**:
  ```julia
  activations = [relu for _ in 1:(length(layer_sizes)-1)]
  ```

- **出力層にのみシグモイドを使用**:
  ```julia
  activations = [relu, relu, sigmoid]
  ```

この改良により、活性化関数を扱うコードが洗練され、簡潔かつ柔軟になりました！