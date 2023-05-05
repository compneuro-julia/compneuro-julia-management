N = 200
num_node = 1000
probs = 4*rand(N)/num_node
max_components = zeros(N)
@showprogress for i in 1:N
    G = nx.erdos_renyi_graph(n=num_node, p=probs[i])
    max_components[i] = maximum([length(component) for component in nx.connected_components(G)])
end