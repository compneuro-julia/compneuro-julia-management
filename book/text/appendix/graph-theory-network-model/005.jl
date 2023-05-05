figure(figsize=(12, 2))
for i in 1:5
    subplot(1,5,i)
    G = nx.erdos_renyi_graph(n=20, p=0.02(i-1))
    pos = nx.spring_layout(G, k=0.4)
    nx.draw(G, pos, node_size=10)
    title(L"$p=$"*string(0.02(i-1)))
end
#tight_layout()