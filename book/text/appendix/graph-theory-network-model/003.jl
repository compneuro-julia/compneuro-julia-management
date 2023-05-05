figure(figsize=(12, 2))
for i in 1:5
    subplot(1,5,i)
    G = nx.watts_strogatz_graph(n=50, k=5, p=0.25(i-1))
    nx.draw_circular(G, node_size=10)
    title(L"$p=$"*string(0.25(i-1)))
end
#tight_layout()