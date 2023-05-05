figure(figsize=(14, 4))
subplot2grid((3, 3), (0, 0), rowspan=2)
plot(t[1:end-1], whiten(Y[:, end]), label="Estimated slow feature")
plot(t[1:end-1], whiten(sin.(t[1:end-1])), "--", label="True slow feature")
ylabel("SF1"); xlim(0, 2π); legend(loc="upper right");
for i in 1:4
    if i == 1
        subplot2grid((3, 3), (2, 0))
        xlabel("Time");
    else
        subplot2grid((3, 3), (i-2, 1))
    end
    plot(t[1:end-1], whiten(Y[:, end-i]))
    ylabel("SF"*string(i+1)); xlim(0, 2π)
end
xlabel("Time")
tight_layout()