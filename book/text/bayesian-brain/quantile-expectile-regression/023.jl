# Results plot
steps = 1:num_steps
figure(figsize=(6,4))
subplot(1,2,1) # 予測価値の変化

for i in 1:num_cells   
    plot(steps, dist_trans[:, i], label=(string((i+1)*25)*"%tile ("*L"$\tau=$"*string((i+1)*0.25)*")"))
end

title("Convergence of value prediction to \n percentile of reward distribution")
xlim(0, num_steps)
ylim(0, 10)
xlabel("Learning steps")
ylabel("Learned Value")
legend()