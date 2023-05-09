figure(figsize=(8, 4))
for i in 1:2
    subplot(1,2,i)
    title("Wall type: "*wall_types[i])
    xlabel("x (meters)"); ylabel("y (meters)")
    plot(positions[i, 1, 1], positions[i, 1, 2], "ko", label="Start")
    plot(positions[i, end, 1], positions[i, end, 2], "ro", label="Goal")
    plot(positions[i, :, 1], positions[i, :, 2], color="k", alpha=0.3)
end
tight_layout()