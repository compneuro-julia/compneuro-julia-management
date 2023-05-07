T = 300 # simulation time (sec)
num_steps = round(Int, T/dt)
wall_types = ["square", "circle"]
positions = zeros(2, num_steps, 2)
for i in 1:2
    positions[i, :, :], _, _, _ =  generate_trajectory(num_steps, wall_types[i]);
end