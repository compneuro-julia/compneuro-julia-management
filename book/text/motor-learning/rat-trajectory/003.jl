box_width, box_height = 0.8, 0.8; # Width and height of environment (meters)
perimeter_dist = 0.03  # Perimeter region distance to walls (meters)
σv = 0.13              # Forward velocity Rayleigh distribution scale (m/sec)
μω = 0.0               # Rotation velocity Gaussian distribution mean (rad/sec)
σω = (330 / 360) * 2π  # Rotation velocity Gaussian distribution standard deviation (rad/sec)
dt = 0.02              # Simulation-step time increment (seconds)
decel_rate = 0.25;     # velocity reduction factor when located in the perimeter