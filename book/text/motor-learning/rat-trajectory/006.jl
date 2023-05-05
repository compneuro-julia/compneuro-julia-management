function generate_trajectory(num_steps, wall_type)
    # store arrays
    position, velocity = zeros(num_steps, 2), zeros(num_steps, 2)
    head_dir = zeros(num_steps) # head direction
    speed = rand(Rayleigh(σv), num_steps) # Forward speed
    random_turn = rand(Normal(μω, σω), num_steps) * dt

    # initial values
    head_dir[1] = rand() * 2π
    position[1, :] = (rand(2) .-0.5) .* ([box_width, box_height] .- perimeter_dist)
    
    # iteration of trajectory
    for t in 1:num_steps-1
        turn_angle = random_turn[t]
        dist_wall, angle_wall = min_dist_angle(position[t, :], head_dir[t], wall_type)
        if (dist_wall < perimeter_dist) && (abs(angle_wall) < π/2) # avoid wall
            speed[t] *= decel_rate # deceleration
            turn_angle += sign(angle_wall) * (π/2 - abs(angle_wall))
        end
        velocity[t, :] = speed[t] * [cos(head_dir[t]), sin(head_dir[t])]
        position[t+1, :] = position[t, :] + velocity[t, :] * dt
        head_dir[t+1] = mod(head_dir[t] + turn_angle, 2π) # turn, 
    end
    return position, velocity, speed, head_dir
end;