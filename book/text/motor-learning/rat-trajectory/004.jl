function min_dist_angle(pos, head_dir, wall_type="square")
    x, y = pos
    if wall_type == "square"
        dists = [box_width/2-x, box_height/2-y, box_width/2+x, box_height/2+y]
        dist_wall, nearest_wall = findmin(dists)
        angle_wall = mod(head_dir - (nearest_wall-1)*π/2 + π, 2π) - π
    elseif wall_type == "circle"
        dist_wall = box_width/2 - sqrt(x^2 + y^2)
        angle_wall = mod(head_dir - atan(y, x) + π, 2π) - π
    else
        @warn "'wall_type' must be 'square' or 'circle'"
    end 
    return dist_wall, angle_wall
end;