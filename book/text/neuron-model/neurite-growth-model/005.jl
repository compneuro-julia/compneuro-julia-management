function segments_lines(tree_info, seg_vec, init_pos=nothing)
    num_segments = size(tree_info)[1]
    
    pos = zeros(num_segments, 2);
    if init_pos != nothing
        pos[1, :] = init_pos
    end
    
    for j in 1:num_segments
        pos[j, :] = pos[tree_info[j][1], :] + seg_vec[j][1] * [cos(seg_vec[j][2]), sin(seg_vec[j][2])]
    end
    
    lines = []
    for j in 1:num_segments
        x1, y1 = pos[tree_info[j][1], :]
        x2, y2 = pos[j, :]
        push!(lines, [(x1, y1), (x2, y2)])
    end
    return lines, pos
end;