x_pos = range(-box_width/2, box_width/2, length=sqNp)
y_pos = range(-box_height/2, box_height/2, length=sqNp)
centers = [[i, j] for i in x_pos for j in y_pos]
X_place = hcat([DoG(c, box_width, box_height, step, sigma, surround_scale)[:] for c in centers]...)';