figure(figsize=(8, 3.5))

subplot2grid((3,3), (0,0), rowspan=3)
imshow(value); colorbar()
title("Value"); ylabel("Trial"); xlabel("Time (s)"); xticks(0:Int(1/dt):nt, -1:1:T-1)

subplot2grid((3,3), (0,1), rowspan=3)
imshow(delta); colorbar()
title("TD error"); ylabel("Trial"); xlabel("Time (s)"); xticks(0:Int(1/dt):nt, -1:1:T-1)

subplot2grid((3,3), (0,2))
plot(-1:0.1:2, delta[6, :]); title("No CS + R (Trial #6)"); xticks([])

subplot2grid((3,3), (1,2))
plot(-1:0.1:2, delta[30, :]); title("CS + R (Trial #30)"); ylabel("TD error"); xticks([])

subplot2grid((3,3), (2,2))
plot(-1:0.1:2, delta[41, :]); title("CS + No R (Trial #41)"); xlabel("Time (s)")

tight_layout()