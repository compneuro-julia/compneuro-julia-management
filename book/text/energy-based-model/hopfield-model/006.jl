testimagelist = ["cameraman", "jetplane", "house", "mandril_gray", "lake_gray"]; # gray & size(512 x 512)
num_data = length(testimagelist)

imgs = [convert(Array{Float64}, imresize(Gray.(testimage(imagename)), ratio=1/8)) for imagename in testimagelist];
imgs_binarized = map(binarize, imgs);
imgs_corrupted = map(corrupted, imgs_binarized);

input_train = reshape(stack(imgs_binarized), (:, num_data))';
input_test = reshape(stack(imgs_corrupted), (:, num_data))';