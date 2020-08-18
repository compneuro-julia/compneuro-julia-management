# 11.3 Predictive coding (Rao & Ballard, 1999) ���f��
Predictive coding�̏��߂̐����I���f���ƂȂ� ([Rao & Ballard, *Nat. Neurosci*. 1999](https://www.nature.com/articles/nn0199_79))����������B

## 11.3.1 �ϑ����E�̊K�w�I�\��
�\�z����l�b�g���[�N�͓��͑w���܂߁A3�w�̃l�b�g���[�N�Ƃ���B�Ԗ��ւ̓��͂Ƃ��ĉ摜 $\boldsymbol{I} \in \mathbb{R}^{n_0}$���l����B�摜 $\boldsymbol{I}$ �̊ϑ����E�ɂ�����B��ϐ��A���Ȃ킿**���ݕϐ�** (latent variable)��$\boldsymbol{r} \in \mathbb{R}^{n_1}$�Ƃ��A�j���[�����Q�ɂ���Ĕ��Η��ŕ\������Ă���Ƃ��� (�^�̕ϐ��� $\boldsymbol{r}$�͈قȂ�̂ŕ����𕪂���ׂ������ȒP�̂��߂ɂ����\��)�B���̂Ƃ��A

$$
\boldsymbol{I} = f(U\boldsymbol{r}) + \boldsymbol{n} \tag{1}
$$

���������Ă���Ƃ���B�������A$f(\cdot)$�͊������֐� (activation function)�A$U \in \mathbb{R}^{n_0 \times n_1}$�͏d�ݍs��ł���B$\boldsymbol{n} \in \mathbb{R}^{n_0} $�͕���0, ���U $\sigma^2$ ��Gaussian �m�C�Y���Ƃ���B

���ݕϐ� $\boldsymbol{r}$�͂���ɍ��� (higher-level)�̐��ݕϐ� $\boldsymbol{r}^h$�ɂ��A�����ŕ\�������B

$$
\boldsymbol{r} = \boldsymbol{r}^{td}+\boldsymbol{n}^{td}=f(U^h \boldsymbol{r}^h)+\boldsymbol{n}^{td} \tag{2}
$$

�������ATop-down�̗\���M���� $\boldsymbol{r}^{td}:=f(U^h \boldsymbol{r}^h)$�Ƃ����B�܂��A$\boldsymbol{r}^{td} \in \mathbb{R}^{n_1}$, $\boldsymbol{r}^{h} \in \mathbb{R}^{n_2}$, $U^h \in \mathbb{R}^{n_1 \times n_2}$ �ł���B $\boldsymbol{n}^{td} \in \mathbb{R}^{n_1} $�͕���0, ���U $\sigma_{td}^2$ ��Gaussian �m�C�Y���Ƃ���B

�b�͔�Ԃ��APredictive coding�̃l�b�g���[�N�̓�����
- �K�w�I�ȍ\��
- �����ɂ��᎟�̗\�� (Feedback or Top-down�M��)
- �᎟���獂���ւ̌덷�M���̓`�� (Feedforward or Bottom-up �M��)

�ł���B�����܂ł͍����\���ɂ��᎟�\���̗\���A�Ƃ���Feedback�M���ɂ��Đ������Ă������A���̕�����Sparse coding�ł������ł���B����ł�Predictive coding�̂�����̗v�ƂȂ�A�᎟���獂���ւ̗\���덷�̓`���Ƃ���Feedforward�M���͂ǂ̂悤�ɓ������̂��낤���B���_���猾���΁A�����**�����덷 (reconstruction error)�̍ŏ������s���ċA�I�l�b�g���[�N (recurrent network)���l�����邱�ƂŎ��R�ɓ������**�B

## 11.3.2 �����֐��Ɗw�K��
### �����֐��̐ݒ�
�O�߂ł�2�w�܂ł̃p�����[�^���œK�����邱�Ƃ��l���܂����A�����̊������l�����đ����֐� $E$�����̂悤�ɍĒ�`����B

$$
\begin{align}
E=\underbrace{\frac{1}{\sigma^{2}}\|\boldsymbol{I}-f(U \boldsymbol{r})\|^2+\frac{1}{\sigma_{t d}^{2}}\left\|\boldsymbol{r}-f(U^h \boldsymbol{r}^h)\right\|^2}_{\text{reconstruction error}}+\underbrace{g(\boldsymbol{r})+g(\boldsymbol{r}^{h})+h(U)+h(U^h)}_{\text{sparsity penalty}}\tag{8}
\end{align}
$$


### �ċA�l�b�g���[�N�̍X�V��
�ȒP�̂��߂�$\boldsymbol{x}:=U\boldsymbol{r}, \boldsymbol{x}^h:=U^h\boldsymbol{r}^h$�Ƃ���B

$$
\begin{align}
\frac{d \boldsymbol{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \boldsymbol{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} U^{T}\bigg[\frac{\partial f(\boldsymbol{x})}{\partial \boldsymbol{x}}\odot\underbrace{(\boldsymbol{I}-f(\boldsymbol{x}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\boldsymbol{r}-f(\boldsymbol{x}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\boldsymbol{r})\Bigg)\tag{9}\\
\frac{d \boldsymbol{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \boldsymbol{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(U^h)^T\bigg[\frac{\partial f(\boldsymbol{x}^h)}{\partial \boldsymbol{x}^h}\odot\underbrace{\left(\boldsymbol{r}-f(\boldsymbol{x}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\boldsymbol{r}^h)\Bigg)\tag{10}
\end{align}
$$

�������A$k_1$�͍X�V�� (updating rate)�ł���B�܂��́A���Η��̎��萔��$\tau:=1/k_1$�Ƃ��āA$k_1$�͔��Η��̎��萔$\tau$�̋t���ł���Ƒ����邱�Ƃ��ł���B������(9)���ɂ����āA���ԕ\�� $\boldsymbol{r}$ �̃_�C�i�~�N�X��bottom-up error��top-down error�ŋL�q����Ă���B���̂悤��bottom-up error�� $\boldsymbol{r}$ �ւ̓��͂ƂȂ邱�Ƃ͎��R�ɓ��o�����B�Ȃ��Atop-down error�Ɋւ��Ă͍�������̗\�� (prediction)�̍� $f(\boldsymbol{x}^h)$��leaky-integrator�Ƃ��Ă̍� $-\boldsymbol{r}$�ɕ������邱�Ƃ��ł���B�܂�$U^T, (U^h)^T$�͏d�ݍs��̓]�u�ƂȂ��Ă���Abottom-up��top-down�̓��˂ɂ����đΏ̂ȏd�ݍs���p���邱�Ƃ��Ӗ����Ă���B$-g'(\boldsymbol{r})$�͔��Η���}�����ăX�p�[�X�ɂ��邱�Ƃ�ړI�Ƃ��鍀�����A���������߂�����Ǝ��ȍċA�I�ȗ}���ƌ�����B

using PyPlot

### �摜�f�[�^�̓ǂݍ���
�f�[�^��<http://www.rctn.org/bruno/sparsenet/>����_�E�����[�h�ł���B`IMAGES_RAW.mat`��10���̎��R�摜�ŁA`IMAGES.mat`�͂���𔒐F���������̂ł���B`mat`�t�@�C���̓ǂݍ��݂ɂ�[MAT.jl](https://github.com/JuliaIO/MAT.jl)��p����B

using MAT

# datasets from http://www.rctn.org/bruno/sparsenet/
mat_images = matopen("_static/datasets/IMAGES.mat")
imgs = read(mat_images, "IMAGES")

close(mat_images)

### ���f���̒�`
�K�v�ȃp�b�P�[�W��ǂݍ��ށB

using Base: @kwdef
using Parameters: @unpack # or using UnPack
using LinearAlgebra
using Random
using Statistics
using ProgressMeter

���f�����`����B

@kwdef struct RBParameter{FT}
    ��::FT = 1.0
    ��h::FT = 0.05
    var::FT = 1.0
    vartd::FT = 10
    inv_var::FT = 1/var       
    inv_vartd::FT = 1/vartd
    k1::FT = 0.3 # k_1: update rate
    ��::FT = 0.02 # regularization parameter
end

@kwdef mutable struct RaoBallard1999Model{FT}
    param::RBParameter = RBParameter{FT}()
    num_units_lv0::UInt16 = 256 # number of units of level0
    num_units_lv1::UInt16 = 32
    num_units_lv2::UInt16 = 128
    num_lv1::UInt16 = 3
    k2::FT = 0.2 # k_2: learning rate
    r::Array{FT} = zeros(num_lv1, num_units_lv1) # activity of neurons
    rh::Array{FT} = zeros(num_units_lv2) # activity of neurons
    U::Array{FT} = randn(num_units_lv0, num_units_lv1) .* sqrt(2.0 / (num_units_lv0+num_units_lv1))
    Uh::Array{FT} = randn(num_lv1*num_units_lv1, num_units_lv2) .* sqrt(2.0 / (num_lv1*num_units_lv1+num_units_lv2))
end

�p�����[�^���X�V����֐����`����B

function update!(variable::RaoBallard1999Model, param::RBParameter, inputs::Array, training::Bool)
    @unpack num_units_lv0, num_units_lv1, num_units_lv2, num_lv1, k2, r, rh, U, Uh = variable
    @unpack ��, ��h, var, vartd, inv_var, inv_vartd, k1, �� = param

    r_reshaped = r[:] # (96)

    fx = r * U' # (3, 256)
    fxh = Uh * rh # (96, )

    # Calculate errors
    error = inputs - fx # (3, 256)
    errorh = r_reshaped - fxh # (96, ) 
    errorh_reshaped = reshape(errorh, (num_lv1, num_units_lv1)) # (3, 32)

    g_r = �� * r ./ (1.0 .+ r .^ 2) # (3, 32)
    g_rh = ��h * rh ./ (1.0 .+ rh .^ 2) # (64, )

    # Update r and rh
    dr = k1 * (inv_var * error * U - inv_vartd * errorh_reshaped - g_r)
    drh = k1 * (inv_vartd * Uh' * errorh - g_rh)
    
    r[:, :] += dr
    rh[:] += drh
    
    if training 
        U[:, :] += k2 * (inv_var * error' * r - num_lv1 * �� * U)
        Uh[:, :] += k2 * (inv_vartd * errorh * rh' - �� * Uh)
    end

    return error, errorh, dr, drh
end

���͂ɏ悶��Gaussian�t�B���^���`����B

# Gaussian mask for inputs
function GaussianMask(sizex=16, sizey=16, sigma=5)
    x = 0:sizex-1
    y = 0:sizey-1
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    
    x0 = (sizex-1) / 2
    y0 = (sizey-1) / 2
    mask = exp.(-((X .- x0) .^2 + (Y .- y0) .^2) / (2.0*(sigma^2)))
    return mask ./ sum(mask)
end

gau = GaussianMask()
figure(figsize=(2,2))
title("Gaussian mask")
imshow(gau)
tight_layout()

�����֐����`����B

function calculate_total_error(error, errorh, variable::RaoBallard1999Model, param::RBParameter)
    @unpack num_units_lv0, num_units_lv1, num_units_lv2, num_lv1, k2, r, rh, U, Uh = variable
    @unpack ��, ��h, var, vartd, inv_var, inv_vartd, k1, �� = param
    recon_error = inv_var * sum(error.^2) + inv_vartd * sum(errorh.^2)
    sparsity_r = �� * sum(r.^2) + ��h * sum(rh.^2)
    sparsity_U = �� * (sum(U.^2) + sum(Uh.^2))
    return recon_error + sparsity_r + sparsity_U
end

�V�~�����[�V���������s����֐����`����B�O����`for loop`�ł͉摜�p�b�`�̍쐬��`r`�̏��������s���B������`for loop`�ł�`r`����������܂ōX�V���s���A���������Ƃ��ɏd�ݍs��`Phi`���X�V����B

function run_simulation(imgs, num_iter, nt_max, eps)
    # Define model
    model = RaoBallard1999Model{Float32}()
    
    # Simulation constants
    H, W, num_images = size(imgs)
    input_scale = 40 # scale factor of inputs
    gmask = GaussianMask() # Gaussian mask
    errorarr = zeros(num_iter) # Vector to save errors    
    
    # Run simulation
    @showprogress "Computing..." for iter in 1:num_iter
        # Get images randomly
        idx = rand(1:num_images)
        img = imgs[:, :, idx]

        # Get the coordinates of the upper left corner of clopping image randomly.
        beginx = rand(1:W-27)
        beginy = rand(1:H-17)
        img_clopped = img[beginy:beginy+15, beginx:beginx+25]

        # Clop three patches
        inputs = hcat([(gmask .* img_clopped[:, 1+i*5:i*5+16])[:] for i = 0:2]...)'
        inputs = (inputs .- mean(inputs)) .* input_scale

        # Reset states
        model.r = inputs * model.U 
        model.rh = model.Uh' * model.r[:]

        # Input an image patch until latent variables are converged 
        for i in 1:nt_max
            # Update r and rh without update weights 
            error, errorh, dr, drh = update!(model, model.param, inputs, false)

            # Compute norm of r and rh
            dr_norm = sqrt(sum(dr.^2))
            drh_norm = sqrt(sum(drh.^2))

            # Check convergence of r and rh, then update weights
            if dr_norm < eps && drh_norm < eps
                error, errorh, dr, drh = update!(model, model.param, inputs, true)
                errorarr[iter] = calculate_total_error(error, errorh, model, model.param) # Append errors
                break
            end

            # If failure to convergence, break and print error
            if i >= nt_max-2
                println("Error at patch:", iter)
                println(dr_norm, drh_norm)
                break
            end
        end


        # Decay learning rate         
        if iter % 40 == 39
            model.k2 /= 1.015
        end

        # Print moving average error
        if iter % 1000 == 0
            moving_average_error = mean(errorarr[iter-999:iter])
            println("iter: ", iter, "/", num_iter, ", Moving average error:", moving_average_error)
        end
    end
    return model, errorarr
end

### �V�~�����[�V�����̎��s

# Simulation constants
num_iter = 5000 # number of iterations
nt_max = 1000 # Maximum number of simulation time
eps = 1e-3 # small value which determines convergence

model, errorarr = run_simulation(imgs, num_iter, nt_max, eps)

### �P�����̑����̕`��
�P�����̑����̕ω���`�悵�Ă݂悤�B�������ቺ���A�w�K���i�s�������Ƃ�������B

function moving_average(x, n=100)
    ret = cumsum(x)
    ret[n:end] = ret[n:end] - ret[1:end-n+1]
    return ret[n - 1:end] / n
end

# Plot error
moving_average_error = moving_average(errorarr)
figure(figsize=(4, 2))
ylabel("Moving error")
xlabel("Iterations")
plot(1:size(moving_average_error)[1], moving_average_error)
tight_layout()

### �d�ݍs�� (��e��)�̕`��
�w�K��̏d�ݍs�� `Phi` ($\Phi$)���������Ă݂悤�B

# Plot Receptive fields
figure(figsize=(6, 3))
subplots_adjust(hspace=0.1, wspace=0.1)
for i in 1:32
    subplot(4, 8, i)
    imshow(reshape(model.U[:, i], (16, 16)), cmap="gray")
    axis("off")
end
suptitle("Receptive fields", fontsize=14)
subplots_adjust(top=0.9)

���F��**ON�̈�**(����)�A���F��**OFF�̈�**(�}��)��\���BGabor�t�B���^�l�̋Ǐ���e�삪�����Ă���A����͈ꎟ���o��(V1)�ɂ�����P���^�זE(simple cells)�̎�e��ɗގ����Ă���B

# Plot Receptive fields of level 2
zero_padding = zeros(80, 32)
U0 = [model.U; zero_padding; zero_padding]
U1 = [zero_padding; model.U; zero_padding]
U2 = [zero_padding; zero_padding; model.U]
U_ = [U0 U1 U2]
Uh_ = U_ * model.Uh 

figure(figsize=(7, 3))
subplots_adjust(hspace=0.1, wspace=0.1)
for i in 1:24
    subplot(4, 6, i)
    imshow(reshape(Uh_[:, i], (16, 26)), cmap="gray")
    axis("off")
end

suptitle("Receptive fields of level 2", fontsize=14)
subplots_adjust(top=0.9)