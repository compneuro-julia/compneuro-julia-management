# 11.3 Predictive coding (Rao & Ballard, 1999) ���f��
Predictive coding�̏��߂̐����I���f���ƂȂ� ([Rao & Ballard, *Nat. Neurosci*. 1999](https://www.nature.com/articles/nn0199_79))����������B

## 11.3.1 �ϑ����E�̊K�w�I�\��
�\�z����l�b�g���[�N�͓��͑w���܂߁A3�w�̃l�b�g���[�N�Ƃ���B�Ԗ��ւ̓��͂Ƃ��ĉ摜 $\mathbf{x} \in \mathbb{R}^{n_0}$���l����B�摜 $\mathbf{x}$ �̊ϑ����E�ɂ�����B��ϐ��A���Ȃ킿**���ݕϐ�** (latent variable)��$\mathbf{r} \in \mathbb{R}^{n_1}$�Ƃ��A�j���[�����Q�ɂ���Ĕ��Η��ŕ\������Ă���Ƃ��� (�^�̕ϐ��� $\mathbf{r}$�͈قȂ�̂ŕ����𕪂���ׂ������ȒP�̂��߂ɂ����\��)�B���̂Ƃ��A

$$
\mathbf{x} = f(\mathbf{U}\mathbf{r}) + \boldsymbol{\epsilon} \tag{1}
$$

���������Ă���Ƃ���B�������A$f(\cdot)$�͊������֐� (activation function)�A$\mathbf{U} \in \mathbb{R}^{n_0 \times n_1}$�͏d�ݍs��ł���B
$\boldsymbol{\epsilon} \in \mathbb{R}^{n_0}$ �� $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ ����T���v�����O�����Ƃ���B

���ݕϐ� $\mathbf{r}$�͂���ɍ��� (higher-level)�̐��ݕϐ� $\mathbf{r}^h$�ɂ��A�����ŕ\�������B

$$
\mathbf{r} = \mathbf{r}^{td}+\boldsymbol{\epsilon}^{td}=f(\mathbf{U}^h \mathbf{r}^h)+\boldsymbol{\epsilon}^{td} \tag{2}
$$

�������ATop-down�̗\���M���� $\mathbf{r}^{td}:=f(\mathbf{U}^h \mathbf{r}^h)$�Ƃ����B�܂��A$\mathbf{r}^{td} \in \mathbb{R}^{n_1}$, $\mathbf{r}^{h} \in \mathbb{R}^{n_2}$, $\mathbf{U}^h \in \mathbb{R}^{n_1 \times n_2}$ �ł���B
$\boldsymbol{\epsilon}^{td} \in \mathbb{R}^{n_1}$��$\mathcal{N}(\mathbf{0}$, $\sigma_{td}^2 \mathbf{I}$) ����T���v�����O�����Ƃ���B

�b�͔�Ԃ��APredictive coding�̃l�b�g���[�N�̓�����
- �K�w�I�ȍ\��
- �����ɂ��᎟�̗\�� (Feedback or Top-down�M��)
- �᎟���獂���ւ̌덷�M���̓`�� (Feedforward or Bottom-up �M��)

�ł���B�����܂ł͍����\���ɂ��᎟�\���̗\���A�Ƃ���Feedback�M���ɂ��Đ������Ă������A���̕�����Sparse coding�ł������ł���B����ł�Predictive coding�̂�����̗v�ƂȂ�A�᎟���獂���ւ̗\���덷�̓`���Ƃ���Feedforward�M���͂ǂ̂悤�ɓ������̂��낤���B���_���猾���΁A�����**�����덷 (reconstruction error)�̍ŏ������s���ċA�I�l�b�g���[�N (recurrent network)���l�����邱�ƂŎ��R�ɓ������**�B

## 11.3.2 �����֐��Ɗw�K��
### ���O���z�̐ݒ�
$\mathbf{r}$�̎��O���z$p(\mathbf{r})$��Cauchy���z��p����B$p(\mathbf{r})$�̕��̑ΐ����O���z��$g(\mathbf{r}):=-\log p(\mathbf{r})$�Ƃ��Ă����B

$$
\begin{align}
p(\mathbf{r})&=\prod_i p(r_i)=\prod_i \exp\left[-\alpha \ln(1+r_i^2)\right]\\
g(\mathbf{r})&=-\ln p(\mathbf{r})=\alpha \sum_i \ln(1+r_i^2)\\
g'(\mathbf{r})&=\frac{\partial g(\mathbf{r})}{\partial \mathbf{r}}=\left[\frac{2\alpha r_i}{1+r_i^2}\right]_i
\end{align}
$$

���ɏd�ݍs��$\mathbf{U}$�̎��O���z $p(\mathbf{U})$��Gaussian���z�Ƃ���B$p(\mathbf{U})$�̕��̑ΐ����O���z��$h(\mathbf{U}):=-\ln p(\mathbf{U})$�Ƃ���ƁA���̂悤�ɕ\�����B

$$
\begin{align}
p(\mathbf{U})&=\exp(-\lambda\|\mathbf{U}\|^2_F)\\
h(\mathbf{U})&=-\ln p(\mathbf{U})=\lambda\|\mathbf{U}\|^2_F\\
h'(\mathbf{U})&=\frac{\partial h(\mathbf{U})}{\partial \mathbf{U}}=2\lambda \mathbf{U}
\end{align}
$$

�������A$\|\cdot \| _ F^2$�̓t���x�j�E�X�m�������Ӗ�����B

### �����֐��̐ݒ�
[11-2](https://compneuro-julia.github.io/11-2_sparse-coding.html)�Ɠ��l�ɍl���邱�Ƃɂ��A�����֐� $E$�����̂悤�ɒ�`����B

$$
\begin{align}
E=\underbrace{\frac{1}{\sigma^{2}}\|\mathbf{x}-f(\mathbf{U} \mathbf{r})\|^2+\frac{1}{\sigma_{t d}^{2}}\left\|\mathbf{r}-f(\mathbf{U}^h \mathbf{r}^h)\right\|^2}_{\text{reconstruction error}}+\underbrace{g(\mathbf{r})+g(\mathbf{r}^{h})+h(\mathbf{U})+h(\mathbf{U}^h)}_{\text{sparsity penalty}}\tag{4}
\end{align}
$$

���ݕϐ� $\mathbf{r}, \mathbf{r}^h$ �� �d�ݍs�� $\mathbf{U}, \mathbf{U}^h$ �̂��ꂼ��Ɏ��O���z�����肵�Ă��邽�߁A�����ɂ��Ă�MAP������s�����Ƃɑ�������B

### �ċA�l�b�g���[�N�̍X�V��
�ȒP�̂��߂�$\mathbf{z}:=\mathbf{U}\mathbf{r}, \mathbf{z}^h:=\mathbf{U}^h\mathbf{r}^h$�Ƃ���B

$$
\begin{align}
\frac{d \mathbf{r}}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}}=k_{1}\cdot\Bigg(\frac{1}{\sigma^{2}} \mathbf{U}^{T}\bigg[\frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}\odot\underbrace{(\mathbf{x}-f(\mathbf{z}))}_{\text{bottom-up error}}\bigg]-\frac{1}{\sigma_{t d}^{2}}\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{top-down error}}-\frac{1}{2}g'(\mathbf{r})\Bigg)\tag{5}\\
\frac{d \mathbf{r}^h}{d t}&=-\frac{k_{1}}{2} \frac{\partial E}{\partial \mathbf{r}^h}=k_{1}\cdot\Bigg(\frac{1}{\sigma_{t d}^{2}}(\mathbf{U}^h)^T\bigg[\frac{\partial f(\mathbf{z}^h)}{\partial \mathbf{z}^h}\odot\underbrace{\left(\mathbf{r}-f(\mathbf{z}^h)\right)}_{\text{bottom-up error}}\bigg]-\frac{1}{2}g'(\mathbf{r}^h)\Bigg)\tag{6}
\end{align}
$$

�������A$k_1$�͍X�V�� (updating rate)�ł���B�܂��́A���Η��̎��萔��$\tau:=1/k_1$�Ƃ��āA$k_1$�͔��Η��̎��萔$\tau$�̋t���ł���Ƒ����邱�Ƃ��ł���B������(5)���ɂ����āA���ԕ\�� $\mathbf{r}$ �̃_�C�i�~�N�X��bottom-up error��top-down error�ŋL�q����Ă���B���̂悤��bottom-up error�� $\mathbf{r}$ �ւ̓��͂ƂȂ邱�Ƃ͎��R�ɓ��o�����B�Ȃ��Atop-down error�Ɋւ��Ă͍�������̗\�� (prediction)�̍� $f(\mathbf{x}^h)$��leaky-integrator�Ƃ��Ă̍� $-\mathbf{r}$�ɕ������邱�Ƃ��ł���B�܂�$\mathbf{U}^T, (\mathbf{U}^h)^T$�͏d�ݍs��̓]�u�ƂȂ��Ă���Abottom-up��top-down�̓��˂ɂ����đΏ̂ȏd�ݍs���p���邱�Ƃ��Ӗ����Ă���B$-g'(\mathbf{r})$�͔��Η���}�����ăX�p�[�X�ɂ��邱�Ƃ�ړI�Ƃ��鍀�����A���������߂�����Ǝ��ȍċA�I�ȗ}���ƌ�����B

### �摜�f�[�^�̓ǂݍ���
11-2�Ɠ��l�Ƀf�[�^��<http://www.rctn.org/bruno/sparsenet/>����_�E�����[�h�ł���t�@�C����p����B

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
using PyPlot

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
�w�K��̏d�ݍs�� ($\mathbf{U}$)���������Ă݂悤�B

# Plot Receptive fields
figure(figsize=(6, 3))
subplots_adjust(hspace=0.1, wspace=0.1)
for i in 1:32
    subplot(4, 8, i)
    imshow(reshape(model.U[:, i], (16, 16)), cmap="gray")
    axis("off")
end
suptitle("Receptive fields of level 1", fontsize=14)
subplots_adjust(top=0.9)

���F��**ON�̈�**(����)�A���F��**OFF�̈�**(�}��)��\���BGabor�t�B���^�l�̋Ǐ���e�삪�����Ă���B���ɁALevel2�̃j���[�����̎�e���$\mathbf{U}$��$\mathbf{U}^h$�̐ς��v�Z���邱�Ƃŕ`��ł���B

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