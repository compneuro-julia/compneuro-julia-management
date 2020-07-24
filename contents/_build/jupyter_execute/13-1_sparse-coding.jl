# 13.1 Sparse coding (Olshausen & Field, 1996) ���f��
���o�ɂ�����Sparse coding�̃��f�� ([Olshausen & Field, *Nature*. 1996](https://www.nature.com/articles/381607a0))�̎�����ڕW�Ƃ���B

```{note}
Sparse coding���ǂ��ɓ���邩�͖��������A�������f���ł͂���̂ł��̏͂ɓ��ꂽ�B
```

## 13.1.1 �摜��sparse coding

�摜 $\boldsymbol{I} (\boldsymbol{x})$�����֐�(basis function) $\Phi = [\phi_i(\boldsymbol{x})]$ �̐��`�a�ŕ\�����Ƃ���B

$$
\boldsymbol{I}(\boldsymbol{x}) = \sum_i r_i \phi_i (\boldsymbol{x}) + \epsilon(\boldsymbol{x})= \Phi \boldsymbol{r} + \epsilon(\boldsymbol{x})
$$

�������A$\boldsymbol{x}$�͉摜��̍��W, $\epsilon$�͕���0��Gaussian�m�C�Y��\���B�܂��A$\boldsymbol{r}$�͌W���ł��邪�A���f���ɂ����Ă͓��͂��������̐_�o�זE�̊����Ƃ݂Ȃ��B���̏ꍇ�A$\Phi$�͏d�ݍs��ƂȂ�B

Sparse coding�́A�����̊��ŉ摜 (��ړI�ϐ�)��\�����Ƃ�ړI�Ƃ���B�v��(1)���ɂ����āA�قƂ�ǂ�0�ŁA�ꕔ����0�ȊO�̒l�����Ƃ����a (=sparse)�ȌW��$\boldsymbol{r}$�����߂����B

## 13.1.2 �ړI�֐��̐ݒ�
Sparse coding�̂��߂̖ړI�֐�(cost function) $E$��(2)���̂悤�ɂȂ�B

$$
E = \underbrace{\left\|\boldsymbol{I}-\Phi \boldsymbol{r}\right\|^2}_{\text{preserve information}} + \lambda \underbrace{\sum_i S\left(\frac{r_i}{\sigma}\right)}_{\text{sparseness of}\ r_i}
$$

�������A$\lambda$�͐������W���A$\sigma$�͒萔(scaling constant)�ł���B�����ŁA��ꍀ�����������A��񍀂������� (�W�����傫�Ȓl�ƂȂ�Ȃ��悤�ɂ��鍀)�ƂȂ��Ă���B

$S(x)$�Ƃ��Ă� $-\exp(-x^2), \ln(1+x^2), |x|$ �Ȃǂ̊֐����p������B�����̊֐��͌��_�ɂ����Đ�����`������Ă���A����0�ɂȂ�₷���Ȃ��Ă���B

##  13.1.3 Locally Competitive Algorithm (LCA) 
$\boldsymbol{r}$�̌��z�@�ɂ��X�V���́A�ړI�֐� $E$�̔����ɂ�莟�̂悤�ɓ�����B

$$
\begin{align}
\frac{d \boldsymbol{r}}{dt} &= -\frac{\eta_\boldsymbol{r}}{2}\frac{\partial E}{\partial \boldsymbol{r}}\\
&=\eta_\boldsymbol{r} \cdot\left[\Phi^T (\boldsymbol{I}-\Phi\boldsymbol{r})- \frac{\lambda}{2\sigma}S'\left(\frac{r_i}{\sigma}\right)\right]
\end{align}
$$

�������A$\eta_{\boldsymbol{r}}$�͊w�K���ł���B���̎��ɂ��$\boldsymbol{r}$����������܂ōœK�����邪�A�P�Ȃ���z�@�ł͂Ȃ��A(Olshausen & Field, 1996)�ł�**�������z�@** (conjugate gradient method)��p���Ă���B�������A�������z�@�͎������ώG�Ŕ�����ł��邽�߁A�������I�������w�I�ȑÓ����̍����w�K�@�Ƃ��āA**LCA**  (locally competitive algorithm)����Ă���Ă��� ([Rozell et al., *Neural Comput*. 2008](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf))�BLCA��**���}��** (local competition, lateral inhibition)��**臒l�֐�** (thresholding function)��p����X�V���ł���BLCA�ɂ��X�V���s��RNN�͒ʏ��RNN�Ƃ͈قȂ�A�R�X�g�֐�(�܂��̓G�l���M�[�֐�)���ŏ������铮�I�V�X�e���ł���B���̂悤�ȋ@�\��Hopfield network�ŗp�����Ă��邽�߂ɁAOlshausen��**Hopfield trick**�ƌĂ�ł���B

### ���臒l�֐���p����ꍇ (ISTA)
$S(x)=|x|$�Ƃ����ꍇ��臒l�֐���p�����@�Ƃ���**ISTA**(Iterative Shrinkage Thresholding Algorithm)������BISTA��L1-norm���������ɑ΂���ߐڌ��z�@�ŁA�v��Lasso��A�ɗp������z�@�ł���B

�����ׂ����͎����ŕ\�����B

$$
\boldsymbol{r} = \mathop{\rm arg~min}\limits_{\boldsymbol{r}}\left\{\|\boldsymbol{I}-\Phi\boldsymbol{r}\|^2_2+\lambda\|\boldsymbol{r}\|_1\right\}
$$

�ڍׂ͌�q���邪�A���̂悤�ɍX�V���邱�Ƃŉ���������B

1. $\boldsymbol{r}(0)$��v�f���S��0�̃x�N�g���ŏ�����
2. $\boldsymbol{r}_*(t+1)=\boldsymbol{r}(t)+\eta_\boldsymbol{r}\cdot \Phi^T(\boldsymbol{I}-\Phi\boldsymbol{r}(t))$
3. $\boldsymbol{r}(t+1) = S_\lambda(\boldsymbol{r}_*(t+1))$
4. $\boldsymbol{r}$����������܂�2��3���J��Ԃ�

������$S_\lambda(\cdot)$��**���臒l�֐�** (Soft thresholding function)�ƌĂ΂�A�����ŕ\�����B

$$
S_\lambda(y)= 
\begin{cases} 
y-\lambda & (y>\lambda)\\ 
0 & (-\lambda\leq y\leq\lambda)\\ 
 y+\lambda & (y<-\lambda) 
\end{cases}
$$

$S_\lambda(\cdot)$���֐��Ƃ��Ē�`����Ǝ��̂悤�ɂȂ�B

# thresholding function of S(x)=|x|
function soft_thresholding_func(x, lmda)
    max(x - lmda, 0) - max(-x - lmda, 0)
end

����$S_\lambda(\cdot)$��`�悷��Ǝ��̂悤�ɂȂ�B�������A���`PyPlot`��ǂݍ���ł����B

using PyPlot

x = range(-5, 5, length=100)
y = soft_thresholding_func.(x, 1)


figure(figsize=(5,4))
plot(x, x, "k--", label="y=x")
plot(x, y, label="Soft thresholding (lambda=1)")
xlabel("x")
ylabel("S (x)")
legend()

�Ȃ��ASoft thresholding�֐��͎��̖ړI�֐�$C$���ŏ�������$x$�����߂邱�Ƃœ��o�ł���B

$$
C=\frac{1}{2}(y-x)^2+\lambda |x|
$$

�������A$x, y, \lambda$�̓X�J���[�l�Ƃ���B$|x|$�������ł��Ȃ����A����͏ꍇ�������l���邱�Ƃŉ�������B$x\geq 0$���l����ƁA(6)����

$$
C=\frac{1}{2}(y-x)^2+\lambda x = \{x-(y-\lambda)\}^2+\lambda(y-\lambda)
$$

�ƂȂ�B(7)���̍ŏ��l��^����$x$�͏ꍇ���������čl����ƁA$y-\lambda\geq0$�̂Ƃ��񎟊֐��̒��_���l����$x=y-\lambda$�ƂȂ�B �����$y-\lambda<0$�̂Ƃ���$x\geq0$�ɂ����ĒP�������Ȋ֐��ƂȂ�̂ŁA�ŏ��ƂȂ�̂�$x=0$�̂Ƃ��ł���B���l�̋c�_��$x\leq0$�ɑ΂��Ă��s�����Ƃ� (5)����������B

## 13.1.4 �d�ݍs��̍X�V��
$\boldsymbol{r}$��������������z�@�ɂ��$\Phi$���X�V����B

$$
\begin{aligned}
\Delta \phi_i(\boldsymbol{x}) &= -\eta \frac{\partial E}{\partial \Phi}\\
&=\eta\cdot\left[\left([\boldsymbol{I}-\Phi\boldsymbol{r}\right)\boldsymbol{r}^T\right]
\end{aligned}
$$

## 13.1.5 Sparse coding network�̎���
�l�b�g���[�N�͓��͑w���܂�2�w�̒P���ȍ\���ł���B����́A���͂̓����_���ɐ؂�o����16�~16 (��256)�̉摜�p�b�`�Ƃ��A�������͑w��256�̃j���[�������󂯎��Ƃ���B���͑w�̃j���[�����͎��w��100�̃j���[�����ɓ��˂���Ƃ���B100�̃j���[���������͂�Sparse�ɕ���������悤�ɂ��̊�������яd�ݍs����œK������B

### �摜�f�[�^�̓ǂݍ���
�f�[�^��<http://www.rctn.org/bruno/sparsenet/>����_�E�����[�h�ł���B`IMAGES_RAW.mat`��10���̎��R�摜�ŁA`IMAGES.mat`�͂���𔒐F���������̂ł���B`mat`�t�@�C���̓ǂݍ��݂ɂ�[MAT.jl](https://github.com/JuliaIO/MAT.jl)��p����B

using MAT

# datasets from http://www.rctn.org/bruno/sparsenet/
mat_images_raw = matopen("_static/datasets/IMAGES_RAW.mat")
imgs_raw = read(mat_images_raw, "IMAGESr")

mat_images = matopen("_static/datasets/IMAGES.mat")
imgs = read(mat_images, "IMAGES")

close(mat_images_raw)
close(mat_images)

�摜�f�[�^��`�悷��B

figure(figsize=(8, 3))
subplots_adjust(hspace=0.1, wspace=0.1)
for i=1:10
    subplot(2, 5, i)
    imshow(imgs_raw[:,:,i], cmap="gray")
    axis("off")
end
suptitle("Natural Images", fontsize=12)
subplots_adjust(top=0.9)  

### ���f���̒�`
�K�v�ȃp�b�P�[�W��ǂݍ��ށB

using Base: @kwdef
using Parameters: @unpack # or using UnPack
using LinearAlgebra
using Random
using Statistics
using ProgressMeter

���f�����`����B

@kwdef struct OFParameter{FT}
    lr_r::FT = 1e-2 # learning rate of r
    lr_Phi::FT = 1e-2 # learning rate of Phi
    lmda::FT = 5e-3 # regularization parameter
end

@kwdef mutable struct OlshausenField1996Model{FT}
    param::OFParameter = OFParameter{FT}()
    num_inputs::Int32
    num_units::Int32
    batch_size::Int32
    r::Array{FT} = zeros(batch_size, num_units) # activity of neurons
    Phi::Array{FT} = randn(num_inputs, num_units) .* sqrt(1/num_units)
end

�p�����[�^���X�V����֐����`����B

function updateOF!(variable::OlshausenField1996Model, param::OFParameter, inputs::Array, training::Bool)
    @unpack num_inputs, num_units, batch_size, r, Phi = variable
    @unpack lr_r, lr_Phi, lmda = param

    # Updates                
    error = inputs .- r * Phi'
    r_ = r +lr_r .* error * Phi

    r[:, :] = soft_thresholding_func.(r_, lmda)

    if training 
        error = inputs - r * Phi'
        dPhi = error' * r
        Phi[:, :] += lr_Phi * dPhi
    end
    
    return error
end

�s���Ƃɐ��K������֐����`����B

function normalize_rows(A::Array)
    return A ./ sqrt.(sum(A.^2, dims=1) .+ 1e-8)
end

�����֐����`����B

function calculate_total_error(error, r, lmda)
    recon_error = mean(error.^2)
    sparsity_r = lmda*mean(abs.(r)) 
    return recon_error + sparsity_r
end

�V�~�����[�V���������s����֐����`����B�O����`for loop`�ł͉摜�p�b�`�̍쐬��`r`�̏��������s���B������`for loop`�ł�`r`����������܂ōX�V���s���A���������Ƃ��ɏd�ݍs��`Phi`���X�V����B

function run_simulation(imgs, num_iter, nt_max, batch_size, sz, num_units, eps)
    H, W, num_images = size(imgs)
    num_inputs = sz^2

    model = OlshausenField1996Model{Float32}(num_inputs=num_inputs, num_units=num_units, batch_size=batch_size)
    errorarr = zeros(num_iter) # Vector to save errors    
    
    # Run simulation
    @showprogress "Computing..." for iter in 1:num_iter
        # Get the coordinates of the upper left corner of clopping image randomly.
        beginx = rand(1:W-sz, batch_size)
        beginy = rand(1:H-sz, batch_size)

        inputs = zeros(batch_size, num_inputs)  # Input image patches

        # Get images randomly
        for i in 1:batch_size        
            idx = rand(1:num_images)
            img = imgs[:, :, idx]
            clop = img[beginy[i]:beginy[i]+sz-1, beginx[i]:beginx[i]+sz-1][:]
            #clop = collect(flatten(img[beginy[i]:beginy[i]+sz-1, beginx[i]:beginx[i]+sz-1]))
            inputs[i, :] = clop .- mean(clop)
        end

        model.r = zeros(batch_size, num_units) # Reset r states
        model.Phi = normalize_rows(model.Phi) # Normalize weights
        # Input image patches until latent variables are converged 
        r_tm1 = zeros(batch_size, num_units)  # set previous r (t minus 1)

        for t in 1:nt_max
            # Update r without update weights 
            error = updateOF!(model, model.param, inputs, false)

            dr = model.r - r_tm1 

            # Compute norm of r
            dr_norm = sqrt(sum(dr.^2)) / sqrt(sum(r_tm1.^2) + 1e-8)
            r_tm1 = copy(model.r) # update r_tm1

            # Check convergence of r, then update weights
            if dr_norm < eps
                error = updateOF!(model, model.param, inputs, true)
                errorarr[iter] = calculate_total_error(error, model.r, model.param.lmda) # Append errors
                break
            end

            # If failure to convergence, break and print error
            if t >= nt_max-1
                print("Error at patch:", iter_, dr_norm)
                errorarr[iter] = calculate_total_error(error, model.r, model.param.lmda) # Append errors
                break
            end
        end
        """
        # Print moving average error
        if iter % 100 == 0
            moving_average_error = mean(errorarr[iter-99:iter])
            println("iter: ", iter, "/", num_iter, ", Moving average error:", moving_average_error)
        end
        """
    end
    return model, errorarr
end

### �V�~�����[�V�����̎��s

# Simulation constants
num_iter = 500 # number of iterations
nt_max = 1000 # Maximum number of simulation time
batch_size = 250 # Batch size

sz = 16 # image patch size
num_units = 100 # number of neurons (units)
eps = 1e-2 # small value which determines convergence

model, errorarr = run_simulation(imgs, num_iter, nt_max, batch_size, sz, num_units, eps)

### �P�����̑����̕`��
�P�����̑����̕ω���`�悵�Ă݂悤�B�������ቺ���A�w�K���i�s�������Ƃ�������B

# Plot error
figure(figsize=(4, 2))
ylabel("Error")
xlabel("Iterations")
plot(1:num_iter, errorarr)
tight_layout()

### �d�ݍs�� (��e��)�̕`��
�w�K��̏d�ݍs�� `Phi` ($\Phi$)���������Ă݂悤�B

# Plot Receptive fields
figure(figsize=(4.2, 4))
subplots_adjust(hspace=0.1, wspace=0.1)
for i in 1:num_units
    subplot(10, 10, i)
    imshow(reshape(model.Phi[:, i], (sz, sz)), cmap="gray")
    axis("off")
end
suptitle("Receptive fields", fontsize=14)
subplots_adjust(top=0.925)

���F��**ON�̈�**(����)�A���F��**OFF�̈�**(�}��)��\���BGabor�t�B���^�l�̋Ǐ���e�삪�����Ă���A����͈ꎟ���o��(V1)�ɂ�����P���^�זE(simple cells)�̎�e��ɗގ����Ă���B