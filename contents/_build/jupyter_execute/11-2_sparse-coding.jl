# 11.2 Sparse coding (Olshausen & Field, 1996) ���f��

## 11.2.1 Sparse coding�Ɛ������f��
Sparse coding���f��([Olshausen & Field, *Nature*. 1996](https://www.nature.com/articles/381607a0))��V1�̃j���[�����̉����������������**���`�������f��** (linear generative model)�ł���B�܂��A�摜�p�b�` $\mathbf{x}$ �����֐�(basis function) $\mathbf{\Phi} = [\phi_j]$ �̃m�C�Y���܂ސ��`�a�ŕ\�����Ƃ��� (�W���� $\mathbf{r}=[r_j]$ �Ƃ���)�B

$$
\mathbf{x} = \sum_j r_j \phi_j +\boldsymbol{\epsilon}= \mathbf{\Phi} \mathbf{r}+ \boldsymbol{\epsilon} \quad \tag{1}
$$

�������A$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ �ł���B���̃��f����_�o�l�b�g���[�N�̃��f���ƍl����ƁA $\mathbf{\Phi}$ �͏d�ݍs��A�W�� $\mathbf{r}$ �͓��͂��������̐_�o�זE�̊����x��\���Ă���Ɖ��߂ł���B�������A$r_j$ �͕��̒l�����̂ŒP���ɔ��Η��Ƒ������Ȃ��̂͂��̃��f���̌��_�ł���B

Sparse coding�ł͐_�o���� $\mathbf{r}$ �����ݕϐ��̐���ʂ�\�����Ă���Ƃ�������̉��A�����̊��ŉ摜 (��ړI�ϐ�)��\�����Ƃ�ړI�Ƃ���B�v�͏㎮�ɂ����āA�قƂ�ǂ�0�ŁA�ꕔ����0�ȊO�̒l�����Ƃ����a (=sparse)�ȌW��$\mathbf{r}$�����߂����B

### �m���I���f���̋L�q
````{margin}
```{note}
�Ȍ�̋L�q��([Olshausen & Field, 1997](https://pubmed.ncbi.nlm.nih.gov/9425546/); [Barello et al., 2018](https://www.biorxiv.org/content/10.1101/399246v2.full))���Q�l�ɂ����B
```
````

���͂����摜�p�b�` $\mathbf{x}_i\ (i=1, \cdots, N)$ �̐^�̕��z�� $p_{data}(\mathbf{x})$ �Ƃ���B�܂��A$\mathbf{x}$ �̐������f���� $p(\mathbf{x}|\mathbf{\Phi})$ �Ƃ���B����ɐ��ݕϐ� $\mathbf{r}$ �̎��O���z (prior)�� $p(\mathbf{r})$, �摜�p�b�` $\mathbf{x}$ �̖ޓx (likelihood)�� $p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})$ �Ƃ���B���̂Ƃ��A

$$
p(\mathbf{x}|\mathbf{\Phi})=\int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r} \quad \tag{2}
$$

�����藧�B$p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})$�́A(1)���ɂ����ăm�C�Y����$\boldsymbol{\epsilon} \sim\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$�Ƃ������Ƃ���A

$$
p(\mathbf{x}|\ \mathbf{r}, \mathbf{\Phi})=\mathcal{N}\left(\mathbf{x}|\ \mathbf{\Phi} \mathbf{r}, \sigma^2 \mathbf{I} \right)=\frac{1}{Z_{\sigma}} \exp\left(-\frac{\|\mathbf{x} - \mathbf{\Phi} \mathbf{r}\|^2}{2\sigma^2}\right)\quad \tag{3}
$$

�ƕ\����B�������A$Z_{\sigma}$�͋K�i���萔�ł���B

### ���O���z�̐ݒ�
���O���z$p(\mathbf{r})$�Ƃ��ẮA0�ɂ����ăs�[�N������A���̏d��(heavy tail)������sparse distribution���邢�� **super-Gaussian distribution** (Laplace ���z��Cauchy���z�Ȃ�Gaussian���z����kurtotic�ȕ��z)��p����̂��ǂ��B���̂悤�ȕ��z�ł́A$\mathbf{r}$�̊e�v�f$r_i$�͂قƂ��0�ɓ������A������͂ɑ΂��Ă͑傫�Ȓl�����B$p(\mathbf{r})$�͈�ʉ����Ď�(4), (5)�̂悤�ɕ\�L����B

$$
\begin{align}
p(\mathbf{r})&=\prod_j p(r_j) \quad \tag{4}\\
p(r_j)&=\frac{1}{Z_{\beta}}\exp \left[-\beta S(r_j)\right] \quad \tag{5}
\end{align}
$$

�������A$\beta$�͋t���x(inverse temperature), $Z_{\beta}$�͋K�i���萔 (���z�֐�) �ł���[^can]�B$S(x)$�ƕ��z�̊֌W���܂Ƃ߂��\���ȉ��ƂȂ� (cf. [Harpur, 1997](https://pdfs.semanticscholar.org/be08/da912362bf40fe3ded78bdadc644f921b4e7.pdf))�B

[^can]: �����̗p��͓��v�͊w�ɂ����鐳�����z (�{���c�}�����z)���痈�Ă���B

|$S(r)$|$\dfrac{dS(r)}{dr}$|$p(r)$|���z��|��x(kurtosis)|
|:-:|:-:|:-:|:-:|:-:|
|$r^2$|$2r$|$\dfrac{1}{\alpha \sqrt{2\pi}}\exp\left(-\dfrac{r^2}{2\alpha^2}\right)$|Gaussian ���z|0|
|$\vert r\vert$|$\text{sign}(r)$|$\dfrac{1}{2\alpha}\exp\left(-\dfrac{\vert r\vert}{\alpha}\right)$|Laplace ���z|3.0|
|$\ln (\alpha^2+r^2)$|$\dfrac{2r}{\alpha^2+r^2}$|$\dfrac{\alpha}{\pi}\dfrac{1}{\alpha^2+r^2}=\dfrac{\alpha}{\pi}\exp[-\ln (\alpha^2+r^2)]$|Cauchy ���z|-|

���z$p(r)$��$S(r)$��`�悷��Ǝ��̂悤�ɂȂ�B

using PyPlot

x = range(-5, 5, length=300)
figure(figsize=(7,3))
subplot(1,2,1)
title(L"$p(x)$")
plot(x, 1/sqrt(2pi)*exp.(-(x.^2)/2), color="black", linestyle="--",label="Gaussian")
plot(x, 1/2*exp.(-abs.(x)), label="Laplace")
plot(x, 1 ./ (pi*(1 .+ x.^2)), label="Cauchy")
xlim(-5, 5); 
xlabel(L"$x$")
legend()

subplot(1,2,2)
title(L"S(x)")
plot(x, x.^2, color="black", linestyle="--",label="Gaussian")
plot(x, abs.(x), label="Laplace")
plot(x, log.(1 .+ x.^2), label="Cauchy")
xlim(-5, 5); ylim(0, 5)
xlabel(L"$x$")

tight_layout()

## 11.2.2 �ړI�֐��̐ݒ�ƍœK��
�œK�Ȑ������f���𓾂邽�߂ɁA���͂����摜�p�b�`�̐^�̕��z $p_{data}(\mathbf{x})$��$\mathbf{x}$�̐������f�� $p(\mathbf{x}|\mathbf{\Phi})$���߂Â���B���̂��߂ɁA2�̕��z��Kullback-Leibler �_�C�o�[�W�F���X $D_{\text{KL}}\left(p_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$���ŏ����������B�������A�^�̕��z�͓����Ȃ��̂ŁA�o�����z 

$$
\hat{p}_{data}(\mathbf{x}):=\frac{1}{N}\sum_{i=1}^N \delta(\mathbf{x}-\mathbf{x}_i) \tag{6}
$$

���ߎ��Ƃ��ėp���� ($\delta(\cdot)$ ��Dirac�̃f���^�֐��ł���)�B�䂦��$D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$���ŏ�������B

$$
\begin{align}
D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)&=\int \hat{p}_{data}(\mathbf{x}) \log \frac{\hat{p}_{data}(\mathbf{x})}{p(\mathbf{x}|\mathbf{\Phi})} d\mathbf{x}\\
&=\mathbb{E}_{\hat{p}_{data}} \left[\ln \frac{\hat{p}_{data}(\mathbf{x})}{p(\mathbf{x}|\mathbf{\Phi})}\right]\\
&=\mathbb{E}_{\hat{p}_{data}} \left[\ln \hat{p}_{data}(\mathbf{x})\right]-\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right] \tag{7}
\end{align}
$$

�����藧�B(7)����1�Ԗڂ̍��͈��Ȃ̂ŁA$D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \Vert\ p(\mathbf{x}|\mathbf{\Phi})\right)$ ���ŏ�������ɂ�$\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]$���ő剻����΂悢�B�����ŁA

$$
\mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]=\sum_{i=1}^N \hat{p}_{data}(\mathbf{x}_i)\ln p(\mathbf{x}_i|\mathbf{\Phi})=\frac{1}{N}\sum_{i=1}^N \ln p(\mathbf{x}_i|\mathbf{\Phi}) \tag{8}
$$

�����藧�B�܂��A(2)�����

$$
\ln p(\mathbf{x}|\mathbf{\Phi})=\ln \int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r}
$$

�����藧�̂ŁA�ߎ��Ƃ��� $\displaystyle \int p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})d\mathbf{r}$ �� $p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r}) \left(=p(\mathbf{x}, \mathbf{r}| \mathbf{\Phi})\right)$ �ŕ]������B�����̋ߎ��̉��A�œK��$\mathbf{\Phi}=\mathbf{\Phi}^*$�͎��̂悤�ɂ��ċ��߂���B

$$
\begin{align}
\mathbf{\Phi}^*&=\text{arg} \min_{\mathbf{\Phi}} \min_{\mathbf{r}} D_{\text{KL}}\left(\hat{p}_{data}(\mathbf{x}) \| p(\mathbf{x}|\mathbf{\Phi})\right)\\
&=\text{arg} \max_{\mathbf{\Phi}} \max_{\mathbf{r}} \mathbb{E}_{\hat{p}_{data}} \left[\ln p(\mathbf{x}|\mathbf{\Phi})\right]\\
&= \text{arg} \max_{\mathbf{\Phi}}\sum_{i=1}^N \max_{\mathbf{r}_i} \ln p(\mathbf{x}_i|\mathbf{\Phi})\\
&\approx \text{arg} \max_{\mathbf{\Phi}}\sum_{i=1}^N \max_{\mathbf{r}_i} \ln p(\mathbf{x}_i|\mathbf{r}_i, \mathbf{\Phi})p(\mathbf{r}_i)\\
&=\text{arg}\min_{\mathbf{\Phi}} \sum_{i=1}^N \min_{\mathbf{r}_i}\ E(\mathbf{x}_i, \mathbf{r}_i|\mathbf{\Phi})\tag{9}
\end{align}
$$

�������A$\mathbf{x}_i$�ɑ΂���_�o������ $\mathbf{r}_i$�Ƃ����B�܂��A$E(\mathbf{x}, \mathbf{r}|\mathbf{\Phi})$�̓R�X�g�֐��ł���A�����̂悤�ɕ\�����B

$$
\begin{align}
E(\mathbf{x}, \mathbf{r}|\mathbf{\Phi}):=&-\ln p(\mathbf{x}|\mathbf{r}, \mathbf{\Phi})p(\mathbf{r})\\
=&\underbrace{\left\|\mathbf{x}-\mathbf{\Phi} \mathbf{r}\right\|^2}_{\text{preserve information}} + \lambda \underbrace{\sum_j S\left(r_j\right)}_{\text{sparseness of}\ r_j}\tag{10}
\end{align}
$$

�������A$\lambda=2\sigma^2\beta$�͐������W��[^lam]�ł���A1�s�ڂ���2�s�ڂւ͎�(3), (4), (5)��p�����B�����ŁA��1�������������A��2���������� (��������)�ƂȂ��Ă���B

��(9)�ŕ\�����œK���菇���œK��$\mathbf{r}$��$\mathbf{\Phi}$�����߂�ߒ��ɕ������悤�B�܂��A $\mathbf{\Phi}$���Œ肵������$E(\mathbf{x}_n, \mathbf{r}_i|\mathbf{\Phi})$���ŏ�������$\mathbf{r}_i=\hat{\mathbf{r}}_i$�����߂� ([11.1.3](#locally-competitive-algorithm-lca))�B

$$
\hat{\mathbf{r}}_i=\text{arg}\min_{\mathbf{r}_i}E(\mathbf{x}_i, \mathbf{r}_i|\mathbf{\Phi})\ \left(= \text{arg}\max_{\mathbf{r}_i}p(\mathbf{r}_i|\mathbf{x}_i)\right)
$$

����� $\mathbf{r}$ �ɂ��� **MAP����** (maximum a posteriori estimation)���s�����Ƃɓ������B����$\hat{\mathbf{r}}$��p����

$$
\mathbf{\Phi}^*=\text{arg}\min_{\mathbf{\Phi}} \sum_{i=1}^N E(\mathbf{x}_i, \hat{\mathbf{r}}_i|\mathbf{\Phi})\ \left(= \text{arg}\max_{\mathbf{\Phi}} \prod_{i=1}^N p(\mathbf{x}_i|\hat{\mathbf{r}}_i, \mathbf{\Phi})\right)
$$

�Ƃ��邱�Ƃɂ��A$\mathbf{\Phi}$���œK������ ([11.1.4](#id6))�B������� $\mathbf{\Phi}$ �ɂ��� **�Ŗސ���** (maximum likelihood estimation)���s�����Ƃɓ������B

[^lam]: ���̎�����t���x$\beta$���������̓x�����𒲐�����p�����[�^�ł��邱�Ƃ��킩��B

##  11.2.3 Locally competitive algorithm (LCA) 
$\mathbf{r}$�̌��z�@�ɂ��X�V���́A$E$�̔����ɂ�莟�̂悤�ɓ�����B

$$
\frac{d \mathbf{r}}{dt}= -\frac{\eta_\mathbf{r}}{2}\frac{\partial E}{\partial \mathbf{r}}=\eta_\mathbf{r} \cdot\left[\mathbf{\Phi}^T (\mathbf{x}-\mathbf{\Phi}\mathbf{r})- \frac{\lambda}{2}S'\left(\mathbf{r}\right)\right]
$$

�������A$\eta_{\mathbf{r}}$�͊w�K���ł���B���̎��ɂ��$\mathbf{r}$����������܂ōœK�����邪�A�P�Ȃ���z�@�ł͂Ȃ��A(Olshausen & Field, 1996)�ł�**�������z�@** (conjugate gradient method)��p���Ă���B�������A�������z�@�͎������ώG�Ŕ�����ł��邽�߁A�������I�������w�I�ȑÓ����̍����w�K�@�Ƃ��āA**LCA**  (locally competitive algorithm)����Ă���Ă��� ([Rozell et al., *Neural Comput*. 2008](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf))�BLCA��**���}��** (local competition, lateral inhibition)��**臒l�֐�** (thresholding function)��p����X�V���ł���BLCA�ɂ��X�V���s��RNN�͒ʏ��RNN�Ƃ͈قȂ�A�R�X�g�֐�(�܂��̓G�l���M�[�֐�)���ŏ������铮�I�V�X�e���ł���B���̂悤�ȋ@�\��Hopfield network�ŗp�����Ă��邽�߂ɁAOlshausen��**Hopfield trick**�ƌĂ�ł���B

### ���臒l�֐���p����ꍇ (ISTA)
$S(x)=|x|$�Ƃ����ꍇ��臒l�֐���p�����@�Ƃ���**ISTA**(Iterative Shrinkage Thresholding Algorithm)������BISTA��L1-norm���������ɑ΂���ߐڌ��z�@�ŁA�v��Lasso��A�ɗp������z�@�ł���B

�����ׂ����͎����ŕ\�����B

$$
\mathbf{r} = \mathop{\rm arg~min}\limits_{\mathbf{r}}\left\{\|\mathbf{x}-\mathbf{\Phi}\mathbf{r}\|^2_2+\lambda\|\mathbf{r}\|_1\right\}
$$

�ڍׂ͌�q���邪�A���̂悤�ɍX�V���邱�Ƃŉ���������B

1. $\mathbf{r}(0)$��v�f���S��0�̃x�N�g���ŏ������F$\mathbf{r}(0)=\mathbf{0}$
2. $\mathbf{r}_*(t+1)=\mathbf{r}(t)+\eta_\mathbf{r}\cdot \mathbf{\Phi}^T(\mathbf{x}-\mathbf{\Phi}\mathbf{r}(t))$
3. $\mathbf{r}(t+1) = \Theta_\lambda(\mathbf{r}_*(t+1))$
4. $\mathbf{r}$����������܂�2��3���J��Ԃ�

������$\Theta_\lambda(\cdot)$��**���臒l�֐�** (Soft thresholding function)�ƌĂ΂�A�����ŕ\�����B

$$
\Theta_\lambda(y)= 
\begin{cases} 
y-\lambda & (y>\lambda)\\ 
0 & (-\lambda\leq y\leq\lambda)\\ 
 y+\lambda & (y<-\lambda) 
\end{cases}
$$

$\Theta_\lambda(\cdot)$���֐��Ƃ��Ē�`����Ǝ��̂悤�ɂȂ� [^softthr]�B


[^softthr]: ReLU (�����v�֐�)��`max(x, 0)`�Ŏ����ł���B���̓_����l�����ReLU�����臒l�֐� (soft nonnegative thresholding function)�Ƒ����邱�Ƃ��ł��� ([Papyan et al., 2018](https://ieeexplore.ieee.org/document/8398588))�B

# thresholding function of S(x)=|x|
function soft_thresholding_func(x, lmda)
    max(x - lmda, 0) - max(-x - lmda, 0)
end

����$\Theta_\lambda(\cdot)$��`�悷��Ǝ��̂悤�ɂȂ�B

xmin, xmax = -5, 5
x = range(xmin, xmax, length=100)
y = soft_thresholding_func.(x, 1)

figure(figsize=(4,4.5))
subplot(2,2,1)
title(L"$S(x)=|x|$")
plot(x, abs.(x))
xlim(xmin, xmax); ylim(0, 10)
hlines(y=xmax, xmin=xmin, xmax=xmax, color="k", alpha=0.2)
vlines(x=0, ymin=0, ymax=xmax*2, color="k", alpha=0.2)

subplot(2,2,2)
title(L"$\frac{\partial S(x)}{\partial x}$")
plot(x, x, "k--")
plot(x, sign.(x))
xlim(xmin, xmax); ylim(xmin, xmax)
hlines(y=0, xmin=xmin, xmax=xmax, color="k", alpha=0.2)
vlines(x=0, ymin=xmin, ymax=xmax, color="k", alpha=0.2)

subplot(2,2,3)
title(L"$f_\lambda(x)=x+\lambda\cdot\frac{\partial S(x)}{\partial x}$")
plot(x, x, "k--")
plot(x, x + 1*sign.(x))
xlabel(L"$x$")
xlim(-5, 5); ylim(-5, 5)
hlines(y=0, xmin=xmin, xmax=xmax, color="k", alpha=0.2)
vlines(x=0, ymin=xmin, ymax=xmax, color="k", alpha=0.2)

subplot(2,2,4)
title(L"$\Theta_\lambda(x)$")
plot(x, x, "k--")
plot(x, y)
xlabel(L"$x$")
xlim(-5, 5); ylim(-5, 5)
hlines(y=0, xmin=xmin, xmax=xmax, color="k", alpha=0.2)
vlines(x=0, ymin=xmin, ymax=xmax, color="k", alpha=0.2)

tight_layout()

�Ȃ��A���臒l�֐��͎��̖ړI�֐�$C$���ŏ�������$x$�����߂邱�Ƃœ��o�ł���B

$$
C=\frac{1}{2}(y-x)^2+\lambda |x|
$$

�������A$x, y, \lambda$�̓X�J���[�l�Ƃ���B$|x|$�������ł��Ȃ����A����͏ꍇ�������l���邱�Ƃŉ�������B$x\geq 0$���l����ƁA(6)����

$$
C=\frac{1}{2}(y-x)^2+\lambda x = \{x-(y-\lambda)\}^2+\lambda(y-\lambda)
$$

�ƂȂ�B(7)���̍ŏ��l��^����$x$�͏ꍇ���������čl����ƁA$y-\lambda\geq0$�̂Ƃ��񎟊֐��̒��_���l����$x=y-\lambda$�ƂȂ�B �����$y-\lambda<0$�̂Ƃ���$x\geq0$�ɂ����ĒP�������Ȋ֐��ƂȂ�̂ŁA�ŏ��ƂȂ�̂�$x=0$�̂Ƃ��ł���B���l�̋c�_��$x\leq0$�ɑ΂��Ă��s�����Ƃ� (5)����������B

�Ȃ��A臒l�֐��Ƃ��Ă͓��臒l�֐������ł͂Ȃ��A�d����臒l�֐���$y=x - \text{tanh}(x)$ (Tanh-shrink)�ȂǗl�X�Ȋ֐���p���邱�Ƃ��ł���B

## 11.2.4 �d�ݍs��̍X�V��
$\mathbf{r}$��������������z�@�ɂ��$\mathbf{\Phi}$���X�V����B

$$
\Delta \phi_i(\boldsymbol{x}) = -\eta \frac{\partial E}{\partial \mathbf{\Phi}}=\eta\cdot\left[\left([\mathbf{x}-\mathbf{\Phi}\mathbf{r}\right)\mathbf{r}^T\right]
$$

## 11.2.5 Sparse coding network�̎���
�l�b�g���[�N�͓��͑w���܂�2�w�̒P���ȍ\���ł���B����́A���͂̓����_���ɐ؂�o����16�~16 (��256)�̉摜�p�b�`�Ƃ��A�������͑w��256�̃j���[�������󂯎��Ƃ���B���͑w�̃j���[�����͎��w��100�̃j���[�����ɓ��˂���Ƃ���B100�̃j���[���������͂�Sparse�ɕ���������悤�ɂ��̊�������яd�ݍs����œK������B

### �摜�f�[�^�̓ǂݍ���
�f�[�^��<http://www.rctn.org/bruno/sparsenet/>����_�E�����[�h�ł��� [^datasets]�B`IMAGES_RAW.mat`��10���̎��R�摜�ŁA`IMAGES.mat`�͂���𔒐F���������̂ł���B`mat`�t�@�C���̓ǂݍ��݂ɂ�[MAT.jl](https://github.com/JuliaIO/MAT.jl)��p����B

[^datasets]: ����̓A�����J�k�����ŎB�e���ꂽ���R�摜�ł���A[van Hateren's Natural Image Dataset](http://bethgelab.org/datasets/vanhateren/)����擾���ꂽ���̂ł���B

using MAT
#using PyPlot

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
            r_tm1 .= model.r # update r_tm1

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
        # Print moving average error
        if iter % 100 == 0
            moving_average_error = mean(errorarr[iter-99:iter])
            println("iter: ", iter, "/", num_iter, ", Moving average error:", moving_average_error)
        end
    end
    return model, errorarr
end

`r_tm1 .= model.r`�̕����́A�v�f���Ƃ̃R�s�[�����s���Ă���B`r_tm1 = copy(model.r)`�ł��悢���A�V���ȃ��������蓖�Ă�������̂Ŕ����Ă���B`@. r_tm1 = model.r`�Ƃ��Ă��悢�B

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
�w�K��̏d�ݍs�� `Phi` ($\mathbf{\Phi}$)���������Ă݂悤�B

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

### �摜�̍č\��
�w�K�������f����p���ē��͉摜���č\������邩�m�F���悤�B

H, W, num_images = size(imgs)
num_inputs = sz^2

# Get the coordinates of the upper left corner of clopping image randomly.
beginx = rand(1:W-sz, batch_size)
beginy = rand(1:H-sz, batch_size)

inputs = zeros(batch_size, num_inputs)  # Input image patches

# Get images randomly
for i in 1:batch_size        
    idx = rand(1:num_images)
    img = imgs[:, :, idx]
    clop = img[beginy[i]:beginy[i]+sz-1, beginx[i]:beginx[i]+sz-1][:]
    inputs[i, :] = clop .- mean(clop)
end

model.r = zeros(batch_size, num_units) # Reset r states

# Input image patches until latent variables are converged 
r_tm1 = zeros(batch_size, num_units)  # set previous r (t minus 1)

for t in 1:nt_max
    # Update r without update weights 
    error = updateOF!(model, model.param, inputs, false)

    dr = model.r - r_tm1 

    # Compute norm of r
    dr_norm = sqrt(sum(dr.^2)) / sqrt(sum(r_tm1.^2) + 1e-8)
    r_tm1 .= model.r # update r_tm1

    # Check convergence of r, then update weights
    if dr_norm < eps
        break
    end
end

�_�o���� $\mathbf{r}$���X�p�[�X�ɂȂ��Ă��邩�m�F���悤�B

println(model.r[1, :])

�v�f���قƂ��0�̃X�p�[�X�ȃx�N�g���ɂȂ��Ă��邱�Ƃ��킩��B���ɉ摜���č\������B

reconst = model.r * model.Phi'
println(size(reconst))

�č\���������ʂ�`�悷��B

figure(figsize=(7.5, 3))
subplots_adjust(hspace=0.1, wspace=0.1)
num_show = 5
for i in 1:num_show
    subplot(2, num_show, i)
    imshow(reshape(inputs[i, :], (sz, sz)), cmap="gray")
    xticks([]); yticks([]); 
    if i == 1
        ylabel("Input\n images")
    end

    subplot(2, num_show, num_show+i)
    imshow(reshape(reconst[i, :], (sz, sz)), cmap="gray")
    xticks([]); yticks([]); 
    if i == 1
        ylabel("Reconstructed\n images")
    end
end

��i�����͉摜�A���i���č\�����ꂽ�摜�ł���B���ق͂�����̂́A�T�ˍč\������Ă��邱�Ƃ��킩��B

```{admonition} �_���ȊO�̎Q�l����
- <http://www.scholarpedia.org/article/Sparse_coding>
- Bruno Olshausen: �gSparse coding in brains and machines�h([Stanford talks](https://talks.stanford.edu/bruno-olshausen-sparse-coding-in-brains-and-machines/)), [Slide](http://www.rctn.org/bruno/public/Simons-sparse-coding.pdf)
- <https://redwood.berkeley.edu/wp-content/uploads/2018/08/sparse-coding-ICA.pdf>
- <https://redwood.berkeley.edu/wp-content/uploads/2018/08/sparse-coding-LCA.pdf>
- <https://redwood.berkeley.edu/wp-content/uploads/2018/08/Dylan-lca_overcompleteness_09-27-2018.pdf>
```