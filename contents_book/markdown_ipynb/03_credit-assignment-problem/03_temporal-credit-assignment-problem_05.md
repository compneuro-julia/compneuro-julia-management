## 適格度トレースによるRTRLの近似

RFRO (random feedback local online learning) \citep{murray2019local}

$$
\frac{\partial h_{j}(t)}{\partial W_{a b}} = (1 - \alpha) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
$$


$$
\frac{\partial h_{j} (t )}{\partial W_{a b}} = (1 - \alpha ) \frac{\partial h_{j} (t - 1 )}{\partial W_{a b}} + \alpha \delta_{j a} \phi^{\prime} (u_{a} (t ) ) h_{b} (t - 1 ) + \alpha \underset{k}{ \left (\sum \right ) } \phi^{\prime} (u_{j} (t ) ) W_{j k} \frac{\partial h_{k} (t - 1 )}{\partial W_{a b}}
$$

Output error

$$
\begin{align}
\epsilon (t)=\mathbf{y}(t)-\hat{\mathbf{y}}(t)\\
\mathcal{L}=\frac{1}{2T}\sum_{t=1}^T \|\epsilon (t)\|^2
\end{align}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}}=-\frac{1}{T}\sum_{t=1}^T \mathbf{W}_{out}^\top \epsilon (t)\frac{\partial \mathbf{h}(t)}{\partial \mathbf{W}}
$$

Update rule
$$
\begin{align}
\Delta \mathbf{W}^{out}_t&=\eta \epsilon_{t} \mathbf{h}_t\\
\Delta \mathbf{W}_{rec}(t)&=\eta \mathbf{B}\epsilon(t) \mathbf{P}(t)\\
\Delta \mathbf{W}_{in}(t)&=\eta \mathbf{B}\epsilon (t) \mathbf{Q}(t)\\
\end{align}
$$

Eligibility trace $\mathbf{P}\in \mathbb{R}^{N_{rec}\times N_{rec}}, \mathbf{Q}\in \mathbb{R}^{N_{rec}\times N_{in}}$

$$
\begin{align}
\mathbf{P}_t&=\alpha f'(\mathbf{u}_t)\mathbf{h}_{t-1}^\top+\left(1-\alpha\right)\mathbf{P}_{t-1}\\
\mathbf{Q}_t&=\alpha f'(\mathbf{u}_t)\mathbf{x}_{t-1}^\top+\left(1-\alpha\right)\mathbf{Q}_{t-1}
\end{align}
$$

RFLOに記載

$$
\Delta \mathbf{W}_{\textrm{out}} = \frac{\eta}{T} \sum_{t=1}^T \epsilon(t) h(t)^\top 
$$

$$
\Delta \mathbf{W}_{\textrm{rec}} = \frac{\eta}{T} \sum_{t=1}^T \mathbf{W}_{\textrm{out}}^\top \epsilon(t) \frac{\partial h(t)}{\partial \mathbf{W}_{\textrm{rec}}}
$$

http://frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full