#### NN

##### Fully Connected Layer

Forward Propagation：\[ z=Wx+b \] → \[ z_i=\sum_jW_{ij}x_j+b_i \]
Local Derivative：\[ g=\frac{\partial L}{\partial z} \] → \[ \frac{\partial L}{\partial W_{ij}}=g_ix_j \quad \frac{\partial L}{\partial x_j}=\sum_i g_iW_{ij} \]

##### Softmax + Cross-Entropy

$p_i=\frac{e^{z_i}}{\sum_{k=1}^{C}e^{z_k}},\quad
L=-\sum_{k=1}^{C}t_k\log p_k=-\log p_c,\quad
\frac{\partial L}{\partial z_i}=p_i-t_i$

##### Sigmoid + Binary Cross-Entropy

$p=\frac{1}{1+e^{-z}},\quad
L=-[t\log p+(1-t)\log(1-p)],\quad
\frac{\partial L}{\partial z}=p-t$

##### Activation Functions

| Activation | Function                                          | Derivative                                    | Advantages            | Disadvantages           |
| ---------- | ------------------------------------------------- | --------------------------------------------- | --------------------- | ----------------------- |
| Sigmoid    | \(\frac{1}{1+e^{-z}}\)                            | \(f(z)(1-f(z))\)                              | Binary Classification | Vanishing Gradients     |
| Tanh       | \(\frac{e^z-e^{-z}}{e^z+e^{-z}}\)                 | \(1-f(z)^2\)                                  | Zero-Centered         | Vanishing Gradients     |
| ReLU       | \(\max(0,z)\)                                     | \(\begin{cases}1&z>0\\0&z<0\end{cases}\)      | Simple                | Dying ReLU              |
| Leaky ReLU | \(\begin{cases}z&z\ge0\\\alpha z&z<0\end{cases}\) | \(\begin{cases}1&z>0\\\alpha&z<0\end{cases}\) | Preserve Gradients    | Hyperparameter $\alpha$ |

##### Multi-Layer Perceptron

MLP：\[ z^{(l)}=W^{(l)}\phi_l(z^{(l-1)})+b^{(l)}\] →  Higher-Dimensional Space → Linearly Separable
Activation → Nonlinearity, Multi-Layer → Representation Efficiency；Linear Layer → Class Logits

##### Regression Losses

\[ L_1=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i| \] → \[ \frac{\partial L_1}{\partial \hat y_i} =\frac{1}{N}\operatorname{sign}(\hat y_i-y_i) \] → robust to outliers
\[ L_2=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2 \] → \[ \frac{\partial L_2}{\partial \hat y_i} =\frac{2}{N}(\hat y_i-y_i) \] → sensitive to outliers

##### Batch Normalization

Mean：\[ \mu_j=\frac1N\sum_i x_{ij} \], Variance：\[ v_j=\frac1N\sum_i(x_{ij}-\mu_j)^2 \], Standardization：\[ \bar x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{v_j+\epsilon}} \]
Normalize Intermediate Features→Stabilize Gradient Scales→Larger Learning Rates / Faster Convergence

##### Dropout

\[ y_i^{\mathrm{train}}=m_iz_i, \quad y_i^{\mathrm{test}}=qz_i \] → Keep Probability $q_i$ → Random Mask \[ m_i\sim\operatorname{Bernoulli}(q) \]
Randomly Deactivate Neurons→Reduce Co-Adaptation→Mitigate Overfitting→Improve Generalization

##### AutoEncoder

Restruction Loss：\[ h=f_\theta(x),\quad \tilde x=g_\phi(h), \quad J=\frac1{2N}\sum_{i=1}^{N}\left\|x^{(i)}-\tilde x^{(i)}\right\|_2^2 \]
Low-Dimensional Bottleneck→Learn Latent Representative → Use Encoder for other tasks

#### Optimization

##### Mini-Batch SGD

Loss Function：$J_{\mathcal{B}_t}(\theta) = \frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t} \ell_i(\theta_t)$
Objective Function：\[ g_t=\frac{1}{B_t}\sum_{i\in\mathcal B_t}\nabla\ell_i(\theta_t) \] 
Gradient Descent：\[ \theta_{t+1}=\theta_t-\eta_t\nabla J(\theta_t) \] 
First-Order Taylor Expansion：$J_{\mathcal{B}_t}(\theta_t - \eta_t g_t) \approx J_{\mathcal{B}_t}(\theta_t) - \eta_t\, g_t^\top g_t = J_{\mathcal{B}_t}(\theta_t) - \eta_t\|g_t\|_2^2$
Smaller Batch Size→High Gradient Variance→Noisy Updates→Better Generalization
Larger Batch Size→High Memory Usage→Fewer Updates→Better Stability
Smaller Learning Rate→Slow Convergence→More Epochs
Larger Learning Rate→Repeated Overshooting→Oscillation / Divergence

##### Momentum

Exponential Moving Average：$v_t=\mu v_{t-1}+(1-\mu)g_t\ \quad \theta_{t+1}=\theta_t-\eta_tv_t$
Accumulate Past Gradients→Reinforce Consistent / Reduce Oscillation→Faster Convergence

##### L2 Regularization

Regularization Term：\[ J_{\mathrm{reg}}(\theta) =J_{\mathrm{data}}(\theta)+\frac{\lambda}{2}\|\theta\|_2^2 \quad \nabla_\theta\frac{\lambda}{2}\|\theta\|_2^2=\lambda\theta\]
Shrinkage Factor：\[\theta_{t+1}=\theta_t-\eta(g_t+\lambda\theta_t)=(1-\eta\lambda)\theta_t-\eta g_t\]
Weight Decay→Penalize Large Weights→Reduce Overfitting→Improve Generalization

##### Resilient Propagation

Rprop：\[ \eta_{t,i}= \begin{cases} \min(\alpha\eta_{t-1,i},\eta_{\max}), &g_{t,i}g_{t-1,i}>0,\,\alpha>1\\ \max(\beta\eta_{t-1,i},\eta_{\min}), &g_{t,i}g_{t-1,i}<0,\,0<\beta<1\end{cases} \quad \theta_{t+1,i} =\theta_{t,i}-\eta_{t,i}\operatorname{sign}(g_{t,i})\]
Consistent Signs→Increase Step Size→Faster Convergence
Opposite Signs→Decrease Step Size→Reduce Oscillation

##### Adaptive Learning Rate

AdaGrad：\[ s_{t,i}=s_{t-1,i}+g_{t,i}^2 \quad \theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{s_{t,i}}+\epsilon}\odot g_{t,i}\]
RMSProp：\[ s_{t,i}=\rho s_{t-1,i}+(1-\rho)g_{t,i}^2 \quad \theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{s_{t,i}}+\epsilon}\odot g_{t,i}\]
Measure Gradient Magnitudes→Adapt Learning Rates→Stabilize Updates→Reduce Oscillation

##### Adaptive Moment Estimation

First-Moment Estimate：\[ m_t=\beta_1m_{t-1}+(1-\beta_1)g_t \] → Determine Update Direction 
Second Raw-Moment Estimate：\[ v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2 \] → Adjust Update Step Size
Bias Correction：\[ \hat m_t=m_t/(1-\beta_1^t) \quad \hat v_t=v_t/(1-\beta_2^t)\] → Correct Zero Initialization Bias
Adam：$\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ AdamW：$\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$

##### Gradients Problems

Vanishing Gradients / Exploding Gradients → Repeated Multiplication of Local Derivatives
Gradient Clipping：$g_{\text{clip}}=g\cdot\min\left(1,\frac{c}{\|g\|_2}\right)$ → Bounded Gradient Norm → More Stable Training

##### Weight Initialization

Xavier / Glort：$W\sim\mathcal N\left(0,\frac{2}{n_{\text{in}}+n_{\text{out}}}\right) \quad W\sim U\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$ → sigmoid、tanh
Kaiming／He：$W\sim\mathcal N\left(0,\frac{2}{n_{\text{in}}}\right) \quad W\sim U\left(-\sqrt{\frac{6}{n_{\text{in}}}},\sqrt{\frac{6}{n_{\text{in}}}}\right)$ → relu
Prevent Symmetry + Preserve Activation / Gradient Scale → Reduce Vanishing / Exploding Gradients

##### Data Augmentation

Rotation、Flipping、Cropping、Scaling、Jittering、Noise、Erasing

##### Learning Rate Schedules

Inverse-Time Decay：\[\eta_t=\frac{\eta_0}{1+dt}\]、Step Decay：\[ \eta_t=\eta_0\gamma^{\lfloor t/K\rfloor} \]、Exponential Decay：\[ \eta_t=\eta_0e^{-kt} \]
Cosine Annealing：\[ \eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min}) \left(1+\cos\frac{\pi t}{T}\right) \]、Warmup：\[ \eta_t=\eta_{\mathrm{max}}\frac{t}{T_w}\]

#### CNN

##### Advantages

Local Connectivity：Sparse Connections→Lower Computation / Local Feature Extraction
Weight Sharing：Reuse Filters→Fewer Parameters / Location-Independent Feature Detection
Translation Equivariance：\[ f(T_\Delta X)=T_\Delta f(X) \] → Input Shifts, Feature Map Output Shifts Accordingly
Translation Invariance：\[ f(T_\Delta X)=f(X) \] → Input Shifts, Classification Output Remains Unchanged

##### Forward Propagation

Feature Map：\[ y_{i,j,o} = b_o+ \sum_{c=1}^{C_{\mathrm{in}}} \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} w_{o,c,u,v}\, x_{is_h-p_h+ud_h,\;js_w-p_w+vd_w,\;c} \]
Kernel Size：$K_{e} = D(K-1) + 1$
Output Size：\[ L_{\mathrm{out}} = \left\lfloor \frac{L_{\mathrm{in}}+2P-K_e}{S} \right\rfloor+1, \quad i_{\max} = \left\lfloor \frac{L_{\mathrm{in}}+2p-K_e}{s} \right\rfloor \]
Regular：Kernel=\(C_{\text{in}}C_{\text{out}}\), Parameter=\[\left(K_hK_wC_{in}+1\right)C_{out} \], Computational=\[ H_{\text{out}}W_{\text{out}}C_{\text{out}}K_hK_wC_{\text{in}} \]
Grouped：$C'_\text{in} = C_\text{in} / G$, Depthwise：$G = C_\text{in}$

##### Backward Propagation

\[ y_i=\sum_{k=0}^{K-1}w_kx_{i+k}+b\], $\delta_i=\frac{\partial J}{\partial y_i}$
$\frac{\partial J}{\partial w_k}=\sum_i\delta_i x_{i+k} \quad \frac{\partial J}{\partial b}=\sum_i\delta_i \quad \frac{\partial J}{\partial x_j}=\sum_i\delta_iw_{j-i}$

##### Receptive Field

\[ r_0=1 \quad j_0=1 \quad j_l=s_lj_{l-1} \quad r_l=r_{l-1}+(k_l-1)d_lj_{l-1} \]
$r_L=1+\sum_{l=1}^{L}(k_l-1)d_l\prod_{m=1}^{l-1}s_m$

##### Pooling

Max：\(y_{u,v,c}=\max_{(i,j)\in\Omega_{u,v}}x_{i,j,c}\) → \(\frac{H}{S}\times\frac{W}{S}\times C\)
Average：\(y_{u,v,c}=\frac{1}{K}\sum_{(i,j)\in\Omega_{u,v}}x_{i,j,c}\) → \(\frac{H}{S}\times\frac{W}{S}\times C\)
Global Average：\(y_c=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}x_{i,j,c}\) → \(1\times1\times C\)

##### BatchNorm / Dropout

\[ \mu_c=\frac1M\sum_{n,h,w}x_{n,h,w,c} \]，\[ v_c=\frac1M\sum_{n,h,w}(x_{n,h,w,c}-\mu_c)^2 \]
$y_{n,h,w,c}=\gamma_c\frac{x_{n,h,w,c}-\mu_c}{\sqrt{v_c+\epsilon}}+\beta_c$，$y_{n,h,w,c}=\frac{m_{n,c}}{1-p}x_{n,h,w,c}$

##### Other Convolution

Partial Convolution：$y_{i,j}=\frac{K}{N_{i,j}}\sum_{(u,v)\in\Omega}w_{u,v}x_{i+u,j+v}m_{i+u,j+v}+b,N_{i,j}=\sum_{(u,v)\in\Omega}m_{i+u,j+v}>0$ → Valid Input Only
Submanifold Sparse Convolution：$y_{i,j}=\sum_{(u,v)\in\Omega}w_{u,v}x_{i+u,j+v}m_{i+u,j+v}+b,m_{i,j}=1$ → Active Input Only



#### RNN

##### Sequence Probability Factorization

Probability Chain Rule：$P(x_{1:T})=P(x_1)\prod_{t=2}^{T}P(x_t\mid x_{1:t-1})$















