# AIMS5771 Midterm Cheat Sheet

## 一、CNN

### 1.1 Convolution 的基本原理与输出尺寸

**原理：** 图像可以看成高、宽、channels 三个维度组成的数组。Convolution 用一个小窗口读取局部区域，对窗口中的数值加权求和，再让窗口滑过整张图像。同一个 filter 在所有位置复用相同权重，因此学到的局部模式可以在不同位置被检测。一个普通 filter 同时读取全部输入 channels；多个 filters 分别产生多个输出 channels。Padding 在边缘补值，Stride 决定窗口每次移动多少格，Dilation 则在 kernel 元素之间留出间隔。

**公式推导：** 设输入为 $X\in\mathbb{R}^{H\times W\times M}$，kernel 边长为 $K$，输出 channels 数为 $N$。对第 $n$ 个 filter，局部计算是

$$
Y_{i,j,n}=b_n+\sum_{u=0}^{K-1}\sum_{v=0}^{K-1}\sum_{m=1}^{M}
W_{u,v,m,n}\,\widetilde X_{iS+ud,jS+vd,m}.
$$

其中输出位置索引 $i,j$ 从 0 开始，$\widetilde X$ 是每侧补了 $P$ 格的输入，$S$ 是 Stride，$d$ 是 Dilation，$b_n$ 是该输出 channel 的 Bias。一个有 $K$ 个采样点、点间距为 $d$ 的窗口实际跨越

$$
K_{\mathrm{eff}}=1+(K-1)d
$$

格。沿高度方向，可用长度为 $H+2P$；窗口左端最多移动到 $H+2P-K_{\mathrm{eff}}$，每次移动 $S$，再加上起点本身，因此

$$
H_o=\left\lfloor\frac{H+2P-K_{\mathrm{eff}}}{S}\right\rfloor+1,
\qquad
W_o=\left\lfloor\frac{W+2P-K_{\mathrm{eff}}}{S}\right\rfloor+1.
$$

输出 shape 为 $H_o\times W_o\times N$。普通 Convolution 取 $d=1$；当 $K$ 为奇数、$S=1$ 且 $P=(K-1)/2$ 时，空间尺寸保持不变。

### 1.2 Convolution 的参数量与计算量

**原理：** 参数量统计模型需要学习多少数值，计算量统计处理一次输入需要执行多少运算，两者不是一回事。Convolution 在不同位置共享 filter，所以图像变大不会增加参数量，但会增加使用这些参数的次数。普通 Convolution 同时混合空间位置和 channels，成本因此同时依赖 kernel 面积、输入 channels 和输出 channels。

**公式推导：** 一个 filter 包含 $K^2$ 个空间位置，每个位置连接 $M$ 个输入 channels，因此有 $K^2M$ 个 weights。共有 $N$ 个 filters，并为每个输出 channel 配一个 Bias，得到

$$
P_{\mathrm{conv}}=N(K^2M+1)=K^2MN+N.
$$

每个输出元素要对 $K^2M$ 项进行乘法并累加。输出共有 $H_oW_oN$ 个元素，所以忽略 Bias、Activation 和 Normalization 的运算后，乘加次数为

$$
\mathrm{MACs}=H_oW_oN\cdot K^2M.
$$

其中一个 MAC 表示一次乘加。若乘法和加法各算一次 FLOP，则

$$
\mathrm{FLOPs}\approx2H_oW_oK^2MN.
$$

例如同时把输入、输出 channels 放大为原来的 $a$ 倍，kernel weights 和主要计算量都会约变为 $a^2$ 倍；若只把输出高、宽放大为原来的 $r$ 倍，参数量不变，MACs 约变为 $r^2$ 倍。这些关系是分析网络缩放和高效架构的基础。

### 1.3 Receptive Field 与 VGG 的小 kernel 堆叠

**原理：** Receptive Field 指一个中间特征最多依赖原始输入中的多大区域。单层小 kernel 只看局部，但多层堆叠后，上层会整合多个下层局部区域，因此可以看到更大的范围。VGG 用连续的 $3\times3$ Convolution 提取特征，再用独立 Pooling 下采样；这既扩大 Receptive Field，也在层间加入更多非线性。这里计算的是理论可覆盖范围，不保证范围内所有像素对输出的实际影响相同。

**公式推导：** 设第 $l$ 层的 Receptive Field 边长为 $r_l$，相邻特征点在原图中的间距为 $j_l$。原始像素满足 $r_0=j_0=1$。若该层 Stride 为 $S_l$，有效 kernel 为 $K_{\mathrm{eff},l}$，则

$$
j_l=S_lj_{l-1},\qquad
r_l=r_{l-1}+(K_{\mathrm{eff},l}-1)j_{l-1}.
$$

第二式来自：窗口中的第一个下层特征已经覆盖 $r_{l-1}$，最后一个下层特征的中心又向外移动了 $(K_{\mathrm{eff},l}-1)j_{l-1}$。因此两层 $3\times3$、Stride 1 的 Convolution 有

$$
r_1=1+2=3,\qquad r_2=3+2=5.
$$

若所有输入、中间、输出 channels 都为 $C$，忽略 Bias，两层小 kernel 的参数量为 $2\cdot3^2C^2=18C^2$，而一个 $5\times5$ 的参数量为 $25C^2$。前者以更少参数获得相同 Receptive Field 尺寸，但由于分解结构和中间 Activation，其函数形式不等同于任意单层大 kernel。

### 1.4 Pooling 与 Global Average Pooling

**原理：** Pooling 通常逐 channel 压缩空间信息，不混合不同 channels。Max Pooling 保留窗口中的最大响应，Average Pooling 保留平均响应；两者都没有可学习 weights，但会丢失部分位置信息。Global Average Pooling（GAP）把一整个 feature map 平均成一个数，让每个 channel 提供一个全局描述，常用于分类头之前，避免 Flatten 后连接庞大的 Fully Connected Layer。

**公式推导：** 对某个局部窗口 $\Omega$，其中包含 $|\Omega|$ 个位置，第 $c$ 个 channel 的两种输出分别为

$$
y_c^{\mathrm{max}}=\max_{(h,w)\in\Omega}x_{h,w,c},
\qquad
y_c^{\mathrm{avg}}=\frac{1}{|\Omega|}\sum_{(h,w)\in\Omega}x_{h,w,c}.
$$

当窗口覆盖整张 $H\times W$ feature map 时，就得到 GAP：

$$
z_c=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{h,w,c}.
$$

它把 $H\times W\times C$ 输入变成长度为 $C$ 的向量，平均系数固定，所以参数量为零。若随后输出 $D$ 个类别，Flatten 后的 FC 需要 $HWCD+D$ 个参数，而 GAP 后的分类 FC 只需 $CD+D$ 个参数。减少的是输入分类层的特征维度；整个分类头仍然可以有可学习参数。由线性平均还可直接得到 $\partial z_c/\partial x_{h,w,c}=1/(HW)$，即梯度均匀分配到该 channel 的所有位置。

### 1.5 AlexNet 与 Transfer Learning

**原理：** AlexNet 由 5 个 Convolution Layers、2 个用于特征变换的 Fully Connected Layers，以及最终分类 FC 组成；Poolings 接在第 1、2、5 个 Convolution 后。它把图像逐渐转换成适合分类的高层表示。Transfer Learning 复用这种已学好的表示：底层特征通常更通用，高层特征通常更依赖原任务。迁移失败既可能来自源、目标任务差异，也可能来自原本协同工作的层被拆开。Freeze 保留已学参数，Fine-tuning 则允许它们继续适应新任务。

**公式推导：** 将网络写为特征提取器 $f(x;\theta_f)$ 和分类器 $g(z;\theta_g)$ 的组合。对目标数据 $(x_i,y_i)$，优化目标为

$$
L(\theta_f,\theta_g)=\frac1T\sum_{i=1}^{T}
\ell\bigl(g(f(x_i;\theta_f);\theta_g),y_i\bigr),
$$

其中 $T$ 是样本数，$\ell$ 是单个样本的分类损失。Freeze 时只更新分类器：

$$
\theta_g\leftarrow\theta_g-\eta_g\nabla_{\theta_g}L,
\qquad \theta_f\text{ 保持不变}.
$$

Fine-tuning 额外执行 $\theta_f\leftarrow\theta_f-\eta_f\nabla_{\theta_f}L$，其中 $\eta_f,\eta_g$ 是学习率。链式法则给出

$$
\nabla_{\theta_f}L=
\left(\frac{\partial f}{\partial\theta_f}\right)^{\!\top}
\nabla_fL,
$$

说明目标任务的误差可以传回特征提取器。Freeze 是主动不执行这一步更新，不意味着该梯度在数学上必为零。迁移是否有效取决于任务与数据，不能从以上优化定义推导出必然提升。

### 1.6 Network in Network 与 Pointwise Convolution

**原理：** $1\times1$ Convolution 又称 Pointwise Convolution，它只读取同一空间位置，却可以同时组合所有 channels，因此主要负责 Channel Mixing。Network in Network（NIN）把这种变换放在空间 Convolution 之后，并配合 Activation，让每个位置都通过一个共享的小型网络。它既能改变 channels 数，也能增强每个位置的非线性表达，但单独使用不会扩大空间 Receptive Field。

**公式推导：** 把位置 $(h,w)$ 上的 $M$ 个 channels 写成行向量 $x_{h,w}\in\mathbb{R}^{1\times M}$，则输出为
$$
y_{h,w}=x_{h,w}W+b,
\qquad W\in\mathbb{R}^{M\times N}.
$$

同一个 $W$ 在所有位置共享，所以忽略 Bias 时参数量为 $MN$，MACs 为 $HWMN$。两个带 Activation 的 Pointwise Layers 可以写成

$$
h_{h,w}=\sigma(x_{h,w}W_1+b_1),\qquad
y_{h,w}=\sigma(h_{h,w}W_2+b_2),
$$

其中 $\sigma$ 是 ReLU 等非线性函数。如果去掉两次 $\sigma$，则

$$
y_{h,w}=x_{h,w}(W_1W_2)+(b_1W_2+b_2),
$$

仍等价于一次线性 channel 变换；因此叠加 Pointwise Layers 的额外表达能力依赖中间非线性，而不只是增加矩阵乘法次数。

### 1.7 Inception 的多分支、降维与 Factorization

**原理：** Inception 让同一输入同时经过不同分支，例如 $1\times1$、$3\times3$、$5\times5$ Convolution 和 Pooling，再沿 channel 维 Concatenation。不同分支提取不同尺度的信息。为了控制成本，昂贵的 spatial Convolution 前先用 $1\times1$ 减少 channels。后续版本加入 BatchNorm、用小 kernel 替代大 kernel，并用非对称 kernel 分解二维空间运算。各分支拼接时必须有相同空间尺寸，但 channels 数可以不同。Auxiliary Classifiers 在中间层增加训练监督，推理时使用主分类输出。

**公式推导：** 设各分支输出为 $F_i(x)\in\mathbb{R}^{H\times W\times C_i}$，则

$$
y=\operatorname{Concat}(F_1(x),\ldots,F_B(x)),
\qquad C_{\mathrm{out}}=\sum_{i=1}^{B}C_i.
$$

其中 $B$ 是分支数。若直接从 $M$ 个输入 channels 用 $K\times K$ Conv 产生 $N$ 个输出 channels，需要 $K^2MN$ 个 weights。先降到 $R$ 个 channels，则两层合计

$$
P_{\mathrm{reduce}}=MR+K^2RN.
$$

第一项来自 $1\times1$ 降维，第二项来自较窄的 spatial Conv；只有当这个和小于 $K^2MN$ 时才真正节省参数。另一种 Factorization 在输入、中间、输出均为 $C$ channels 时，把 $n\times n$ 换成 $n\times1$ 后接 $1\times n$，参数由

$$
n^2C^2\quad\longrightarrow\quad nC^2+nC^2=2nC^2.
$$

两步分别覆盖竖直和水平方向，因此总体 Receptive Field 仍为 $n\times n$。这是结构化替代，不保证能表示原来任意二维 kernel。下采样也可让 stride-2 Conv 与 Pooling 并行后拼接；Inception-ResNet 则进一步在多分支变换外加入 Residual Addition。若主分类损失为 $L_0$，辅助分类损失为 $L_1,L_2$，训练目标写成 $L=L_0+\lambda_1L_1+\lambda_2L_2$，其中 $\lambda_1,\lambda_2$ 控制辅助监督强度。共享主干参数 $\theta$ 的梯度相应为 $\nabla_\theta L=\nabla_\theta L_0+\lambda_1\nabla_\theta L_1+\lambda_2\nabla_\theta L_2$，即中间监督也向主干提供训练信号。

### 1.8 ResNet、Projection 与 Pre-activation

**原理：** 更深的 Plain Network 有时连训练误差都更高，称为 Degradation。ResNet 不要求一组层重新生成全部输出，而是让它学习输入需要增加的修正量，再通过 Shortcut 把输入加回来。Basic Block 使用两个 $3\times3$ Conv；Bottleneck 使用 $1\times1$ 降维、$3\times3$ 处理、$1\times1$ 升维。Full Pre-activation 将 BN 和 ReLU 放在 Conv 前，并取消 Addition 后的 Activation，使主通路更接近完整的 Identity Mapping。

**公式推导：** 对期望映射 $H(x)$，定义 Residual 为 $F(x)=H(x)-x$，于是

$$
H(x)=x+F(x).
$$

若目标就是保留输入，只需令 $F(x)=0$。这解释了 Identity 的表达方式，但“更容易优化”仍是设计动机与实验结论，不是由等式自动保证。相加要求 shape 相同；若尺寸或 channels 改变，则用 Projection $P(x)$ 对齐：

$$
y=\operatorname{ReLU}(P(x)+F(x)).
$$

通常 shape 一致时 $P(x)=x$，否则用 $1\times1$ Conv，必要时设置 Stride 2。对于不在 Addition 后接 Activation 的 block，令 $J_F=\partial F/\partial x$，有

$$
y=x+F(x)
\quad\Rightarrow\quad
\frac{\partial y}{\partial x}=I+J_F.
$$

这里 $I$ 表示梯度可以通过 Shortcut 直接传播。若 Addition 后还有 ReLU，则导数变为

$$
\frac{\partial y}{\partial x}
=D_{\mathrm{ReLU}}(x+F(x))(I+J_F),
$$

其中 $D_{\mathrm{ReLU}}$ 是由各元素 ReLU 导数组成的对角矩阵。Full Pre-activation 避免这个位于主通路上的额外截断；若把 ReLU 放在 Residual 分支末端，则又会限制 $F(x)$ 只能非负，无法直接表示负向修正。

### 1.9 DenseNet 与特征增长

**原理：** ResNet 用 Addition 合并新旧特征，DenseNet 则保留此前所有层的输出，并用 Concatenation 一起提供给后续层。后层可以直接读取早期细节，也可以使用后来提取的高级特征，这称为 Feature Reuse。一个 Dense Block 内保持空间尺寸一致；不同 Blocks 之间使用 Transition Layer，例如 BN、$1\times1$ Conv 和 Average Pooling，调整 channels 并下采样。保留大量中间特征可能增加 Activation Memory，参数较少不等于显存一定较少。

**公式推导：** 设 $x_0$ 是 Dense Block 输入，第 $l$ 层只产生一组新特征 $x_l$，则

$$
x_l=H_l\bigl(\operatorname{Concat}(x_0,x_1,\ldots,x_{l-1})\bigr),
$$

其中 $H_l$ 是该层的可学习变换。若 $x_0$ 有 $C_0$ 个 channels，每层固定生成 $k$ 个新 channels，$k$ 称为 Growth Rate，那么第 $l$ 层输入包括初始特征及前面 $l-1$ 层的新增特征：

$$
C_{\mathrm{in},l}=C_0+(l-1)k.
$$

经过 $L$ 层并拼接所有结果，得到

$$
C_{\mathrm{out}}=C_0+\underbrace{k+\cdots+k}_{L\text{ 次}}=C_0+Lk.
$$

因此 block 的 channels 随层数线性增长，而单层输出仍是 $k$ channels。Concat 不对数值求和，只把向量接长；若空间尺寸不同，对应位置无法直接排列在一起，所以跨分辨率需要先变换尺寸。

### 1.10 ResNeXt 与 Grouped Convolution

**原理：** ResNeXt 通过多个结构相同的低维分支完成变换，再相加并接入 Residual Shortcut。分支数称为 Cardinality，与串行层数 Depth、每层 channels 数 Width 是不同维度。实际实现通常用 Grouped Convolution：把 channels 分组，每组独立执行 spatial Conv，再由 Pointwise Conv 混合特征。它与 Inception 的区别主要是分支结构高度统一，减少了逐个设计不同 kernel 分支的需求。

**公式推导：** 设有 $G$ 个分支，第 $i$ 个分支的变换为 $T_i(x)$，每个分支输出都与 $x$ 同 shape，则

$$
y=x+\sum_{i=1}^{G}T_i(x).
$$

这里使用 Addition，因此分支数增加不会直接让输出 channels 变为原来的 $G$ 倍。对一个 $K\times K$ Grouped Conv，若输入 $M$、输出 $N$ 个 channels 都均分为 $G$ 组，每组有 $M/G$ 个输入和 $N/G$ 个输出 channels，忽略 Bias 后：

$$
P=G\left(K^2\frac{M}{G}\frac{N}{G}\right)
=\frac{K^2MN}{G}.
$$

MACs 再乘输出空间面积 $H_oW_o$。普通 Conv 对应 $G=1$；当 $G=M$ 且 $N=M$ 时，每个输入 channel 单独使用一个 filter，就是 multiplier 为 1 的 Depthwise Conv。增加 Cardinality 在特定预算下可能比增深、增宽更有效，但参数公式本身只能说明连接与成本，不能证明准确率一定提高。

### 1.11 MobileNet-v1 与 Depthwise Separable Convolution

**原理：** 普通 Conv 同时处理空间关系和 channels 关系。MobileNet-v1 将这两件事拆开：Depthwise Conv 对每个 channel 独立提取局部空间信息，随后 Pointwise Conv 组合不同 channels，并决定输出 channels 数。两步合称 Depthwise Separable Convolution。只做第一步无法建立不同 channels 之间的信息交换，因此不能把 Depthwise 本身当作完整替代。

**公式推导：** 设输入、输出 channels 分别为 $M,N$，kernel 为 $K\times K$，最终空间尺寸为 $D_F\times D_F$。忽略 Bias、BN 和 Activation，普通 Conv 的 MACs 为

$$
C_{\mathrm{std}}=D_F^2K^2MN.
$$

Depthwise 每个输出位置、每个输入 channel 只计算 $K^2$ 项；随后 stride-1 Pointwise 在每个位置执行 $M\times N$ 的 channel 变换，因此

$$
C_{\mathrm{sep}}=\underbrace{D_F^2K^2M}_{\mathrm{Depthwise}}
+\underbrace{D_F^2MN}_{\mathrm{Pointwise}}.
$$

相除并分别约去共同因子，得到

$$
\frac{C_{\mathrm{sep}}}{C_{\mathrm{std}}}
=\frac{K^2M+MN}{K^2MN}
=\frac1N+\frac1{K^2}.
$$

参数量也分别为 $K^2MN$ 和 $K^2M+MN$，具有相同比值。当 $K=3$、$N$ 较大时，剩余计算比例接近 $1/9$。若比值为 $q$，减少比例是 $1-q$，理论 MACs 加速倍数是 $1/q$；实际运行速度还取决于硬件利用率与内存访问。

### 1.12 MobileNet-v2 的 Inverted Residual 与 Linear Bottleneck

**原理：** MobileNet-v2 先用 $1\times1$ Conv 扩张 channels，再在高维空间执行 Depthwise Conv，最后用 $1\times1$ 投影回较低维度。这种 Narrow→Wide→Narrow 结构称为 Inverted Bottleneck，区别于 ResNet 的 Wide→Narrow→Wide。输出投影后不接 ReLU，称为 Linear Bottleneck；这样可以减少低维表示被非线性截断造成的信息损失。当 Stride 为 1 且输入输出 shape 相同时，Shortcut 连接低维两端。

**公式推导：** 设输入有 $C$ 个 channels，Expansion Ratio 为 $t$，输出为 $C'$ 个 channels，简化省略 BN 后写成

$$
u=\operatorname{ReLU6}(\operatorname{PW}_{C\to tC}(x)),
$$

$$
v=\operatorname{ReLU6}(\operatorname{DWConv}(u)),
\qquad F(x)=\operatorname{PW}_{tC\to C'}(v).
$$

其中 PW 是 Pointwise Conv，DWConv 是 Depthwise Conv，$\operatorname{ReLU6}(a)=\min(\max(a,0),6)$。在 $C'=C$ 且空间尺寸一致时输出 $y=x+F(x)$，否则只输出 $F(x)$。保留三层且忽略 Bias 时，参数量依次为 $tC^2$、$K^2tC$、$tCC'$，总计

$$
P=tC^2+K^2tC+tCC'.
$$

Linear Bottleneck 的动机可由 ReLU 的多对一性质理解：不同负数都会映射到零，即 $\operatorname{ReLU}(-1)=\operatorname{ReLU}(-2)=0$。低维投影后再截断会进一步合并不同表示；取消末端 Activation 避免这一步额外损失，但不意味着整个 block 可逆或完全不丢信息。

### 1.13 EfficientNet 的 Compound Scaling

**原理：** 扩大 CNN 可以增加 Depth、Width 或输入 Resolution。只扩其中一个维度可能形成容量或输入信息的瓶颈；EfficientNet 用 Compound Scaling 协调三个维度，并从较小的基准模型搜索比例，再按预算扩大网络。这是一种经验有效的架构缩放规则，而不是保证任何任务准确率最优的定理。扩展后的模型仍需要训练，复用的是架构与缩放规则。

**公式推导：** 设深度、宽度、分辨率倍率分别为 $d,w,r$。深度增加 $d$ 倍，相当于类似操作重复 $d$ 倍；宽度使输入和输出 channels 都乘 $w$；分辨率使高和宽都乘 $r$。因此对主要由普通 Conv 构成的计算，近似有

$$
\frac{C_{\mathrm{new}}}{C_{\mathrm{base}}}\approx dw^2r^2.
$$

令 $d=\alpha^\phi$、$w=\beta^\phi$、$r=\gamma^\phi$，其中 $\alpha,\beta,\gamma\ge1$ 是固定缩放比例，$\phi$ 控制资源规模，代入得到

$$
dw^2r^2=(\alpha\beta^2\gamma^2)^\phi.
$$

如果搜索时约束 $\alpha\beta^2\gamma^2\approx2$，则 $\phi$ 每增加 1，理论计算预算约翻倍。实际构造某个 stage 时，若基准 channels、重复数和分辨率分别为 $C_0,L_0,R_0$，则

$$
C'\approx wC_0,\qquad L'=\lceil dL_0\rceil,\qquad R'\approx rR_0.
$$

channels 常对齐到 8 的倍数，层数需要取整；Depthwise 与 Pointwise 的缩放规律也不完全相同，所以实际 FLOPs 不会严格遵循这一近似等式。

### 1.14 MBConv 与 Squeeze-and-Excitation

**原理：** EfficientNet 的核心 MBConv 沿用 Inverted Bottleneck，并加入 Squeeze-and-Excitation（SE）。SE 先将每个 channel 的空间响应汇总成一个数，再根据整组 channel 描述产生权重，最后重新缩放原 feature map。它学习“当前输入中哪些 channels 更重要”；同一个 channel 的所有位置共享一个权重。MBConv1 的 Expansion Ratio 为 1，通常省略最初的扩张 Conv；MBConv6 则扩张到输入 channels 的 6 倍。

**公式推导：** 设进入 SE 的特征为 $U\in\mathbb{R}^{H\times W\times C}$。Squeeze 使用 GAP 得到行向量 $z\in\mathbb{R}^{1\times C}$：

$$
z_c=\frac1{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}U_{h,w,c}.
$$

一种常用 Excitation 写法为

$$
s=\operatorname{sigmoid}\bigl(\sigma(zW_1+b_1)W_2+b_2\bigr),
$$

其中 $W_1\in\mathbb{R}^{C\times r}$、$W_2\in\mathbb{R}^{r\times C}$，$r$ 是内部较小维度，$\sigma$ 是 Activation。两次线性变换允许不同 channels 的统计量共同影响某一个 gate。Sigmoid 将各元素限制在 $(0,1)$，输出为

$$
\widetilde U_{h,w,c}=s_cU_{h,w,c}.
$$

因为 $s_c$ 没有 $h,w$ 下标，它在空间维广播，属于 Channel Reweighting。忽略 Bias，SE 中两个线性层共需 $Cr+rC=2Cr$ 个 weights。MBConv 的整体路径是 Expansion→Depthwise Conv→SE→Linear Projection；只有输入输出 shape 一致时，才能再与输入作 Residual Addition。

### 1.15 ConvNeXt 的结构与计算分工

**原理：** ConvNeXt 保留 CNN 的局部空间运算，但调整 block 的分工：先用 $7\times7$ Depthwise Conv 进行 Spatial Mixing，再用 LayerNorm 和两层 Pointwise Conv 进行 Channel Mixing，中间只保留一次 GELU。Pointwise 中间 channels 扩张为原来的 4 倍，最后投影回来并加 Shortcut。与 MobileNet-v2 不同，Depthwise 在扩张之前执行。网络入口用 $4\times4$、Stride 4 的 Patchify Stem，stage 之间用独立的 LN 和 stride-2 Conv 下采样。

**公式推导：** 设输入有 $C$ 个 channels，同一 block 保持空间尺寸。其主要运算可以写成

$$
u=\operatorname{LN}(\operatorname{DWConv}_{7\times7}(x)),
$$

$$
v=\operatorname{PW}_{4C\to C}
\bigl(\operatorname{GELU}(\operatorname{PW}_{C\to4C}(u))\bigr),
\qquad y=x+\eta\odot v.
$$

其中 $\eta$ 是可选的、按 channel 学习的 Layer Scale，$\odot$ 表示逐元素相乘并在空间维广播；没有 Layer Scale 时取 $\eta=1$。忽略 Bias、LN 和 Layer Scale 的参数，Depthwise 使用 $49C$ 个 weights，两层 Pointwise 分别使用 $4C^2$，总计

$$
P=49C+4C^2+4C^2=49C+8C^2.
$$

如果把同一个 $7\times7$ Depthwise 移到扩张之后，它需要处理 $4C$ 个 channels，成本项便从 $49C$ 增为 $196C$。这解释了先做 Spatial Mixing 的效率原因。LN 在每个位置的 channels 上计算统计量，GELU 只位于两个 Pointwise Layers 之间；这些位置是架构定义的一部分，不能随意调换后仍认为是同一个 block。

## 二、Attention 与原版 Transformer

### 2.1 Attention：根据当前需求，对输入信息加权

**原理：** 传统 Seq2seq 将整段输入压缩成一个固定长度的 Context Vector，长序列的信息容易在压缩中丢失。Attention 保留每个输入位置的表示；Decoder 每生成一个位置，就重新计算哪些输入更相关，再进行加权求和。用于 Image Captioning 时，输入位置变成图像上的空间位置，原理相同。

**公式推导：** 设 $h_i$ 是第 $i$ 个输入位置的表示，$s_{t-1}$ 是 Decoder 上一步的状态。先计算相关性分数，再将分数变为非负、总和为 1 的权重：

$$
e_{t,i}=\operatorname{score}(s_{t-1},h_i),\qquad
\alpha_{t,i}=\frac{e^{e_{t,i}}}{\sum_j e^{e_{t,j}}}.
$$

于是当前所需的信息为

$$
c_t=\sum_i\alpha_{t,i}h_i,\qquad
s_t=f(s_{t-1},y_{t-1},c_t),
$$

其中 $c_t$ 是当前 Context Vector，$y_{t-1}$ 是前一个输出。由于 $\sum_i\alpha_{t,i}=1$，$c_t$ 就是根据当前需求计算的加权平均。

对行向量 $q,k$，常见分数定义为

$$
\begin{aligned}
\text{Dot-product: }&qk^\top,\\
\text{Cosine: }&\frac{qk^\top}{\|q\|\|k\|},\\
\text{Bilinear: }&qWk^\top,\\
\text{Additive: }&\tanh([q,k]W_a+b_a)w_a.
\end{aligned}
$$

Cosine 去除长度影响；Bilinear 学习匹配矩阵；Additive 将两个向量拼接后，用一个小型网络输出单个分数。$W,W_a,w_a,b_a$ 均为可学习参数。

### 2.2 Q、K、V 与 Scaled Dot-product Attention

**原理：** Query 表示当前要查找的内容，Key 表示各位置用于匹配的特征，Value 表示被读取的信息。Q 与 K 决定读取权重，再用这些权重汇总 V。Self-attention 的三者来自同一序列，但经过不同的可学习投影。

**公式推导：** 本文的矩阵默认每行表示一个 Token。输入 $X\in\mathbb R^{n\times d}$，用三个 Linear Projection 得到

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

其中 $W_Q,W_K\in\mathbb R^{d\times d_k}$，$W_V\in\mathbb R^{d\times d_v}$；$n$ 是 Token 数，$d$ 是输入维度，$d_k,d_v$ 是 Key 和 Value 的维度。

每个 Query 与每个 Key 做点积，因此

$$
S=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb R^{n\times n}.
$$

对 $S$ 的每一行做 Softmax，得到该 Query 对所有 Key 的权重；再汇总 Value：

$$
A_{ij}=\frac{e^{S_{ij}}}{\sum_l e^{S_{il}}},\qquad
O_i=\sum_jA_{ij}V_j.
$$

将所有位置写成矩阵，便得到

$$
\boxed{O=\operatorname{softmax}_{\rm row}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
\in\mathbb R^{n\times d_v}.
$$

单个 Head 的 $QK^\top$ 需要计算 $n^2$ 个长度为 $d_k$ 的点积；乘 V 再需约 $n^2d_v$ 次乘加。因此 Attention 核心计算为 $O(n^2(d_k+d_v))$，显式权重矩阵需要 $O(n^2)$ 存储。

### 2.3 为什么 Attention 要除以 $\sqrt{d_k}$

**原理：** 向量维度越大，点积的典型幅度越大，容易让 Softmax 将权重压成接近 One-hot，导致梯度很小。除以 $\sqrt{d_k}$ 是为了让点积的统计尺度不随维度增长。

**公式推导：** 假设 $q_i,k_i$ 相互独立，均值为 0、方差为 1。对单个乘积：

$$
\mathbb E[q_ik_i]=0,
\qquad
\operatorname{Var}(q_ik_i)
=\mathbb E[q_i^2]\mathbb E[k_i^2]=1.
$$

点积是 $d_k$ 个独立项之和，所以

$$
z=qk^\top=\sum_{i=1}^{d_k}q_ik_i,
\qquad \operatorname{Var}(z)=d_k.
$$

利用 $\operatorname{Var}(az)=a^2\operatorname{Var}(z)$：

$$
\operatorname{Var}\left(\frac{z}{\sqrt{d_k}}\right)
=\frac{d_k}{d_k}=1.
$$

这里缩放的是单个 Head 的 Key Dimension；上述结论依赖给定的独立性与方差假设。

### 2.4 Softmax 的稳定计算与梯度

**原理：** Softmax 将任意分数转换为概率。决定概率的是分数之间的差距。差距过大时，最大值获得接近 1 的概率，其余概率接近 0，此时输出对分数变化不敏感。

**公式推导：** 对 $p_i=e^{z_i}/Z$，其中 $Z=\sum_j e^{z_j}$，使用商法则：
$$
\frac{\partial p_i}{\partial z_j}
=\frac{\delta_{ij}e^{z_i}Z-e^{z_i}e^{z_j}}{Z^2}
=p_i(\delta_{ij}-p_j).
$$

$\delta_{ij}$ 在 $i=j$ 时为 1，否则为 0。特别地，$\partial p_i/\partial z_i=p_i(1-p_i)$；当 $p_i$ 接近 0 或 1 时，导数接近 0。

为避免指数溢出，可以减去最大分数 $m=\max_jz_j$：

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
=\frac{e^{-m}e^{z_i}}{e^{-m}\sum_j e^{z_j}}
=p_i.
$$

这样只改变数值计算方式，不改变概率；它不能消除分数差距过大造成的饱和。

### 2.5 Causal Mask 与右移输入

**原理：** Autoregressive 模型预测下一 Token 时只能读取已有前文。Causal Mask 屏蔽未来位置；右移输入则保证当前位置的输入也不是正在预测的答案。Padding Mask 另外屏蔽为了补齐长度而加入的 PAD。

**公式推导：** 若目标序列为 $[y_1,y_2,\ldots,y_T]$，Decoder 输入为 $[\mathrm{BOS},y_1,\ldots,y_{T-1}]$。第 $t$ 个输入位置预测 $y_t$，只允许关注 $j\le t$：
$$
M_{tj}=\begin{cases}
0,&j\le t,\\
-\infty,&j>t.
\end{cases}
$$

将 Mask 加到 Softmax 前的分数上：

$$
A_{tj}=\frac{\exp(S_{tj}+M_{tj})}
{\sum_l\exp(S_{tl}+M_{tl})}.
$$

被屏蔽位置因 $e^{-\infty}=0$ 而得到零权重。因此各位置在训练中可并行计算，且都只使用相应前文；推理时仍需逐个生成未知的 Token。

### 2.6 Cross-attention

**原理：** Self-attention 在同一序列内部交换信息；Cross-attention 让 Decoder 根据当前生成需求，读取 Encoder 表示的输入信息。因此 Q 来自 Decoder，K 和 V 来自 Encoder。

**公式推导：** 设 Decoder 表示 $Y\in\mathbb R^{m\times d}$，Encoder 表示 $E\in\mathbb R^{n\times d}$：

$$
Q=YW_Q\in\mathbb R^{m\times d_k},\quad
K=EW_K\in\mathbb R^{n\times d_k},\quad
V=EW_V\in\mathbb R^{n\times d_v}.
$$

于是

$$
QK^\top\in\mathbb R^{m\times n},\qquad
\operatorname{softmax}_{\rm row}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\in\mathbb R^{m\times d_v}.
$$

$m$ 个输出位置分别对 $n$ 个输入位置分配权重。Attention Matrix 可以是长方形；输出位置数由 Q 决定，每个位置的输出维度由 V 决定。

### 2.7 Multi-head Attention

**原理：** 单个 Head 只学习一组匹配和信息汇总规则。Multi-head Attention 使用多组独立投影，让不同 Head 在不同特征子空间中学习关系，再将结果拼接、混合。

**公式推导：** 第 $i$ 个 Head 为

$$
H_i=\operatorname{Attention}(X_QW_{Q,i},X_KW_{K,i},X_VW_{V,i}).
$$

其中 $X_Q,X_K,X_V$ 的来源由 Self-attention 或 Cross-attention 决定。若共有 $h$ 个 Head，每个输出 $d_v$ 维，则拼接后宽度为 $hd_v$：

$$
O=\operatorname{Concat}(H_1,\ldots,H_h)W_O,
\qquad W_O\in\mathbb R^{hd_v\times d}.
$$

通常设置 $d_k=d_v=d/h$。此时每种 Q/K/V 投影的总参数均为

$$
h\cdot d\cdot\frac dh=d^2.
$$

再加上 $W_O$ 的 $d^2$ 个参数，MHA 的 Weight 参数量为 $4d^2$。因此固定 $d$ 时，增加 Head 数主要是将每个 Head 变窄；忽略 Bias 后不会增加这部分参数量。

### 2.8 Position-wise FFN

**原理：** Attention 负责不同 Token 之间的信息交换，FFN 对每个 Token 的特征进行非线性变换。它先升维，组合出更多中间特征，再降回原维度。所有 Token 共享同一套 FFN 参数。

**公式推导：** 对行向量 $x\in\mathbb R^{1\times d}$，先计算 $xW_1+b_1\in\mathbb R^{1\times f}$，经过 ReLU 后再映射回 $d$ 维：

$$
\operatorname{FFN}(x)
=\operatorname{ReLU}(xW_1+b_1)W_2+b_2,
$$

其中 $W_1\in\mathbb R^{d\times f}$，$W_2\in\mathbb R^{f\times d}$，$f$ 是中间维度。参数量为

$$
df+f+fd+d=2df+f+d.
$$

通常 $f=4d$，忽略 Bias 后得到 $8d^2$。若去掉中间的非线性，两层可合并为 $x(W_1W_2)$，就失去了通过堆叠获得的非线性表达能力。

### 2.9 LayerNorm 与 BatchNorm

**原理：** Normalization 调整特征的数值尺度。LayerNorm 对一个 Token 自己的所有特征计算统计量；BatchNorm 通常对同一 Channel 跨样本计算统计量，在 CNN 中还包含空间位置。因此 LayerNorm 不依赖当前 Batch 中其他样本。

**公式推导：** 对一个 Token 的 $d$ 维向量 $x$，先求均值和方差：

$$
\mu=\frac1d\sum_{i=1}^dx_i,\qquad
\sigma^2=\frac1d\sum_{i=1}^d(x_i-\mu)^2.
$$

减去均值使平均值为 0，除以标准差使方差约为 1，再用可学习的缩放与平移恢复表达能力：

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

$\epsilon>0$ 防止除零，$\gamma_i,\beta_i$ 为共享于所有 Token 的参数。在缩放和平移之前，归一化结果的方差为 $\sigma^2/(\sigma^2+\epsilon)\approx1$。BatchNorm 使用相同形式，但统计量的求和范围换成同一 Channel 的 Batch 和空间位置。

### 2.10 Transformer Encoder 与 Decoder

**原理：** Encoder 在完整输入中建立上下文表示，由 Self-attention 和 FFN 组成。Decoder 先读取已生成的前文，再通过 Cross-attention 查询 Encoder，最后经过 FFN。原版 Transformer 的每个子层都使用 Residual Connection 和 Post-LN；Residual 相加要求输入与输出维度一致。

**公式推导：** 记 $\operatorname{MHA}(X_Q,X_K,X_V)$ 内部包含各自的 Q/K/V 投影。一个 Encoder Block 可逐步写成

$$
U=\operatorname{LN}\bigl(X+\operatorname{MHA}(X,X,X)\bigr),
$$

$$
E=\operatorname{LN}\bigl(U+\operatorname{FFN}(U)\bigr).
$$

设最终 Encoder 输出为 $E$，一个 Decoder Block 则为

$$
\begin{aligned}
U&=\operatorname{LN}\bigl(Y+\operatorname{MHA}_{\rm causal}(Y,Y,Y)\bigr),\\
T&=\operatorname{LN}\bigl(U+\operatorname{MHA}(U,E,E)\bigr),\\
Z&=\operatorname{LN}\bigl(T+\operatorname{FFN}(T)\bigr).
\end{aligned}
$$

$Y$ 是当前 Block 的 Decoder 输入。多个 Block 堆叠后，最后位置的表示 $z_t$ 经词表投影和 Softmax：

$$
P(y_t\mid y_{<t},X)
=\operatorname{softmax}(z_tW_{\rm vocab}+b_{\rm vocab}),
$$

其中词表大小为 $V$ 时，$W_{\rm vocab}\in\mathbb R^{d\times V}$。以上省略 Dropout，以突出信息流。

### 2.11 Sinusoidal Positional Encoding

**原理：** 只依赖内容的 Self-attention 不知道位置顺序，因此需要添加位置信息。Sinusoidal Positional Encoding 对每对特征维度使用同一频率的 Sin/Cos，不同维度对使用不同频率。

**公式推导：** 位置为 $p$，模型维度为偶数 $d$，维度对索引 $i=0,\ldots,d/2-1$。定义 $\omega_i=10000^{-2i/d}$：

$$
PE(p,2i)=\sin(p\omega_i),\qquad
PE(p,2i+1)=\cos(p\omega_i).
$$

这是位置表示的设计定义。它的一个关键性质来自三角恒等式。位置移动 $\Delta$ 后：

$$
\begin{bmatrix}
\sin((p+\Delta)\omega_i)\\
\cos((p+\Delta)\omega_i)
\end{bmatrix}
=
\begin{bmatrix}
\cos(\Delta\omega_i)&\sin(\Delta\omega_i)\\
-\sin(\Delta\omega_i)&\cos(\Delta\omega_i)
\end{bmatrix}
\begin{bmatrix}
\sin(p\omega_i)\\
\cos(p\omega_i)
\end{bmatrix}.
$$

因此固定的位置偏移对应固定的线性变换，有利于模型表达相对位置。将 $PE(p)$ 与同为 $d$ 维的 Token Embedding 相加，维度保持不变。

## 三、Modern Transformer

### 3.1 Post-LN 与 Pre-LN：Normalization 的位置如何影响梯度

**原理：** Transformer Block 由 Attention、FFN 和 Residual Connection 组成。Attention 让不同 Token 交换信息，FFN 分别加工每个 Token 的特征；Residual Connection 把输入直接加回输出，使原信息不必完全经过复杂变换。Post-LN 在相加之后做 LayerNorm，Pre-LN 在分支计算之前做 LayerNorm，因此两者的梯度路径不同。

Pre-LN 的主通路保留了直接传递输入的分支，梯度可以通过这条路径跨越多层；Post-LN 的所有路径都经过相加后的 LayerNorm，其梯度会受 Normalization 的 Jacobian 影响。这解释了 Pre-LN 通常更容易训练，但不代表任意学习率、深度下都不会发散。Pre-LN 堆叠通常还在最终输出之前增加一次 Normalization。

**公式推导：** 令 $x$ 为一个 Token 的特征，$F$ 为 Attention 或 FFN 子层，定义两种结构：

$$
\text{Post-LN:}\quad y=\operatorname{LN}(x+F(x))
$$

$$
\text{Pre-LN:}\quad y=x+F(\operatorname{LN}(x)).
$$

用 $J_F$、$J_{\operatorname{LN}}$ 表示相应函数的 Jacobian，$I$ 为恒等矩阵。根据 Chain Rule：

$$
J_{\text{Post-LN}}=J_{\operatorname{LN}}(I+J_F),
\qquad
J_{\text{Pre-LN}}=I+J_FJ_{\operatorname{LN}}.
$$

Pre-LN 的导数中直接出现 $I$；Post-LN 的 $I$ 仍需经过外层 Normalization。这是“直接梯度通路”的数学含义。一个完整的 Pre-LN Block 为：

$$
u=x+\operatorname{Attention}(\operatorname{LN}(x)),
\qquad
y=u+\operatorname{FFN}(\operatorname{LN}(u)).
$$

### 3.2 RMSNorm：只控制尺度，不减去均值

**原理：** LayerNorm 对同一个 Token 的特征计算均值与方差，先减均值、再除标准差。RMSNorm 使用均方根控制向量整体尺度，省略减均值操作，因此计算与数据访问更简单。它保留了稳定数值尺度的作用，但不保证输出均值为零，也不具有对输入平移的不变性。

Normalization 通常还包含可学习的 Gain，让模型在统一尺度之后重新调整各特征的重要性。公式中的小正数 $\epsilon$ 防止分母为零；它也意味着实际操作只近似具有理想的尺度不变性。

**公式推导：** 对 $x=(x_1,\ldots,x_d)$，定义：

$$
\mu=\frac1d\sum_i x_i,
\qquad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2.
$$

展开平方并利用 $\sum_i x_i=d\mu$：

$$
\sigma^2
=\frac1d\sum_i x_i^2-\mu^2,
\qquad
\operatorname{RMS}(x)=\sqrt{\frac1d\sum_i x_i^2}
=\sqrt{\sigma^2+\mu^2}.
$$

因此两种 Normalization 分别为：

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,
\qquad
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.
$$

其中 $\gamma_i$ 为 Learnable Gain，$\beta_i$ 为 Bias。当 $\mu=0$、Bias 为零且其他设置一致时，两式对应。忽略 $\epsilon$，对正数 $a$：

$$
\operatorname{RMS}(ax)=a\operatorname{RMS}(x)
\quad\Rightarrow\quad
\frac{ax}{\operatorname{RMS}(ax)}
=\frac{x}{\operatorname{RMS}(x)}.
$$

这说明正比例放大输入不会改变归一化结果。pRMSNorm 进一步只用 $k=\lceil pd\rceil$ 个特征估计 $\sqrt{\frac1k\sum_{i=1}^kx_i^2}$，以较少计算换取近似统计量。

### 3.3 QK-Norm：防止 Attention logits 过度放大

**原理：** Pre-Norm 约束的是进入子层的输入，后面的 Query、Key 投影仍可能将它放大。Q 与 K 的点积变大后，Softmax 容易只关注某一个 Token，其他 Token 的权重与相关梯度趋近于零。QK-Norm 因而在每个 Head 内再次归一化 Q/K，通常放在 Projection 之后、RoPE 之前；V 不因此被归一化。

Pre-Norm 在模型特征维度上操作，QK-Norm 在每个 Head 的维度上操作，两者解决不同位置的尺度问题。Stable Softmax 减去最大 logit 可以避免直接指数溢出，却不能解决概率分布已经饱和的问题。

**公式推导：** 采用行向量约定，$Q=XW_Q$、$K=XW_K$，单个 logit 为：

$$
s_{ij}=\frac{q_i k_j^\top}{\sqrt{d_k}}.
$$

若两个投影权重同时放大 $c$ 倍，则：

$$
q_i'=cq_i,\quad k_j'=ck_j
\quad\Rightarrow\quad s_{ij}'=c^2s_{ij}.
$$

对一行 logits，$p_i=e^{s_i}/\sum_j e^{s_j}$。由 Quotient Rule 得：

$$
\frac{\partial p_i}{\partial s_j}
=p_i(\delta_{ij}-p_j).
$$

当一个概率接近 1、其他接近 0，上式各项趋近于零。QK-RMSNorm 改为使用：

$$
\widetilde q=\gamma_Q\odot
\frac{q}{\sqrt{\frac1{d_k}\sum_aq_a^2+\epsilon}},
\qquad
\widetilde k=\gamma_K\odot
\frac{k}{\sqrt{\frac1{d_k}\sum_ak_a^2+\epsilon}}.
$$

若忽略 $\epsilon$ 且 Gain 全为 1，则两者范数均为 $\sqrt{d_k}$。由 Cauchy–Schwarz：

$$
\left|\frac{\widetilde q\widetilde k^\top}{\sqrt{d_k}}\right|
\le\frac{\|\widetilde q\|\|\widetilde k\|}{\sqrt{d_k}}
=\sqrt{d_k}.
$$

存在任意 Learnable Gain 时，上界还需乘 $\|\gamma_Q\|_\infty\|\gamma_K\|_\infty$，因此不能忽略 Gain 而声称无条件固定上界。

### 3.4 SwiGLU：用 Gate 控制 FFN 特征

**原理：** 普通 FFN 先把特征升维，经过 Activation，再降回原维度。SwiGLU 将升维阶段分成两条路径：一条生成内容，另一条经过 Swish 形成 Gate；两条路径逐元素相乘，再做输出投影。这使每个 Token 的不同特征能够被动态放大或抑制。

Swish 又称 SiLU，是平滑且允许小负输出的 Activation。SwiGLU 的 Gate 不是概率，不必处于零到一之间；它控制特征，不是在选择 MoE Experts。采用三个矩阵会增加同宽度下的参数量，因此通常缩小中间维度，使其与普通 FFN 的计算预算相近。

**公式推导：** 对 $x\in\mathbb R^{1\times d}$，普通 FFN 的中间宽度为 $m_0$：

$$
\operatorname{FFN}(x)=\phi(xW_1)W_2,
\quad W_1\in\mathbb R^{d\times m_0},\quad
W_2\in\mathbb R^{m_0\times d}.
$$

定义 $\sigma(z)=1/(1+e^{-z})$，则 $\operatorname{Swish}(z)=z\sigma(z)$。将内容与 Gate 相乘，得到：

$$
\operatorname{SwiGLU}(x)
=[\operatorname{Swish}(xW_g)\odot(xW_v)]W_o,
$$

其中 $W_g,W_v\in\mathbb R^{d\times m}$，$W_o\in\mathbb R^{m\times d}$，$\odot$ 表示逐元素乘法。忽略 Bias，参数量分别为：

$$
P_{\text{FFN}}=2dm_0,
\qquad P_{\text{SwiGLU}}=3dm.
$$

普通 FFN 常取 $m_0=4d$。令两者参数量相等：

$$
3dm=2d(4d)
\quad\Rightarrow\quad m=\frac83d.
$$

这个比例来自参数匹配；SwiGLU 的实际效果优势来自实验，不能由参数量相等直接证明。

### 3.5 FlashAttention：通过分块精确计算 Attention

**原理：** Standard Attention 会生成完整的 Token 两两关联矩阵，反复在较慢的大容量 HBM 中读写。FlashAttention 将 Q/K/V 分成能放入快速 SRAM 的小块，计算局部 Attention，同时维护可以合并的 Softmax 统计量。它改变计算顺序与存储方式，不删去允许关注的 Token Pair。

难点在于 Softmax 的分母依赖整行所有 Keys，不能对每块独立归一化后直接平均。Online Softmax 利用共同的最大值重新缩放旧结果，使每次合并都与完整计算一致。反向传播时可以按块重算概率，避免存储完整 Attention Matrix。

**公式推导：** 一行 Query 对所有 Keys 的 logits 为 $s_j=qk_j^\top/\sqrt{d_k}$，最终输出是：

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.
$$

分子分母同除 $e^m$，其中 $m=\max_js_j$，得到稳定形式：

$$
o=\frac{\sum_j e^{s_j-m}v_j}{\sum_j e^{s_j-m}}.
$$

对一个已处理的 Key 集合 $A$，维护标量 $m_A,\ell_A$ 和向量 $u_A$：

$$
m_A=\max_{j\in A}s_j,
\quad\ell_A=\sum_{j\in A}e^{s_j-m_A},
\quad u_A=\sum_{j\in A}e^{s_j-m_A}v_j.
$$

新块 $B$ 有同样的三项统计量。令 $m=\max(m_A,m_B)$，将两块统一到这个最大值：

$$
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,
\qquad
u=e^{m_A-m}u_A+e^{m_B-m}u_B.
$$

其精确性来自：

$$
e^{m_A-m}e^{s_j-m_A}=e^{s_j-m}.
$$

因此合并后的 $\ell,u$ 正好是集合 $A\cup B$ 的完整分母与分子，最终 $o=u/\ell$。每个 Query 独立维护这些状态，直到处理完所有 Key Blocks。

若序列长度为 $N$、Head Dimension 固定为 $d_h$，所有 Token Pair 仍被计算，FLOPs 仍为 $O(N^2d_h)$；但无需存储 $N\times N$ 的中间概率矩阵，主要 Attention 存储可降至 $O(Nd_h)$。Exact 表示相同数学运算，浮点重排仍可能产生微小舍入差异。

### 3.6 RoPE：让位置以相对角度进入点积

**原理：** RoPE 不把位置向量直接加到内容上，而是根据 Token 的位置旋转 Q/K。两个向量分别旋转后，它们的点积取决于旋转角的差，因此位置影响可以用相对距离表达。Rotation 保持向量范数，不会仅因位置变化放大特征。

高维向量被拆成多个二维对，不同维度使用不同频率：快频率表达细粒度位置变化，慢频率表达较长尺度的变化。多频率改善单个频率周期回绕造成的歧义，但不保证任何 Token Pair 的关联强度随距离单调下降，也不自动保证超长 Context 的质量。

**公式推导：** 本节为说明旋转，暂用二维列向量 $q,k$。位置 $m$ 对应旋转角 $m\theta$：

$$
R_m=\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix},
\qquad q_m=R_mq,\quad k_n=R_nk.
$$

由于旋转矩阵的转置等于逆矩阵，先反向转 $m\theta$ 再正向转 $n\theta$ 等于转 $(n-m)\theta$：

$$
R_m^\top R_n=R_{n-m}.
$$

因此：

$$
q_m^\top k_n
=q^\top R_m^\top R_nk
=q^\top R_{n-m}k.
$$

位置部分只通过 $n-m$ 出现，内容 $q,k$ 仍然影响结果。范数保持则来自：

$$
\|R_mq\|^2=q^\top R_m^\top R_mq=q^\top q.
$$

对偶数维 $d_h$，第 $i$ 对通常使用 $\theta_i=10000^{-2(i-1)/d_h}$。二维对 $(a,b)$ 在位置 $m$ 的输出为：

$$
(a\cos(m\theta_i)-b\sin(m\theta_i),\;
a\sin(m\theta_i)+b\cos(m\theta_i)).
$$

逐对计算只需要 $O(d_h)$ 操作，无需构造完整 Rotation Matrix。

### 3.7 MHA、MQA 与 GQA：减少 KV Cache

**原理：** Autoregressive Decoding 每次新增一个 Token。过去 Token 的 K/V 可以缓存，当前 Query 直接检索它们，不必重新投影整个历史；过去的 Q 通常无需缓存。随着 Context 和 Batch 增大，读取这些缓存成为重要开销。

MHA 为每个 Query Head 配置独立的 KV Head。MQA 让所有 Query Heads 共享一对 K/V，GQA 则把 Query Heads 分组，每组共享一对 K/V。三者保留多个 Query Heads，区别在于独立 KV Heads 的数量。共享越多，缓存越小，但表示能力可能受影响；这种共享是模型架构的一部分，需要相应训练。

**公式推导：** 设 Query Heads 数为 $H$，KV Heads 数为 $G$，第 $i$ 个 Query Head 属于组 $g(i)$：

$$
Q_i=XW_{Q,i},\quad K_g=XW_{K,g},\quad V_g=XW_{V,g},
$$

$$
O_i=\operatorname{softmax}\left(
\frac{Q_iK_{g(i)}^\top}{\sqrt{d_k}}\right)V_{g(i)},
\qquad O=\operatorname{Concat}(O_1,\ldots,O_H)W_O.
$$

MHA 对应 $G=H$；MQA 对应 $G=1$；GQA 通常为 $1<G<H$，每组有 $H/G$ 个 Query Heads。若 $d_k=d_v=d_h$，Prefill 的 Q Shape 为 $[B,H,N,d_h]$，K/V 各为 $[B,G,N,d_h]$；单步 Decoding 的 Query Length 通常是 1。

每个 Token 每层需要缓存 $Gd_k$ 个 Key 元素与 $Gd_v$ 个 Value 元素。设模型层数 $L$、Batch 为 $B$、历史长度 $N$、每元素 $b$ bytes，则：

$$
M_{\text{KV}}=LBNG(d_k+d_v)b.
$$

相同其他设置下：

$$
\frac{M_{\text{GQA}}}{M_{\text{MHA}}}=\frac GH,
\qquad
\frac{M_{\text{MQA}}}{M_{\text{MHA}}}=\frac1H.
$$

这只计算 KV Cache，不包含模型权重与临时工作区。若各序列长度不同，以 $\sum_{a=1}^BN_a$ 替代 $BN$。从 MHA 初始化 GQA 时，可对同组 KV 投影取平均，再通过 Uptraining 适应共享结构。

### 3.8 MLA：缓存低维 Latent，并吸收投影矩阵

**原理：** MLA 用一个低维 KV Latent 保存历史 Token 的共同信息，各 Head 通过不同投影读取它。与 MQA 共享同一套完整 K/V 不同，MLA 仍能形成 Head-specific 的内容表示。推理时利用线性乘法结合律，将展开 K/V 的矩阵移到 Query 或 Output 一侧，避免保存完整的多头 K/V。

普通 RoPE 会引入位置相关 Rotation，不便直接吸收到固定投影中。因此 MLA 将内容路径与位置路径分开：内容路径保持可折叠的线性关系，小维度的独立 Q/K 路径负责 RoPE。实际缓存必须同时包含 KV Latent 和 RoPE Key。

**公式推导：** 对行向量 $h_t\in\mathbb R^{1\times d}$，先得到 $d_c$ 维 Latent：

$$
c_t=h_tW_D,\qquad k_{t,h}^{C}=c_tW_{K,h},\qquad
v_{t,h}=c_tW_{V,h}.
$$

其中 $W_{K,h}\in\mathbb R^{d_c\times d_k}$、$W_{V,h}\in\mathbb R^{d_c\times d_v}$，$h$ 的下标表示 Head，$C$ 表示 Content。将历史 Latents 排成 $C\in\mathbb R^{N\times d_c}$，则：

$$
q_h^C(CW_{K,h})^\top
=(q_h^CW_{K,h}^\top)C^\top
=\bar q_h C^\top.
$$

新 Query $\bar q_h$ 的维度为 $d_c$，可直接检索 $C$。若 $q_h^C=xW_{Q,h}$，可以预先合并固定权重 $W_{Q,h}W_{K,h}^\top$。对于 Attention 概率 $p_h$ 和该 Head 的输出矩阵 $W_{O,h}$：

$$
(p_hCW_{V,h})W_{O,h}
=(p_hC)(W_{V,h}W_{O,h}).
$$

因此也可先在 Latent 空间汇总，再投影到输出维度。加入 $d_r$ 维的独立 RoPE 分支后，完整 logit 为：

$$
s_{t,j,h}
=\frac{\bar q_{t,h}c_j^\top+q_{t,h}^{R}(k_j^{R})^\top}
{\sqrt{d_k+d_r}}.
$$

每个历史 Token 缓存 $(c_j,k_j^R)$，所以：

$$
M_{\text{MLA}}=LBN(d_c+d_r)b.
$$

它是否比 MQA 或 GQA 更小，取决于 $d_c+d_r$ 与相应的 $G(d_k+d_v)$。若在展开后的 Key 上加入 RMSNorm，其分母依赖当前 Token 和 Head，就不能直接吸收到一个固定矩阵中，需要额外处理。

### 3.9 QK-Clip：通过权重缩放控制观测到的 logits

**原理：** QK-Norm 在 Forward 中加入数据依赖的归一化，可能妨碍 MLA 的简单投影吸收。QK-Clip 的思路是在参数更新过程中监测每层、每个 Head 的大 logits，并调整相关投影权重，而不是在 Attention 内部增加非线性操作。

其基本数学依据是点积对 Q/K 双线性：两侧同时缩放，logits 按缩放系数的平方变化。这个关系只对实际被同样缩放的完整点积成立；MLA 含独立 RoPE 分支时必须协调相关项。根据当前样本得到的控制，也不等于未来所有输入上的绝对保证。

**公式推导：** 对 $s=qk^\top/\sqrt{d_k}$，若 $q'=\gamma q,k'=\gamma k$，则：

$$
s'=\frac{(\gamma q)(\gamma k)^\top}{\sqrt{d_k}}
=\gamma^2s.
$$

假设当前观测到的最大 logit 为 $S_{\max}>\tau>0$，希望同一组点积缩放后最大值为 $\tau$，解得：

$$
\gamma^2S_{\max}=\tau
\quad\Rightarrow\quad
\gamma=\sqrt{\frac{\tau}{S_{\max}}}.
$$

当 $S_{\max}\le\tau$ 时不缩放。线性投影 $q=xW_Q$ 可通过 $W_Q\leftarrow\gamma W_Q$ 实现相同效果。对于 $s=s_C+s_R$，若只缩放 Content 分支，则：

$$
s'=\gamma^2s_C+s_R\ne\gamma^2(s_C+s_R).
$$

因此简单缩放结论不能直接套用到未同步处理的 RoPE 分支，也不宜随意改变多个 Heads 共享的权重。

### 3.10 MoE：稀疏 Expert Routing 与负载均衡

**原理：** MoE 用多个 Expert FFNs 替换一个普通 FFN，并用 Router 为每个 Token 选择少数 Experts。这样模型可以拥有很多参数，但一个 Token 只执行其中一小部分。Attention 等模块通常仍共享，因此总参数量和每 Token 的活跃参数量不同。

Router 若只偏向早期表现好的 Experts，就可能形成“得到更多 Token、训练更多、进一步变强”的循环，其他 Experts 得不到训练。Noisy Top-k 通过噪声增加探索，Importance Loss 则惩罚 Gate 总量过度集中。它们鼓励均衡，但不要求不同 Experts 学到完全相同的功能。

**公式推导：** 设共有 $E$ 个 Experts，$E_i(x)$ 为其输出，$g_i(x)$ 为 Gate：

$$
y=\sum_{i=1}^Eg_i(x)E_i(x).
$$

定义含噪声的 Router logits：

$$
z_i=(xW_g)_i+\epsilon_i\operatorname{Softplus}((xW_n)_i),
\quad\epsilon_i\sim\mathcal N(0,1),
\quad\operatorname{Softplus}(a)=\ln(1+e^a).
$$

选出最大 $k$ 项的集合 $S$，其余项设为 $-\infty$。Softmax 后得到：

$$
g_i=\begin{cases}
\dfrac{e^{z_i}}{\sum_{j\in S}e^{z_j}},&i\in S,\\
0,&i\notin S.
\end{cases}
$$

因此未选中的 Experts 不需要执行。若共享参数为 $P_s$，每个 Expert 有 $P_e$ 个参数，忽略额外小项：

$$
P_{\text{total}}\approx P_s+EP_e,
\qquad P_{\text{active}}\approx P_s+kP_e.
$$

对 $T$ 个 Token，Expert 的 Importance 为 $I_i=\sum_tg_i(x_t)$。由于每个 Token 的 Gates 求和为 1，有 $\sum_iI_i=T$，平均值为 $\bar I=T/E$。使用总体标准差时：

$$
\operatorname{CV}(I)^2
=\frac{\frac1E\sum_i(I_i-\bar I)^2}{\bar I^2}
=\frac{E\sum_iI_i^2}{T^2}-1.
$$

由 $\sum_iI_i^2\ge T^2/E$，此项非负，且各 $I_i=T/E$ 时为零；因此可加入 $L_{\text{importance}}=w\operatorname{CV}(I)^2$ 鼓励 Gate 总量均衡。

### 3.11 Switch Transformer：Top-1 Routing 与辅助损失

**原理：** Switch 每个 Token 只执行一个 Expert，从而降低 Expert 计算与通信需求。它先对全部 Experts 做 Softmax，再选择概率最大的 Expert，并保留这个原始概率作为输出系数。保留概率很重要：虽然选择索引的 argmax 不可微，但概率仍能把主任务的梯度传回 Router。

Switch 还使用实际路由比例与平均预测概率构造辅助损失。实际分配较多的 Expert 会受到更大的概率惩罚，鼓励其他 Experts 接收 Token。硬路由计数本身不通过 argmax 反向传播，梯度主要经过连续的 Softmax 概率。

**公式推导：** 对 Router logits $s=xW_r$：

$$
p_i=\frac{e^{s_i}}{\sum_j e^{s_j}},\quad
i^*=\arg\max_i p_i,\quad
y=p_{i^*}E_{i^*}(x).
$$

固定当前选择时，$\partial p_{i^*}/\partial s_j=p_{i^*}(\delta_{i^*j}-p_j)$。如果先只留下一个 logit 再做 Softmax，Gate 会恒等于 1，其导数为零，这与 Switch 不同。

对 $T$ 个 Token、$E$ 个 Experts，定义硬路由比例 $f_i$ 和平均概率 $P_i$：

$$
f_i=\frac1T\sum_t\mathbf1\{\arg\max_jp_j(x_t)=i\},
\qquad P_i=\frac1T\sum_t p_i(x_t).
$$

辅助损失定义为：

$$
L_{\text{aux}}=\alpha E\sum_i f_iP_i.
$$

把当前 $f_i$ 视为固定值，设第 $t$ 个 Token 的 logits 为 $s_{t,j}$，则由 Chain Rule：

$$
\frac{\partial L_{\text{aux}}}{\partial s_{t,j}}
=\frac{\alpha E}{T}\sum_i f_ip_{t,i}(\delta_{ij}-p_{t,j})
=\frac{\alpha E}{T}p_{t,j}
\left(f_j-\sum_i f_ip_{t,i}\right).
$$

当 $f_i=P_i=1/E$ 时损失为 $\alpha$；当二者全部集中到同一个 Expert 时为 $\alpha E$。这说明它对典型集中状态的惩罚更大，但不能据此认定均匀状态是任意可行 $f,P$ 下的严格全局最小值。

### 3.12 Mixtral：Top-2 与共享 Attention

**原理：** Mixtral 在每层使用八个 SwiGLU Experts，每个 Token 选择两个；Attention 部分共享，并使用 GQA 降低 KV Cache。Top-2 输出综合两个 Experts 的结果，计算开销通常高于同尺寸的 Top-1，但每 Token 仍无需执行所有 Experts。

Mixtral 对选中的两个概率重新归一化，使它们之和为一。这与 Switch 保留全体 Softmax 中的单个原概率不同。模型名中的八个 Experts 不表示八份完整模型，因为 Attention、Embedding 等并未全部复制。Fine-grained MoE 则可使用更多、较小的 Experts；比较开销时需要同时考虑激活数量和单个 Expert 大小。

**公式推导：** 设全体 Router logits 为 $s_i$，完整 Softmax 概率为 $p_i=e^{s_i}/Z$，其中 $Z=\sum_j e^{s_j}$。选择最大的两项组成 $S$，重新归一化：

$$
g_i=\frac{p_i}{\sum_{j\in S}p_j}
=\frac{e^{s_i}/Z}{\sum_{j\in S}e^{s_j}/Z}
=\frac{e^{s_i}}{\sum_{j\in S}e^{s_j}},\qquad i\in S.
$$

其余 Gate 为零，因此：

$$
y=g_aE_a(x)+g_bE_b(x),\qquad g_a+g_b=1.
$$

若原概率中最大的两项为 $0.6,0.3$，对应权重为 $2/3,1/3$。设整个模型的共享参数为 $P_s$，每个 Expert 分支在所有 MoE Layers 合计为 $P_e$，则：

$$
P_{\text{total}}\approx P_s+8P_e,
\qquad P_{\text{active}}\approx P_s+2P_e.
$$

固定输入维度和 FFN 形式时，Expert 的主要计算与 Hidden Width 成正比；若激活 $k$ 个宽度为 $m_e$ 的小 Experts，其主要计算量与 $km_e$ 成正比，而不只由 $k$ 决定。

### 3.13 LoRA：用低秩增量完成 Fine-tuning

**原理：** LoRA 冻结预训练权重，只学习一个由两个小矩阵组成的更新分支。其假设是某个任务需要的权重变化可以由较低 Rank 近似表达；这是一种有用的建模约束，不是所有任务的最优更新都必然低秩。

低秩分支与原线性层并行，相加得到输出。它减少可训练参数及其梯度、Optimizer States，但原模型仍需存储与计算。训练后可把增量合并回原权重，消除额外分支；如果为了切换 Adapters 而不合并，则仍存在支路开销。

**公式推导：** 沿用行向量约定，原线性层为 $y=xW_0$，$W_0\in\mathbb R^{d_{\text{in}}\times d_{\text{out}}}$。令：

$$
A\in\mathbb R^{d_{\text{in}}\times r},\qquad
B\in\mathbb R^{r\times d_{\text{out}}},\qquad
\Delta W=\frac\alpha r AB.
$$

因为 $AB$ 的列空间受 $A$ 的至多 $r$ 列限制，$\operatorname{rank}(AB)\le r$。前向计算为：

$$
y=xW_0+\frac\alpha r xAB.
$$

其中 $r$ 控制更新容量，$\alpha/r$ 是设定的缩放系数。两个矩阵的可训练参数量为：

$$
P_{\text{LoRA}}=d_{\text{in}}r+rd_{\text{out}}
=r(d_{\text{in}}+d_{\text{out}}),
\qquad P_{\text{full}}=d_{\text{in}}d_{\text{out}}.
$$

当输入输出维度均为 $d$，比例为 $2r/d$。例如 $d=4096,r=8$，LoRA 参数仅为全矩阵的 $1/256$；这不等于整个训练显存也按此比例下降。

通常令 $A$ 随机、$B=0$，使初始 $\Delta W=0$。设 $G=\partial L/\partial(AB)$，其中包含后续计算与缩放带来的梯度，则：

$$
\frac{\partial L}{\partial A}=GB^\top,
\qquad\frac{\partial L}{\partial B}=A^\top G.
$$

初始 $A$ 的梯度为零，但随机 $A$ 通常使 $B$ 获得非零梯度；若两者都初始化为零，两条梯度都会为零。训练后根据分配律合并：

$$
W_{\text{merged}}=W_0+\frac\alpha r AB,
\qquad y=xW_{\text{merged}}.
$$

合并时必须保留 Scaling。LoRA 可应用于 Q/K/V/O Projections 和 FFN Linear Layers；增加 Target Modules 与增加 Rank 都能扩大适配容量，但作用方式不同。

## 四、LLM

### 4.1 Tokenization 与 BPE

**原理：** 文本先拆成 Token，再映射为 ID，通过 Embedding Table 转为向量。Token 可以是词、Subword、字符或 Byte。整词词表容易遇到 Out-of-Vocabulary；纯字符序列又过长。Subword 在词表规模与序列长度之间折中。

BPE 从基本符号开始，每轮合并语料中最高频的相邻 Pair，再重新统计，直到达到目标词表大小。推理时按学到的 Merge Ranking 执行合并，不重新统计输入频率。Byte-level BPE 以 Byte 为基本单位覆盖文本。

**公式推导：** 设语料中的不同词为 $w$，词频为 $c(w)$，当前拆分中相邻 Pair $(a,b)$ 在词 $w$ 内出现 $n_w(a,b)$ 次，则它的语料频数为

$$
f(a,b)=\sum_w c(w)n_w(a,b).
$$

定义本轮合并对象为

$$
(a^*,b^*)=\arg\max_{(a,b)}f(a,b).
$$

必须乘词频：一个词出现十次，其内部 Pair 也贡献十次。合并改变相邻关系，因此下一轮需要重新统计。该准则是 BPE 的定义。

### 4.2 WordPiece 的训练 Score 与推理规则

**原理：** WordPiece 建词表时根据两个片段的共现频数及各自频数评分。非词首片段通常带有 `##`，表示接在前面的片段后面。

推理时采用 Greedy Longest-match-first，从左到右选择最长匹配片段，不回溯；若剩余部分无法匹配，则整个词映射为 `[UNK]`。训练 Score 与推理规则必须区分。

**公式推导：** 课件使用如下合并 Score 定义：

$$
\operatorname{score}(a,b)=\frac{f(a,b)}{f(a)f(b)}.
$$

$f(a,b)$ 是相邻 Pair 的加权语料频数，$f(a),f(b)$ 是两个片段各自的加权频数。固定分子时，分母越大，Score 越低；固定分母时，共现越频繁，Score 越高。该准则表达的是归一化共现强度，不应把它直接称为下一 Token 的条件概率。

### 4.3 SentencePiece 与 Unigram 分词模型

**原理：** SentencePiece 是支持 BPE 和 Unigram LM 的分词框架。它直接处理字符流，不依赖按空格切词；空白标记保留边界信息。默认 NFKC Normalization 会改变部分字符，因此恢复的通常是规范化文本。

Unigram 从大词表开始，估计片段概率，再优化和删减。同一字符串可有多条分词路径；每条路径的概率是片段概率的乘积。该独立假设仅用于 Tokenizer，不用于后面的 Transformer。

**公式推导：** 令 $X=c_1\cdots c_L$ 是包含 $L$ 个字符的字符串，$V$ 是词表。一种分词为 $s=(u_1,\ldots,u_m)$，其中每个 $u_i\in V$，且拼接后等于 $X$。定义

$$
P(s)=\prod_{i=1}^{m}p(u_i),\qquad \sum_{u\in V}p(u)=1.
$$

令 $\mathcal S(X)$ 表示所有合法路径。因为真实分词没有被观测到，需要对它求和：

$$
P(X)=\sum_{s\in\mathcal S(X)}P(s).
$$

对训练集 $\mathcal D$，最大化字符串的 Log-likelihood：

$$
\mathcal L=\sum_{X\in\mathcal D}\log P(X).
$$

这里 $m$ 是某条路径的 Token 数，可以随路径变化；$L$ 始终是原字符串的字符数。同一个片段在路径中重复出现时，其概率也要重复相乘。

### 4.4 Unigram 的 Forward–Backward 与 EM

**原理：** 分词路径是未观测的隐藏变量。Expectation-Maximization 的 E-step 计算各 Token 跨所有路径的 Expected Count，M-step 据此更新概率。只保留最可能路径不能代替这里的 E-step。

Lattice 将字符间位置设为节点，词表中的子串对应一条边。Forward–Backward 复用前缀和后缀概率，避免枚举所有完整路径。

**公式推导：** 记 $u_{ij}=c_{i+1}\cdots c_j$。Forward $\alpha(j)$ 是覆盖前 $j$ 个字符的总路径概率，Backward $\beta(i)$ 是覆盖位置 $i$ 之后字符的总路径概率：

$$
\begin{aligned}
\alpha(0)&=1, &
\alpha(j)&=\sum_{i<j:\,u_{ij}\in V}\alpha(i)p(u_{ij}),\\
\beta(L)&=1, &
\beta(i)&=\sum_{j>i:\,u_{ij}\in V}p(u_{ij})\beta(j).
\end{aligned}
$$

所以 $P(X)=\alpha(L)=\beta(0)$。经过边 $i\to j$ 的完整路径，必然由“前缀路径、当前 Token、后缀路径”组成，因此该边的 Posterior 为

$$
P(i\to j\mid X)=\frac{\alpha(i)p(u_{ij})\beta(j)}{\alpha(L)}.
$$

把所有字符串中标记为 $u$ 的边贡献加起来，得到期望计数：

$$
c(u)=\sum_{X\in\mathcal D}\sum_{i<j:\,u_{ij}=u}
\frac{\alpha(i)p(u)\beta(j)}{\alpha(L)}.
$$

M-step 最大化 $\sum_u c(u)\log p(u)$，同时满足 $\sum_u p(u)=1$。引入 Lagrange multiplier $\lambda$：

$$
\frac{\partial}{\partial p(u)}
\left[\sum_v c(v)\log p(v)+\lambda\left(1-\sum_v p(v)\right)\right]
=\frac{c(u)}{p(u)}-\lambda=0.
$$

由此 $p(u)=c(u)/\lambda$；再利用归一化条件，得到

$$
\lambda=\sum_v c(v),\qquad p(u)=\frac{c(u)}{\sum_v c(v)}.
$$

### 4.5 Unigram 的 Pruning、Viterbi 与 Sampling

**原理：** Pruning 删除对语料 Log-likelihood 贡献较小的 Token，再运行 EM；保留必要的基本字符和回退符号。

推理时，Viterbi 找概率最高的完整路径；Subword Regularization 则按概率采样，让同一文本呈现不同切分，形成 Data Augmentation。

**公式推导：** 若 $\mathcal L(V)$ 表示当前词表的语料 Log-likelihood，定义删除 Token $u$ 的损失为

$$
\Delta\mathcal L_u=\mathcal L(V)-\mathcal L(V\setminus\{u\}).
$$

删除造成的损失越小，保留它的必要性越低。Viterbi 的最优路径满足

$$
s^*=\arg\max_s\prod_{u\in s}p(u)
=\arg\max_s\sum_{u\in s}\log p(u).
$$

第二个等号来自 Log 的单调性，以及乘积取 Log 后变为求和。令 $v(j)$ 为覆盖前 $j$ 个字符的最大 Log-probability，则

$$
v(0)=0,\qquad
v(j)=\max_{i<j:\,u_{ij}\in V}\left[v(i)+\log p(u_{ij})\right].
$$

保存每一步的最优前驱，再回溯即可得到路径。Sampling 可定义为

$$
P_a(s\mid X)=\frac{P(s)^a}{\sum_{s'\in\mathcal S(X)}P(s')^a},\qquad a>0.
$$

两条路径的概率比为 $[P(s_1)/P(s_2)]^a$，所以 $a$ 越大，分布越偏向高概率路径；$a$ 越小，采样越分散。

### 4.6 BERT：输入表示、MLM 与 NSP

**原理：** BERT 是 Bidirectional Encoder-only 模型。Token、Segment、learned Position Embedding 相加后进入 Transformer Encoder；各位置可读取左右上下文。`[CLS]` 用于整体分类，`[SEP]` 分隔句子，Segment 标记 A/B。

MLM 随机选中约 15% 的普通 Token 作为预测目标；在这些位置内部，80% 替换为 `[MASK]`，10% 替换为随机 Token，10% 保持原样。所有选中位置都预测原始 Token。NSP 则判断句子 B 是否为句子 A 的真实下一句，正负样本各占约一半。BERT 联合学习 Token 恢复与句间关系，之后可为不同任务添加输出头并 Fine-tune。

**公式推导：** 三种输入向量均为 $d$ 维，逐元素相加后维度不变：

$$
x_i=E_{\mathrm{token},i}+E_{\mathrm{segment},i}+E_{\mathrm{position},i}.
$$

令 $M$ 为被选中的位置集合，$X$ 为原文，$\widetilde X$ 为替换后的输入。最大化原 Token 的预测概率，等价于最小化

$$
\mathcal L_{\mathrm{MLM}}=-\sum_{i\in M}\log P_\theta(x_i\mid\widetilde X).
$$

损失只在 $M$ 中计算，包括随机替换和保持原样的位置。按全体普通 Token 计，三类位置的期望比例为

$$
0.15\times0.8=12\%,\qquad
0.15\times0.1=1.5\%,\qquad
0.15\times0.1=1.5\%.
$$

令 $y\in\{0,1\}$ 表示 NSP 标签，$q_\theta$ 表示根据 `[CLS]` 输出预测 B 为下一句的概率，则 Binary Cross-entropy 为

$$
\mathcal L_{\mathrm{NSP}}=-y\log q_\theta-(1-y)\log(1-q_\theta).
$$

当 $y=1$ 时只保留 $-\log q_\theta$；当 $y=0$ 时只保留 $-\log(1-q_\theta)$。联合目标为 $\mathcal L_{\mathrm{MLM}}+\mathcal L_{\mathrm{NSP}}$，各项可在样本或预测位置上取平均。

### 4.7 T5：Text-to-Text 与 Span Corruption

**原理：** T5 使用 Bidirectional Encoder、Causal Decoder 和 Cross-attention，将翻译、摘要、分类统一为 Text-to-Text。任务前缀说明任务；Decoder 根据此前目标 Token 与 Encoder 输出生成答案，分类标签也生成为文字。

Span Corruption 删除若干连续 Token 片段，每个片段用不同的 Sentinel Token 替换。Encoder 读取带缺口的输入；Decoder 输出被删除的片段及其 Sentinel 边界，而不是完整复制原文。与 BERT 在各被选位置预测 Token 不同，T5 把恢复内容组织成一个目标序列，因此可以处理不同长度的缺口。

**公式推导：** 令 $\widetilde X=\operatorname{Corrupt}(X)$，$Y=(y_1,\ldots,y_T)$ 表示按顺序拼接的 Sentinel 与被删片段。由条件概率的 Chain Rule，

$$
P_\theta(Y\mid\widetilde X)
=\prod_{t=1}^{T}P_\theta(y_t\mid y_{<t},\widetilde X).
$$

取负 Log，将乘积转成逐 Token 的损失和：

$$
\mathcal L_{\mathrm{T5}}
=-\sum_{t=1}^{T}\log P_\theta(y_t\mid y_{<t},\widetilde X).
$$

$y_{<t}$ 是此前的目标 Token；训练时由右移后的真实目标提供，生成时由已生成 Token 提供。Sentinel 也属于要预测的目标 Token。对翻译或摘要等任务，保持同样的条件生成目标，只需将 $\widetilde X,Y$ 换成相应任务的输入与目标。

### 4.8 GPT：Autoregressive 目标与 GPT-1 Fine-tuning

**原理：** GPT 是 Decoder-only Transformer，以 Causal Self-attention 读取历史，通常没有 Cross-attention。Hidden State 经词表投影和 Softmax 得到下一 Token 分布。训练用右移输入并行计算各位置；生成时逐步追加输出。

GPT-1 先用未标注文本做 Language Modeling，再在有标签的任务上 Fine-tune Transformer 与新增任务头。分类任务可读取最后位置的表示；同时保留辅助 Language Modeling 目标，有助于避免只适配分类而丢失原有语言建模能力。

**公式推导：** 对序列 $x_{1:T}$，Chain Rule 给出

$$
P_\theta(x_{1:T})=\prod_{t=1}^{T}P_\theta(x_t\mid x_{<t}),
\qquad
\mathcal L_{\mathrm{AR}}=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t}).
$$

第一项可通过起始 Token 建模；实际可读取的历史受 Context Window 限制。

对分类任务，令 $h\in\mathbb R^d$ 为最后位置的 row vector，$W_y\in\mathbb R^{d\times C}$ 为 $C$ 类的输出投影，则

$$
p_\theta(y\mid X)=\operatorname{softmax}(hW_y)_y,
\qquad
\mathcal L_{\mathrm{task}}=-\log p_\theta(y^*\mid X).
$$

$y^*$ 是正确标签。联合最小化 $\mathcal L_{\mathrm{task}}+\lambda\mathcal L_{\mathrm{AR}}$，其中 $\lambda$ 控制辅助目标的权重。

### 4.9 Prompting 与 In-context Learning

**原理：** Prompt 将任务说明和示例放入上下文。Zero-shot 不给示例，One-shot 给一个，Few-shot 给少量。ICL 的示例占用 Context Window，但不触发 Gradient Update；Fine-tuning 则通过训练改变参数。

先进行 Instruction Tuning 的模型仍可使用 ICL；训练方式与调用方式是两个维度。

**公式推导：** 令 $C$ 为任务说明及示例构成的上下文，$x$ 为当前问题，$y$ 为回答。Prompting 使用

$$
P_\theta(y\mid C,x),\qquad \Delta\theta=0.
$$

改变 $C$ 可以改变条件概率分布，而不改变参数 $\theta$。对于多 Token 回答，仍然按 Autoregressive 方式分解：

$$
P_\theta(y\mid C,x)=\prod_tP_\theta(y_t\mid C,x,y_{<t}).
$$

### 4.10 SFT、Reward Model 与 PPO

**原理：** SFT 使用 Prompt 与示范 Response 训练生成模型；Reward Model 根据人工对候选回答的偏好学习评分；PPO 用评分信号更新从 SFT 得到的生成 Policy。它们分别学习示范、拟合偏好、优化策略。

PPO 阶段通常固定 Reward Model、更新 Policy，并控制更新幅度。人类不必实时评价每次生成，奖励由学到的评分模型提供。

**公式推导：** 对 Prompt $x$ 与示范 $y^*$，SFT 最大化条件生成概率，得到

$$
\mathcal L_{\mathrm{SFT}}=-\sum_t\log P_\theta(y_t^*\mid x,y_{<t}^*).
$$

这里采用常见的 Response-only Loss，$t$ 只索引回答中的 Token。令 Reward Model 为 $r_\phi(x,y)\in\mathbb R$，当人工偏好 $y^+$ 胜过 $y^-$ 时，学习方向是让

$$
r_\phi(x,y^+)-r_\phi(x,y^-)>0.
$$

$\phi$ 是评分模型参数，$\theta$ 是生成模型参数。令 $\pi_\theta(y\mid x)$ 为 Policy 生成完整回答的概率，则奖励驱动的优化方向可以写为

$$
J(\theta)=\mathbb E_{x}\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)].
$$

这个式子表达“提高策略所生成回答的期望评分”，并不是 PPO 的完整 Loss；PPO 另有控制更新幅度的机制。SFT 的监督是示范 Token，Reward Model 的监督是偏好，Policy 优化的信号是回答评分，三者不能使用同一个损失解释。

### 4.11 FFN 的 Key–Value Memory 解释与参数比例

**原理：** Attention 汇总上下文，FFN 变换当前 Token 特征。FFN 的第一层检测与可学习模式的匹配程度，Activation 产生响应，第二层组合输出方向，因此可解释为 Key–Value Memory。

这种解释不意味着知识只存在于 FFN。这里的 Memory 是参数中的模式，不是推理时临时保存的 KV Cache。

**公式推导：** 对 row vector $x\in\mathbb R^d$，忽略 Bias，标准 FFN 为

$$
\operatorname{FFN}(x)=\sigma(xW_1)W_2,
\qquad W_1\in\mathbb R^{d\times f},\quad W_2\in\mathbb R^{f\times d}.
$$

把 $W_1$ 的第 $j$ 列记为 $k_j$，把 $W_2$ 的第 $j$ 行记为 $v_j$，则

$$
\operatorname{FFN}(x)=\sum_{j=1}^{f}\sigma(xk_j)v_j.
$$

因此 $xk_j$ 可以理解为匹配分数，$\sigma(xk_j)$ 为激活强度，$v_j$ 为被组合的输出方向。这些系数不要求总和为 1；是否非负取决于使用的 Activation。

若一个 Block 只有一个标准 MHA，且 $d_k=d_v=d/h$，Q/K/V/O 投影共 $4d^2$ 个 Weight 参数。FFN 的 Weight 参数为 $df+fd=2df$。令 $f=4d$，得到

$$
N_{\mathrm{FFN}}=8d^2,\qquad
\frac{N_{\mathrm{FFN}}}{N_{\mathrm{MHA}}+N_{\mathrm{FFN}}}=\frac{8}{4+8}=\frac23.
$$

该比例忽略 Bias、Normalization、Embedding，并依赖“一个 MHA、标准 FFN、$f=4d$”这些条件，不能直接推广到所有 Transformer 架构。

### 4.12 Scaling：参数、训练数据与计算预算

**原理：** 增大模型和增加训练 Token 通常能降低 Loss，但二者竞争固定的训练预算。Compute-optimal Scaling 研究如何分配模型规模与数据量，不能只追求参数最多。

对 Dense Transformer，参数相关矩阵运算占主导时，成本近似正比于参数量乘训练 Token 数。最佳分配仍需经验拟合；长序列 Attention、实现和数据质量都会影响结果。

**公式推导：** 令 $N$ 为参数量，$D$ 为训练中处理的 Token 总数，$\kappa$ 为每个参数每个 Token 对应的近似计算系数，则

$$
C\approx\kappa ND.
$$

这里 $D$ 统计实际训练处理量，重复使用的数据也会计入。若参数量扩大 $a$ 倍，Token 数扩大 $b$ 倍，且系数近似不变，则

$$
\frac{C'}C\approx\frac{\kappa(aN)(bD)}{\kappa ND}=ab.
$$

固定预算时，$D\approx C/(\kappa N)$；参数量增大，会压缩可处理的 Token 数。如果额外假设参数与 Token 数等比例增长，即 $a=b$，预算增加 $r$ 倍时应有 $a=b\approx\sqrt r$。这是给定分配假设下的计算关系，不是“等比例增长永远最优”的证明。

### 4.13 CoT、Self-consistency 与问题分解

**原理：** Zero-shot CoT 直接要求中间步骤；Few-shot CoT 提供含推理过程的示例。Self-consistency 采样多条路径，再按最终答案聚合。

Divide and Conquer 或 Least-to-Most 拆出子问题，用已有子答案解决后续问题。它们改变上下文组织；Code-based Pretraining 改变训练数据。多条路径仍可能共享同一错误，不保证正确。

**公式推导：** 令 $x$ 为问题，$r$ 为中间推理路径，$a$ 为最终答案。由全概率公式，将未指定的路径求和：

$$
P_\theta(a\mid x)=\sum_rP_\theta(a,r\mid x)
=\sum_rP_\theta(r\mid x)P_\theta(a\mid x,r).
$$

Self-consistency 希望选择总概率最高的答案，而不是概率最高的某一条路径：

$$
\hat a=\arg\max_a\sum_rP_\theta(a,r\mid x).
$$

实际采样 $K$ 条路径得到答案 $a^{(1)},\ldots,a^{(K)}$。在等权采样估计下，

$$
\widehat P(a\mid x)=\frac1K\sum_{k=1}^{K}\mathbf 1[a^{(k)}=a],
\qquad \hat a=\arg\max_a\widehat P(a\mid x).
$$

$\mathbf 1[\cdot]$ 在条件成立时为 1，否则为 0，因此上式就是按答案多数投票。对问题分解，若第 $j$ 个子问题为 $q_j$、答案为 $b_j$，逐步求解使用

$$
P_\theta(b_j\mid x,q_j,q_{<j},b_{<j}),
$$

再根据所有子问题与子答案生成最终回答。这个条件概率写法说明，前面的结果怎样成为后续输入，而不是声称分解本身能证明最终答案正确。
