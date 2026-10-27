# 内容核查与修订

核查范围：原 `cheetsheet.md` 全部内容，对照 `../Material/` 下的四份课程讲义及下列一手资料。修订已同步到 Markdown 和 PDF；没有为满足页数删除知识点。

## 前一版完整性复核

去掉页眉页脚后，再次对照 Git 中的最初 Markdown、当前 Markdown、生成的 TeX 和两页 PDF：

- 当时原始与当前文档的 4 个模块、33 个小节标题及顺序完全一致：NN 9、Optimization 10、CNN 7、RNN 7。
- 激活函数表的 20 个数据单元格全部保留。
- 原始 151 行非空内容中，133 行原样保留；其余 18 行逐项核对，均为下文记录的纠错或等价重写，没有删除知识点。
- 原始 116 个公式片段在补充条件、拆分推导后成为 140 个；当前源文的全部 140 个公式片段与生成 TeX 按顺序一致，仅有排版空白与换行指令差别。
- PDF 文字提取未发现源文英文词遗漏，全部模块与小节标题均存在；两页渲染图已检查，无页眉页脚、裁切、溢出或缺字。

## 最新用户指定调整

在上述完整版本上，按用户要求删除以下说明：

- Softmax 后的 “The equality ... assumes one-hot targets ...” 一句。
- 激活函数表后的 “At z=0, ReLU / Leaky ReLU ...” 一句。
- MLP 描述中的括号 “(possibly Higher-Dimensional)”。
- Regression Losses 后的 “For L1 at ... choose a subgradient ...” 一句。
- MLP 公式后的 “→ Learned Nonlinear Features → May Enable Linear Separation”。
- AutoEncoder 描述末尾的 “→ Use Encoder for other tasks”。
- “L2 and weight decay are equivalent for plain SGD; AdamW decouples decay from adaptive gradients.” 一句。

MLP 的四项说明保持原顺序，每行两项：第一行 Activation → Introduce Nonlinearity；Linear Function → Classification Logits，第二行 Multi-Layer → Improve Representation Efficiency；Perceptron → Final Linear Classification。将用户的 Perception 拼写修正为 Perceptron。

FC 的上游梯度由向量写法 `g=∂L/∂z` 改为对应坐标写法 `g_i=∂L/∂z_i`，与后续权重、输入梯度下标一致。

Batch Normalization 的 Standardization 与 Affine Transform 合并在同一排版行；训练 / 推理统计量的说明按用户要求删除。

Dropout 在 Keep Probability 前换行；第二行包含保留概率 q 及 Random Mask 的 Bernoulli 分布，训练 / 推理公式不变。

全文说明性箭头 `→` 两侧统一加入空格；数学公式内的 `\to` 保留 TeX 的关系符间距，不改变公式。

优化部分进一步精简标签与排版：

- First-Order Taylor Expansion 改为 Taylor Expansion，并与公式同行；批量大小比较改用 B_t，学习率比较改用 η_t，Too Large 改为 Larger。
- Rprop 删除 “(simplified course rule)”，标签与分段公式同行，算法公式不变。
- Correct Zero Initialization Bias 删去末尾 Bias，保留 Bias Correction 标签。
- Vanishing Gradients / Exploding Gradients 简化为 Vanishing / Exploding Gradients。
- 学习率计划的 Inverse-Time、Step、Exponential 标签删去 Decay，公式不变。
- Smaller B_t 的说明删去 May；Gradient Clipping 删去末尾 “→ More Stable Training”，保留 Bounded Gradient Norm。
- Kaiming／He 改为 Kaiming / He；Xavier / Glorot 的完整公式及 sigmoid、tanh 排在同一行，数学内容不变。
- Warmup 从 Cosine Annealing 后移至独立一行，两者公式均保留。

上述删除均为用户明确指定，并非排版遗漏。当前 Markdown 为 167 行非空内容、123 个公式片段；池化公式和 Modern Convolutions 均为三列表格。文档包含 4 个模块、34 个小节；激活函数表的 20 个数据单元格均保留。最新构建结果以 `output/validation.json` 为准。

## 第二页可读性调整

排版调整：第 2 页正文由 7.7 pt 调为 7.2 pt；两页统一使用 9 pt 正文行距、3 pt 段间距和 5 pt / 2 pt 小节前后间距，公式和表格按自身高度排布。RNN 门名称加粗。按用户要求，箭头条目使用普通段落自动换行。

源文件已同步当前 PDF 的修改，并将 Rprop 的参数更新公式排在步长分段公式下方的同一公式块内；最新源文件校验值见 `output/validation.json`。完整性按 167 行非空内容、123 个公式片段、34 个小节核对。

## 实质修正

| 位置 | 原问题 | 修订 |
| --- | --- | --- |
| GRU | 状态更新把 z 当作写入门，直接通路导数却使用 diag(z) | 按课程 RNN 第 31–32 页约定，改为 z 保留旧状态：h = z·h_prev + (1−z)·candidate，并调整门的说明 |
| Mini-batch SGD | 损失函数自变量与右侧不一致；梯度被标为 objective；更新未使用已定义的 mini-batch 梯度 | 统一 θ、g_t 和批量大小的定义及更新式 |
| Rprop | 遗漏相邻梯度乘积为零的分支 | 补“保持步长不变”，沿用课程的简化算法 |
| BPTT | 中间链式推导未统一向量 / Jacobian 方向 | 统一列梯度与转置，完整保留三步推导，并补终止条件 δ_(T+1)=0 |
| Partial convolution | 只定义了有效输入数量大于零的情况 | 补零有效输入时输出为零；mask 更新公式后按用户要求删除 |

## 补充条件和表述校正

- 前一版补充了 Softmax 的 one-hot 条件及 L1 / ReLU 在零点的次梯度约定；最新版本按用户要求删除这些说明。
- MLP 显式分离线性层和激活层；保留非线性、分类 logits、表示效率和最终线性分类四项功能说明。
- NN BatchNorm 补可学习仿射变换；曾补训练批统计量与推理运行统计量的说明，后按用户要求删除。
- 普通 Dropout 与 CNN inverted dropout 分别标明约定并统一 q；曾补 channel mask、drop probability 和推理恒等映射说明，后按用户要求删除。
- 小 batch 的泛化说明曾加入 May，后按用户要求删除；大 batch 的更新次数限定为每 epoch。曾补充 L2 与 weight decay 仅在 plain SGD 下等价的说明，后按用户要求删除。
- 池化尺寸箭头及其适用条件按用户要求删除；平均池化分母直接写为窗口元素数。
- 卷积等变性补 stride 和边界条件；分类平移不变性作为目标，不能视为任意 CNN 的保证。
- 计算量标为 MACs；CNN BatchNorm 曾补 M=NHW 的说明，后按用户要求删除。
- RNN 参数量注明单层、单向、每门一个合并 bias、包含输出层的计数约定。
- 修正 Reconstruction、Glorot、Latent Representation 等拼写；去除逐坐标标量更新中的冗余逐元素乘号。

## 核查资料

- 本地课程讲义：`nn.pdf`（BN 第 38–39 页、Dropout 第 41 页）、`optimization.pdf`（Rprop 第 14 页）、`cnn.pdf`（Partial convolution 第 33–34 页）、`RNN.pdf`（GRU 第 31–32 页）。
- [PyTorch GRU 定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)：门与状态更新约定。
- [PyTorch Rprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.Rprop.html)：步长分支及实现变体；PDF 保留课件简化版本。
- [BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)、[Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)、[Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)：仿射变换、统计量和缩放约定。
- [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)、[MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)：卷积与池化输出尺寸。
- [Partial Convolution 原论文 §3.1](https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)：空窗口输出和 mask 更新。
- [AdamW 原论文](https://arxiv.org/abs/1711.05101)：L2 regularization 与 decoupled weight decay 的区别。
- [Deep Learning 第 9 章](https://www.deeplearningbook.org/contents/convnets.html)：卷积等变性与池化不变性。

未发现其余核心公式存在需要修正的实质错误；这是课程速查表，保留课件符号和简化约定，不将特定框架的参数化强行替换进来。
