# AIMS5710 两页速查表

成品：[cheetsheet.pdf](cheetsheet.pdf)。A4 竖版，两页双栏，无页眉页脚；左右页边距 2 mm，上下页边距 8 mm；第 1 页正文 7.7 pt，第 2 页正文 7.2 pt。按 **实际大小 / 100% / 长边翻转** 打印即可使用一张纸正反面。

- 第 1 页：NN、Optimization，各占一栏。
- 第 2 页：CNN、RNN，各占一栏。
- CNN 反向传播的 1D 和 2D 公式各分两行：第一行给出输出和误差，第二行给出三个梯度。
- 池化部分使用三列表格，保留三个运算公式并列出各自优势；平均池化的分母直接写为窗口元素数。
- Modern Convolutions 使用模型、核心方法和优势三列表格；EfficientNet 的方法概括为 Compound Scaling。
- Partial / Submanifold Sparse Convolution 使用不含偏置的 Conv 简写，偏置单独相加；分段公式给出无效位置的输出，按要求省略 mask 更新式。
- 内容来源为上一级的 [cheetsheet.md](../cheetsheet.md)，包含 4 个模块、34 个小节；按用户要求精简说明、统一 FC 梯度下标，并调整 MLP、BN、Dropout 分行，具体范围见 [REVIEW.md](REVIEW.md)。
- 激活函数使用原来的五列表格，4 行、20 个数据单元格全部保留。
- BN 的 Standardization 与 Affine Transform 排在同一行；按要求删除训练 / 推理统计量说明。全文说明性箭头两侧统一留空格。
- 优化部分按要求简化标签：批量大小和学习率使用 B_t、η_t；Rprop 的参数更新公式排在步长分段公式下方，同属一个对齐的公式块。
- Adam 和 AdamW 各占一行，公式后分别用箭头接对应说明。
- Xavier / Glorot 的完整公式与 sigmoid、tanh 排在同一行；Warmup 独立成行，Kaiming / He 统一使用带空格的斜杠。
- 两页均使用 9 pt 正文行距、3 pt 段间距，小节前后间距为 5 pt / 2 pt；各栏按内容自然排布，不强制底部对齐。公式和表格按自身高度排布；RNN 门名称加粗，箭头说明使用普通段落自动换行。

## 重新生成

需要 Python 3、XeLaTeX（含 TeX Gyre、newtxmath、multicol 等常规 TeX Live 包）和 Poppler 的 `pdftoppm`。Python 依赖见 `requirements.txt`。

```sh
cd AIMS5710/cheetsheet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py
```

构建脚本直接读取 Markdown，自动生成 `cheetsheet.tex`，编译两次并渲染 PNG；不需要手动同步 TeX 内容。`--font-size` 控制第 1 页正文，默认 7.7 pt；`--second-page-font-size` 控制第 2 页正文，默认 7.2 pt。

## 验收

`output/validation.json` 记录源文件 SHA-256、每个源块的行号、公式数、页数及排版检查结果。脚本检查：

1. 所有非空源行恰好处理一次。
2. 全部 123 个公式片段按顺序逐一核对，源文与生成 TeX 完全一致（只忽略空白）。
3. PDF 恰好两页，4 个模块与 34 个小节标题全部可提取。
4. 提取 PDF 后，原文所有英文词的出现次数均不少于源文，防止文字遗漏。
5. 无 LaTeX overfull 警告、缺字警告或越出安全页边界的文字。

同时逐页查看 `output/page-1.png`、`output/page-2.png`，检查公式、上下标、分段函数和栏间隔，确认没有页眉页脚。自动检查不替代人工检查公式含义。通过后自动清理临时编译文件；失败时保留 `tmp/` 供排查。
