# Agent 项目筛选结果（无机器人项目）

已排除机器人、无人机和偏硬件设计的项目。推荐指数按 Agent 相关性、范围清晰度及可评估性主观给出。

|表格里的序号|Project Title|项目描述|项目要求|推荐指数|
|---:|---|---|---|---:|
|6|LLM Agent for Ultrasound Decision Support|【Agent；可改子题】超声报告辅助智能体：维护病例信息、调用规则或模板工具，指出缺失证据；原文允许改做其他超声子任务。|完成提示词、短期记忆、病例卡及至少一种工具；演示结构化报告/证据核对并做定性评估。|4/5|
|12|Spatial Intelligence in Vision-Language Models for Building Reconstruction|【Agent相关】VLM 根据建筑图像和已有模型，选择建模工具并规划可执行的修复动作；与工具型 Agent 的空间推理相关。|实现视觉观察—动作序列—Blender 执行—独立评估的小型原型，比较空间描述和工具抽象层级。|4/5|
|14|Trace-Guided Recursive Self-Improvement of LLM Agent Harnesses|【Agent】根据运行轨迹自动改进 LLM Agent 的提示、检索、记忆和工具路由框架。|实现候选框架生成—验证—评测闭环；设置隔离测试集，比较质量、成本、延迟和鲁棒性。|5/5|
|52|Catching the Confident Wrong Answer: An Automatic Grounding Auditor|【Agent配套】自动核查文档助手答案的事实依据；属于 Agent 可靠性评估配套工具，原文明确说明不开发助手本体。|把回答拆成事实声明、检索证据并判定支持/矛盾/无依据；用已有标注测量风险评分。|3/5|
|53|Knowing When to Say No: Tool-Using Agents That Detect Impossible Requests|【Agent】研究工具型 Agent 何时应停止并拒绝无法完成的请求；用提供的旅行规划沙盒和基线 Agent 评测。|熟练使用 Python、JSON 和 REST API；设计决策循环与停止策略，优于基线并分析错误。|5/5|
|61|Developing Time Aware AI Agents for Longitudinal Electronic Health Record Question Answering|【Agent】面向纵向电子病历的时间感知问答 Agent，调用只读工具检索多次就诊记录并给出可追溯证据。|实现时间检索接口，处理信息缺失/时间歧义；对比固定流程与静态 RAG，评估准确率和工具成本。|4/5|
|64|Supporting Antimicrobial Stewardship with AI Agents by Modeling Patient Trajectories|【Agent】抗菌药物管理 Agent：整合患者病程、检验和用药记录，辅助指定时点的抗生素复核。|利用公开或获批 EHR 数据；输出来源、疑点及人工复核理由，对比自适应与固定检索。|4/5|
|70|From Video to Clinical Record: An Agentic Multimodal LLM for Automated Colonoscopy Reporting|【Agent】多模态 Agent 将肠镜图像/视频整理为结构化临床报告，并用 Agent 框架限制幻觉。|分阶段做病灶检测、图像报告、视频报告与重复病灶合并；要求视觉及多模态模型能力。|3/5|
|74|Improving the Reliability of LLM Agents on Long-Horizon Scientific Data Analysis Tasks|【Agent】研究长链科学数据分析 Agent 的记忆、过程验证、自我纠错和失败恢复。|构建可复现任务集与模块化 Agent；对记忆、验证、恢复策略做同成本消融评估。|5/5|
|84|A Self-Correcting LLM Agent for Logistics Optimization Modeling|【Agent】自纠错物流优化 Agent：从自然语言生成数学模型与求解代码，再利用求解器反馈修正。|两人团队分工模型生成与验证；用 Pyomo/OR-Tools 等构建基准，评估可行性与修正轮次。|4/5|
|90|Using AI Coding Agent to write high-performance and provable multi-threaded code|【Agent】编程 Agent 生成高性能多线程 Rust 程序、形式化规格，并根据验证反馈迭代修复。|围绕队列、哈希表或 B+ 树实现证明与并发测试闭环；对编程和并发验证要求高。|3/5|
|91|AI Investing Agent|【Agent】投资 Agent 分析市场并产生可解释投资建议，设置风险限制及交易前人工批准。|设计并评估人在环路的投资 Agent；需要数据分析、业务决策与较强编程能力。|4/5|
|92|Data Agent|【Agent】数据 Agent 将自然语言分析问题转为 SQL，执行查询并给出有数据依据的解释。|交付可工作的 Agent 与业务洞察；重点检查 SQL 效率、执行结果及引用依据。|5/5|
|113|LLM Agent for Adaptive Orchestration of Heterogeneous Sensor Systems|【Agent】智能家居传感 Agent：理解请求、选择传感器工具、验证依赖并生成有证据的回答。|整合约 2–3 类传感方式；实现工具注册表及计划验证，建立约 30–60 条问题的评测集。|5/5|
|121|How Long Can a Website Hold an AI Agent? Measuring the Stopping Rules of LLM Web Agents|【Agent】研究网页如何影响 LLM Web Agent 的停留时间、停止规则、Token 和成本。|设置正常页面对照；测量抓取页数、轮次、耗时和成本，提交代码、数据及报告。|4/5|
|123|Long-term personal memory from mobile and wearable data|【Agent配套】从手机和可穿戴设备数据构建长期个人记忆，供持续型个人 Agent 使用；重点在记忆系统。|构建事件提取、记忆更新和选择性处理原型；评估准确率、效率及时间一致性。|3/5|
|124|Proactive on-device AI agents|【Agent】端侧主动式 AI Agent：判断何时帮助、何时保持沉默，并可靠调用工具。|用小型语言模型做设备端原型；构建交互轨迹与评测集，测量工具调用、延迟和资源消耗。|4/5|
|127|MyopiaBench-Agent: A Multimodal Benchmark and Agent Workflow for Myopia Diagnosis|【Agent】近视诊断研究 Agent 与多模态基准：整理可追溯数据、调用分析工具并总结证据。|构建版本化数据及评测任务；Agent 必须核验工具输出、检索获批资料并在证据冲突时转人工。|4/5|
|128|Building an Expert-Guided AI Agent|【Agent】专家指导的眼科 Agent：依据获批专家资料检索证据、追问缺失信息，并适时拒答或转医生。|实现病例状态、规则和来源日志；比较基础 LLM、RAG 与规则引导工作流。|5/5|
|135|LLM Agents for Strategic Persuasion|【Agent】研究 LLM Agent 在多轮说服中的受众建模、规划、记忆和事实约束。|建立可复现实验框架，评估说服效果、事实准确性与操纵性风险；偏好有 Agent 使用经验。|4/5|
|136|LLM Agents in Bargaining Games|【Agent】研究 LLM Agent 在讨价还价游戏中的谈判策略、记忆和效用追踪。|设计重复博弈与不同对手；评估成交率、效用、公平性、一致性并分析对话记录。|4/5|
|142|Autonomous CAE: Agentic Harness for Closed-Loop Composite Prepreg Forming Optimization|【Agent】复合材料 CAE Agent 自动操作有限元流程，循环规划、模拟、检查、优化并记录决策。|实现 ReAct 式框架连接现有仿真管线，交付自动设计演示、代码和文档；需要较强工程背景。|3/5|
|43|Adapting Vision-Language Foundation Models for Endoscopic Surgical Video Understanding|【可选子任务】在内镜视频的视觉语言模型课题内，可按兴趣选择解剖结构识别、手术阶段分类或图像描述。|应用预训练 VLM，比较基线并评估所选任务；原文未承诺可完全脱离主题自拟题目。|3/5|
|44|Predictive AI for Surgical Video Analysis|【方向较开放】外科视频预测项目列出流程预测、未来动作、异常检测等多种研究方向；具体方向取决于数据和讨论。|需做至少一项时间预测任务、方法比较与准确性/鲁棒性分析；自提新想法应先与导师确认。|3/5|
|60|Machine Learning for Financial Applications|【可自选问题】原文明确写明学生选择一个感兴趣的金融问题，如预测、交易、组合管理、衍生品定价或对冲。|在金融机器学习范围内自行确定问题，开发方法并与传统基线比较；提供可复现代码。|4/5|
|85|Next-generation LLM Architecture Research|【方向较开放】新一代 LLM 架构研究可覆盖 token 交互、参数高效变换、记忆与长上下文等；范围宽但未明确承诺自由自拟。|需提出并评估架构改动的效率和泛化效果，提交技术报告与开源代码；先与导师确认具体 idea。|3/5|

注：#113 使用现有传感工具，#124 在现有移动设备上部署模型，均不以设计机器人或硬件为目标；若你希望完全不接触设备，也可跳过这两项。原表没有明确承诺可完全自由自拟题目；#60 明确允许在金融机器学习范围内自选问题。
