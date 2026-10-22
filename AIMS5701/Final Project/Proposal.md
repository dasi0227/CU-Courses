# A Coding Agent for Editable Presentations — HJ06 Proposal

## Team Member

| Name      | Student ID |
| --------- | ---------- |
| Wan Yiwei | 1155261963 |

## Project Understanding

This project focuses on building a coding agent that converts source materials into editable presentations through structured slide specifications and code generation. It combines content planning, reusable layouts, rendering, and local repair, with evaluation covering factual fidelity, readability, layout defects, and editability compared with direct code generation.

## Background and Related Work

Presentation generation requires accurate content selection, coherent organization, and readable visual design. Direct code generation can produce incomplete content or defective layouts even when the code executes successfully. [PPTAgent (Zheng et al., 2025)](https://arxiv.org/abs/2501.03936) analyzes reference presentations, plans outlines, and generates slides through iterative editing, evaluating content, design, and coherence. [DeepPresenter (Zheng et al., 2026)](https://arxiv.org/abs/2602.22839) uses observations of rendered artifacts to guide reflection and revision. These studies motivate our use of structured planning and observable feedback. This project will investigate their practical benefits and costs within an LLM-based presentation authoring system through controlled comparisons.

## Implementation Plan

The system will represent slides as HTML/CSS to preserve editable text, visual elements, and layout rules. A multimodal LLM supporting visual input and tool use will coordinate content planning, source-code editing, component reuse, and rendering diagnostics. The model version, inference settings, and execution budget will be recorded for reproducibility. The workflow and evaluation are organized as follows.

1. **Content Planning:** Organize source materials and audience requirements into an outline and structured slide specifications defining each slide's message, content, and layout.
2. **Code Generation:** Generate editable HTML/CSS slides using a shared theme and reusable components, supporting targeted content and layout changes.
3. **Rendering and Repair:** Render slides on the CPU and use diagnostics and screenshot comparisons to identify layout defects. Revise the code for up to three repair rounds while preserving content.
4. **Evaluation:** Evaluate the presentations for content accuracy, readability, layout quality, and editability, and compare them with slides generated directly by an LLM.
