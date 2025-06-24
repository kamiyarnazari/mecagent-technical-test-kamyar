# CADQuery Code Generation - GenCAD Challenge Submission

## The Task
Create the best CadQuery code generator model.

![GenCAD demonstrates generative CAD models](assets/demo.png)

### Challenge Steps:
1. Load the dataset (147K pairs of Images/CadQuery code).
2. Create a baseline model and evaluate it with the given metrics.
3. Enhance the baseline model by any means and evaluate again.
4. Explain your choices and potential bottlenecks.
5. Show what enhancements you would have done with more time.

> Creativity and methodology matter more than absolute results!

---

## Step 1: Literature-Informed Design

After reviewing recent works and their approaches (see the table below), the following insights informed our methodology:

### Notable References:
- GenCAD & CSGNet use custom architectures trained from scratch
- CAD-Coder (LLaVA 1.5) fine-tunes vision-language models
- LLM4CAD, CAD-Assistant rely on GPT-4(o) with tool augmentation

Our goal: develop a **practical**, **open**, and **extensible** solution that aligns with CadQuery’s structure, and is trainable on commodity GPUs.

![OVERVIEW OF SOME EXISTING APPROACHES FOR CONDITIONAL, EDITABLE-CAD GENERATION](assets/model_comparison.png)


Source: Doris, A. C., Alam, M. F., Nobari, A. H., & Ahmed, F. (2025, May 20). CAD-Coder: An Open-Source Vision-Language model for Computer-Aided design code generation. [arXiv:2505.14646](https://arxiv.org/abs/2505.14646)

---

##  Step 2: Baseline Model - Vision Encoder + GPT-2

We developed a simple, extensible baseline based on:

-  **Visual Encoder**: Pretrained ResNet18 for grayscale CAD image embeddings
-  **Language Decoder**: GPT-2 (gpt2) autoregressive transformer
-  **Fusion**: Visual embeddings projected to token space and prepended

### Limitations
- GPT-2 struggles with indentation-sensitive Python code
- Frequent CadQuery syntax errors
- Lacks code-specific pretraining

---

##  Step 3: Enhancement with Salesforce CodeGen

### Why `Salesforce/codegen-350M-mono`?
-  **Tailored for code generation** (unlike GPT-2)
-  **Monolingual Python** model (perfect for CadQuery DSL)
-  **Better results on syntax-aware metrics** (e.g., VSR, IoU)
-  **Lightweight & fine-tunable** on a single GPU
-  **Compatible with self-debugging loops** to repair code

This resulted in higher Valid Syntax Rates and improved geometric alignment.

---

##  Step 4: Bottlenecks and Observations

### Bottlenecks:
-  Lack of loop context or multiple-sketch memory
-  Limited training time for full convergence
-  Errors in `loop1` generation not resolved without reference tracking
-  Visual encoder not fine-tuned with decoder jointly (yet)

---

##  Step 5: Future Enhancements (from GenCAD repo)

-  **Train ResNet + Decoder jointly** for better alignment
-  **Use multi-sketch memory** or recurrence for CAD programs with multiple operations
-  **Implement CAD-Code-aware loss** using structural program diffs
-  **Pretrain with synthetic CAD corpus** before fine-tuning on GenCAD
-  **Use tool-augmented LLMs** (e.g. GPT4-o + debugger)
-  **Evaluation with execution-based metrics** (e.g., rendered IoU)

---

##  Conclusion
This project demonstrates a full training loop, from baseline GPT-2 to CodeGen-based enhancement. While absolute metrics remain improvable, the architecture choices and pipeline provide a solid foundation for real-world CAD code generation.

