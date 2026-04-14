<div align="center">

## 🛰️ CogRestore: Chain-of-Thought Reasoning for All-in-One Remote Sensing Image Restoration

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---

## 📖 Overview

**CogRestore** is a cognitive-driven framework that introduces **Chain-of-Thought (CoT) reasoning** into All-in-One Remote Sensing Image Restoration (AiORSIR). It transforms human-like cognitive processes into structured textual descriptions and injects them as multimodal priors to guide the restoration of multiple degradation types within a single unified model.

### ✨ Highlights

- **CoT Reasoning for Restoration**: First framework to introduce chain-of-thought reasoning into RS image restoration.
- **Physical Reasoning Module**: Decomposes degradation characteristics into structured residual priors.
- **Dynamic Prompt Generator**: Fuses textual scene descriptions with physical priors into a multimodal CoT representation.
- **Iterative Coarse-to-Fine Refinement**: Two-pass mechanism that progressively eliminates artifacts under text-guided semantic constraints.

<div align="center">
  <img src="figs/framework.png" width="90%">
</div>

---

## 🔧 Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.7

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/CogRestore.git
cd CogRestore


---

This is the official PyTorch codes for the paper:

>**CogRestore: Chain-of-Thought Reasoning for All-in-One Remote Sensing Image Restoration**<br>  [Xu Zhang<sup>1</sup>](https://house-yuyu.github.io/), [Xuhui Cao<sup>1</sup>](), [Kangzhe Yuan<sup>1</sup>](), [Laibin Chang<sup>1</sup>](), [Huan Zhang<sup>2</sup>](), [Lefei Zhang<sup>1📧</sup>](https://scholar.google.com.hk/citations?user=BLKHwNwAAAAJ&hl=zh-CN)<br>
> <sup>1</sup>Wuhan University, <sup>2</sup>Guangdong University of Technology<br>
> <sup>📧</sup>Corresponding author.

![teaser_img](fig/model.png)


:star: If CogRestore is helpful to your images or projects, please help star this repo. Thank you! :point_left:


## Contact

If you have any questions, please feel free to reach us out at <a href="zhangx0802@whu.edu.cn">zhangx0802@whu.edu.cn</a>.
