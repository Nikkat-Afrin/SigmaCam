<div align="center">

# SigmaCam - Exact Decision Boundary Extraction for Deep Neural Networks

### 📄 Published at IEEE IJCNN 2025 · First-author research

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-Read%20the%20Paper-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11227622)
[![Conference](https://img.shields.io/badge/IJCNN-2025-blue)](https://2025.ijcnn.org/)

**📥 [Read the paper on IEEE Xplore →](https://ieeexplore.ieee.org/abstract/document/11227622)** · [PDF in this repo](paper/IJCNN_2025__SigmaCam.pdf) · [Slides](presentation/SigmaCam_v1.pdf)

</div>

---

## 🔬 Overview
**SigmaCam** is a method for **extracting the *exact* decision boundary of deep neural networks** - a contribution to neural-network **interpretability and verification**. Where most explainability tools approximate or sample where a model changes its prediction, SigmaCam targets the boundary precisely, enabling more faithful analysis of how a trained network separates classes.

- **Venue:** International Joint Conference on Neural Networks (**IJCNN 2025**), IEEE.
- **Role:** First author.

> **Abstract.** While existing tools such as SplineCam visualize exact decision boundaries for neural networks with piecewise polynomial activation functions, they do not support smooth activations like Sigmoid and SiLU commonly used in contemporary DNNs. SigmaCam addresses this gap with a computationally efficient, theoretically exact recursive algorithm that generates decision boundaries for MLPs employing smooth nonlinear activation functions, enabling precise visualization across data domains and architectures.

## 🎯 Why it matters
- **Interpretability:** exact boundaries reveal *why* and *where* a model makes decisions, beyond saliency-map approximations.
- **Robustness / verification:** decision-boundary geometry relates directly to adversarial vulnerability and model reliability.
- **Research depth:** a peer-reviewed IEEE contribution demonstrates the ability to formulate a novel method, implement it, and validate it rigorously.

## 🧩 Contributions
- A theoretically exact, recursive decision-boundary extraction algorithm for MLPs with **smooth** activations (Sigmoid, SiLU) - a class prior exact methods (e.g., SplineCam) do not cover.
- A computationally efficient PyTorch implementation that scales to real inputs via 2-D PCA projections of high-dimensional domains.
- Validation across synthetic (spirals, spheres), image (MNIST, PneumoniaMNIST), and implicit-neural-representation domains, with interactive Colab notebooks for each.

## 📊 See it in action
The repo includes exact-boundary training animations for SiLU and Sigmoid MLPs on the two-spiral task (`assets/`), plus 8 Colab notebooks in the [README examples table](README.md#examples).

## 📚 Citation
```bibtex
@inproceedings{afrin2025sigmacam,
  title     = {SigmaCam: Exact Decision Boundary Extraction for DNNs with Smooth Nonlinearities},
  author    = {Afrin, Nikkat and others},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/abstract/document/11227622}
}
```

## 🛠️ Tech stack
`Python` · `PyTorch` · `NumPy` · `Matplotlib` · deep-learning interpretability

---
*First-author IEEE IJCNN 2025 publication. Portfolio-facing overview; implementation, tests, and Colab examples live in this repository.*
