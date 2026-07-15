<div align="center">

# SigmaCam - Exact Decision Boundary Extraction for Deep Neural Networks

### 📄 Published at IEEE IJCNN 2025 · First-author research

[![Paper](https://img.shields.io/badge/Paper-IJCNN%202025%20(PDF)-00629B?logo=ieee&logoColor=white)](paper/IJCNN_2025__SigmaCam.pdf)
[![Conference](https://img.shields.io/badge/IJCNN-2025-blue)](https://2025.ijcnn.org/)
[![Follow-up](https://img.shields.io/badge/SigmaCam%2B%2B-Under%20Review%20%40%20IEEE%20TNNLS-orange)]()

**📥 [Read the paper (PDF in this repo) →](paper/IJCNN_2025__SigmaCam.pdf)** · IEEE Xplore link coming once the proceedings are indexed

</div>

---


## 🔬 Overview
**SigmaCam** is a method for **extracting the *exact* decision boundary of deep neural networks** - a contribution to neural-network **interpretability and verification**. Where most explainability tools approximate or sample where a model changes its prediction, SigmaCam targets the boundary precisely, enabling more faithful analysis of how a trained network separates classes.

- **Venue:** International Joint Conference on Neural Networks (**IJCNN 2025**), IEEE.
- **Role:** First author.
- **Follow-up:** **SigmaCam++** - extended method, submitted to **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)** *(under review)*.

> `ADD_ABSTRACT_HERE` - paste the paper abstract (2-4 sentences) so readers get the precise contribution and results.

## 🎯 Why it matters
- **Interpretability:** exact boundaries reveal *why* and *where* a model makes decisions, beyond saliency-map approximations.
- **Robustness / verification:** decision-boundary geometry relates directly to adversarial vulnerability and model reliability.
- **Research depth:** a peer-reviewed IEEE contribution demonstrates the ability to formulate a novel method, implement it, and validate it rigorously.

## 🧩 Contributions (fill from paper)
- `ADD_CONTRIBUTION_1` (e.g., the exact-boundary extraction algorithm and its formulation)
- `ADD_CONTRIBUTION_2` (e.g., complexity / guarantees vs. approximate methods)
- `ADD_KEY_RESULT` (e.g., datasets/architectures evaluated and headline metrics)

## 📊 Figures
Add 1-2 key figures from the paper here (e.g., extracted boundary visualization, comparison vs. baselines):
```
figures/boundary_example.png
figures/comparison.png
```

## 📚 Citation
```bibtex
@inproceedings{afrin2025sigmacam,
  title     = {SigmaCam: Exact Decision Boundary Extraction for Deep Neural Networks},
  author    = {Afrin, Nikkat and ADD_COAUTHORS},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {ADD_DOI}
}
```

## 🛠️ Tech stack
`Python` · `PyTorch / TensorFlow` *(confirm which)* · `NumPy` · `Matplotlib` · deep-learning interpretability

---
*First-author IEEE IJCNN 2025 publication. Portfolio-facing overview; the implementation, Colab exa