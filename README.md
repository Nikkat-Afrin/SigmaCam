# SigmaCam: Exact Decision Boundary Extraction for DNNs with Smooth Nonlinearities

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK_HERE)

### [Paper Link](YOUR_PAPER_PDF_LINK_HERE) | [Website](YOUR_PROJECT_WEBSITE_LINK_HERE)

<img src="YOUR_MAIN_IMAGE_LINK_HERE" width="100%">

Fig: Decision boundary extraction using SigmaCam for a spiral dataset demonstrating clear and smooth decision contours using Sigmoid and SiLU activations.

#### Authors: Anonymous (to be updated post-acceptance)

**Abstract:** Understanding how a trained deep neural network (DNN) classifies input data is critical for interpretability and trust in AI systems. While existing tools such as SplineCam visualize exact decision boundaries for neural networks with piecewise polynomial activation functions, they do not support smooth activations like Sigmoid and SiLU commonly used in contemporary DNNs. SigmaCam addresses this gap by introducing a computationally efficient, theoretically exact recursive algorithm capable of generating decision boundaries for Multi-Layer Perceptrons (MLPs) employing smooth nonlinear activation functions. SigmaCam extends previous visualization methods to a broader class of activations, allowing precise visualization of decision boundaries across various data domains and network architectures, significantly enhancing model interpretability and transparency.

/assets/spiral%20sigmoid%20sigmacam.mp4![SigmaCam in Action](/assets/spiral%20sigmoid%20sigmacam.mp4)

**Video:** SigmaCam visualization of decision boundaries evolving during training of a neural network with SiLU activations on a spiral dataset. Observe how the decision boundary smoothly adapts to the learned representation.

## Examples

Examples are available in the `./examples` folder, with Google Colab notebooks provided for interactive demonstrations:

| Model | Data | Filename | Link |
| :---- | :---- | :---- | :---- |
| MLP (Sigmoid) | Spiral Dataset | spiral_sigmoid_sigmacam.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hIumz2lmjULSWqy7EDZc0LijIg1UhgQh?usp=sharing) |
| MLP (Sigmoid) | MNIST (Digit 2 vs 3) | mnist_binary.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15-lAqQua9Q5oYl2NJiS0RL86efbN7Chs?usp=sharing) |
| MLP (SiLU) | PneumoniaMNIST | pneumonia_mnist.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/188trnP4XmtNIuvsougGMfQKzEWu3XM1U?usp=sharing) |
| MLP (SiLU) | Synthetic 3D Sphere | sphere_classification.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1twiGdHsaZ6H4yz4cZuAWw8HqkHmbZqDF?usp=sharing) |
| MLP (SiLU) | Man Face | man_face_SiLU_2D_INR.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gJ4fRKg4pp7zMuJNjRXXMIgvHce_v4D1?usp=sharing) |
| MLP (SiLU) | Sphere Image | sphere_image_Periodic_encoding_INR.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15lTdQIty1HCInuurb9qJo6AsbtuEclkE?usp=sharing) |

## Usage

SigmaCam can visualize exact decision boundaries by projecting high-dimensional inputs to a two-dimensional plane, computing a structured grid, and recursively extracting boundaries using a trained neural network.

Example usage:

```python
import torch
import sigmacam

# 1. Prepare your PyTorch model
model = YourTrainedModel()
model.cuda(); model.eval()

# 2. Compute projection and centroid
data = torch.tensor(...)  # shape (N, D)
T, x0 = sigmacam.utils.get_projection_matrix_and_centroid(data)

# 3. Wrap your model
NN = sigmacam.wrapper(model, input_shape=(D,), T=T, centroid=x0, device='cuda')

# 4. Compute boundaries
regions, boundary = sigmacam.compute_boundaries(NN, domain=data, grid_size=200)

# 5. Plot
fig, ax = sigmacam.plot_boundaries(regions, boundary)
plt.show()

# 6. (Optional) Record animation during training
# mp4 = sigmacam.record_boundary_animation(NN, data, epochs=range(0,201), save_dir='anim', interval=10)
# print(f"Animation saved at {mp4}")
```

## Requirements

SigmaCam utilizes PyTorch and standard numerical libraries:

```
torch>=1.9
numpy
matplotlib
tqdm
scikit-learn
imageio
imageio-ffmpeg
```

## Installation

Clone the repository:
```bash
git clone https://github.com/YourUsername/SigmaCam.git
cd SigmaCam
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Release Notes

SigmaCam supports a broad range of smooth nonlinear activations, notably:

```python
torch.nn.Sigmoid,
torch.nn.SiLU,
torch.nn.ReLU
```

It is optimized for GPU acceleration and efficiently computes exact boundaries via a recursive algorithm.

## To Do

1. Add detailed benchmarks across more complex networks
2. Include additional datasets for real-world validation
3. Provide integration for transformer architectures
4. Expand documentation and tutorials

## Citation

If you find SigmaCam helpful, please cite our paper:

```latex
@inproceedings{anonymous2025sigmacam,
  title={SigmaCam: Exact Decision Boundary Extraction for DNNs with Smooth Nonlinearities},
  author={Anonymous},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
}
```

