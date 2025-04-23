import imageio
import os
import torch
from .boundaries import compute_boundaries
from .plotting import plot_boundaries


def record_boundary_animation(model, dataset_domain,
                              epochs, save_dir='sigmacam_anim',
                              interval=1, **compute_kwargs):
    """
    Train model externally and call this between epochs to save boundary frames.
    Args:
        model: SigmaCamModel instance
        dataset_domain: Tensor domain
        epochs: iterable of epoch indices
        save_dir: folder to store frames
        interval: save every `interval` epochs
    Returns:
        mp4_path: Path to saved mp4 file
    """
    os.makedirs(save_dir, exist_ok=True)
    for epoch in epochs:
        if epoch % interval != 0:
            continue
        regions, boundary = compute_boundaries(model, dataset_domain, **compute_kwargs)
        fig, ax = plot_boundaries(regions, boundary)
        fig.savefig(f'{save_dir}/frame_{epoch:04d}.png')
        plt.close(fig)
    # compile
    images = []
    files = sorted(os.listdir(save_dir))
    for fname in files:
        if fname.endswith('.png'):
            images.append(imageio.imread(os.path.join(save_dir, fname)))
    mp4_path = os.path.join(save_dir, 'animation.mp4')
    imageio.mimwrite(mp4_path, images, fps=10, macro_block_size=None)
    return mp4_path
