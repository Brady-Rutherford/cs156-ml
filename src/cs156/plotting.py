"""Shared plotting helpers. Import from notebooks with:

    from cs156.plotting import show_grid
"""
import matplotlib.pyplot as plt


def show_grid(images, labels=None, cols=8, figsize=(10, 3)):
    """Show a row/grid of small images (e.g. MNIST digits)."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="gray")
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=8)
    plt.tight_layout()
    return fig
