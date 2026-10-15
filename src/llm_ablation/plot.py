"""
    Matplotlib plotting helpers shared by the notebooks.
"""

try:
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
except Exception as e:
    raise(e)

# plot the logit drop for each head in each layer
def plot_ablation_heatmap(
    matrix: np.ndarray, title: str = "Head ablation: logit-diff drop",
    cmap: str = "viridis",
    aspect: str = "auto",
    xlabel: str = "head", ylabel: str = "layer",
    label: str = "logit-diff drop"
) -> Figure:
    """
        [layers x heads] matrix -> heatmap, layers on the y-axis (0 at top).
    """
    fig, ax = plt.subplots(figsize=(0.5 * matrix.shape[1] + 2, 0.5 * matrix.shape[0] + 2))
    im = ax.imshow(matrix, aspect=aspect, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()

    # return Figure object for display
    return fig
