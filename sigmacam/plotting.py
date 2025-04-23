import matplotlib.pyplot as plt


def plot_boundaries(regions, decision_boundary,
                    cmap='coolwarm', fig=None, ax=None):
    """
    Plot region heatmap and overlay decision boundary.
    Args:
        regions: dict from compute_boundaries
        decision_boundary: list of arrays for contour lines
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    X, Y, Z = regions['X'], regions['Y'], regions['Z']
    cf = ax.contourf(X, Y, Z, levels=100, cmap=cmap, alpha=0.7)
    for line in decision_boundary:
        ax.plot(line[:,0], line[:,1], color='k', linewidth=2)
    ax.set_aspect('equal')
    ax.set_title('Decision Boundary')
    return fig, ax
