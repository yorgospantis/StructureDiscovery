import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def plot_weight_trajectory(result, filename="NNs_yes_biases.png"):
    FONT_SIZE = 28
    
    plt.rc('font', size=FONT_SIZE)
    fig, ax = plt.subplots(figsize=(11.5, 9))

    # Custom colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['Red', 'Blue'])

    x_val = np.arange(-5, 5, 0.01)
    plt.plot(x_val, x_val, color='k', linestyle='--', linewidth=3.5, label='$\operatorname{span}(u)$', zorder=1)

    skip = 10

    # Plot weight evolution
    for i in range(len(result)):
        plt.scatter(result[i, ::skip, 0], result[i, ::skip, 1], alpha=0.5, s=1,
                    color=cmap((i + 1) / (len(result) * 1.025)))

    # Highlight initial and final points
    plt.scatter(result[0, ::skip, 0], result[0, ::skip, 1], color='Red', alpha=0.85, s=100, label='Initialized')
    plt.scatter(result[-1, ::skip, 0], result[-1, ::skip, 1], color='Blue', alpha=0.9, s=100, zorder=2, label='Trained')

    # Set axis limits from -4 to 4
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    # Format tick labels
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_yticks(np.arange(-4, 5, 2))

    # Add title
    plt.title("Weight Progression With Perturbed Gradient Descent", fontsize=20, pad=20)

    # Improved legend placement
    legend = plt.legend(fontsize=22, loc='upper left', frameon=True, facecolor='white', edgecolor='grey', fancybox=True)
    legend.get_frame().set_linewidth(2)

    plt.savefig(finename, format='png', bbox_inches='tight')
    plt.show()
