import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def plot_weight_trajectory(result, filename="NNs_yes_biases.png"):
    FONT_SIZE = 28
    plt.rc('font', size=FONT_SIZE)
    fig, ax = plt.subplots(figsize=(11.5, 9))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['Red', 'Blue'])

    x_val = np.arange(-5, 5, 0.01)
    plt.plot(x_val, x_val, color='k', linestyle='--', linewidth=3.5,
             label='$\\operatorname{span}(u)$', zorder=1)

    skip = 10
    for i in range(len(result)):
        plt.scatter(result[i, ::skip, 0], result[i, ::skip, 1], alpha=0.5, s=1,
                    color=cmap((i + 1) / (len(result) * 1.025)))

    plt.scatter(result[0, ::skip, 0], result[0, ::skip, 1], color='Red', alpha=0.85, s=100, label='Initialized')
    plt.scatter(result[-1, ::skip, 0], result[-1, ::skip, 1], color='Blue', alpha=0.9, s=100, zorder=2, label='Trained')

    plt.xlim([-4, 4]); plt.ylim([-4, 4])
    ax.set_xticks(np.arange(-4, 5, 2)); ax.set_yticks(np.arange(-4, 5, 2))
    plt.title("Weight Progression With Training Biases", fontsize=20, pad=20)

    legend = plt.legend(fontsize=22, loc='upper left', frameon=True, facecolor='white',
                        edgecolor='grey', fancybox=True)
    legend.get_frame().set_linewidth(2)

    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()
