import matplotlib.pyplot as plt
import numpy as np

def plot_distortion(dists, mean_initial_distortion, min_initial_distortion):
    plt.figure(figsize=(6, 6))

    x1 = np.linspace(0, 100, 50)
    y1 = dists[0:100:2]
    x2 = np.linspace(100, 5000, 490)
    y2 = dists[100:5000:10]

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    plt.plot(x, y, label="Optimized Distortion", color='black')
    plt.xlabel("Iterations")
    plt.ylabel("Distortion")
    plt.title("Distortion Over Iterations", fontsize=15, pad=20)
    plt.ylim(0, max(y) + 0.2)
    plt.axhline(mean_initial_distortion, color='red', linestyle='-', linewidth=1,
                label="Average Distortion of N(0, 1)")
    plt.axhline(min_initial_distortion, color='red', linestyle='--', linewidth=1,
                label="Minimum Distortion of N(0, 1)")
    plt.legend(loc='upper right')
    plt.savefig("JL_distortion.png", format='png')
    plt.show()

def plot_variance(s2s):
    plt.figure(figsize=(6, 6))
    plt.plot(s2s, label=r"Variance $\sigma^2$", color="black")
    plt.xlabel("Iterations")
    plt.ylabel(r"Variance $\sigma^2$")
    plt.title(r"Variance $\sigma^2$ Over Iterations", fontsize=15, pad=20)
    plt.legend(loc='upper right')
    plt.savefig("JL_variance.png", format='png')
    plt.show()
