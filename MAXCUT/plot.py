import matplotlib.pyplot as plt

def plot_sigma_evolution(max_sigmas):
    plt.figure(figsize=(6, 6))
    plt.plot(max_sigmas, label=r"Maximum $\sigma$", color="black")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\sigma$")
    plt.title(r"Maximum $\sigma$ Over Iterations", fontsize=15, pad=20)
    plt.legend(loc='upper right')
    plt.savefig("maxcut_sigma.png", format='png')
    plt.show()

def plot_cut_values(cut_values, max_cut_value):
    plt.figure(figsize=(6, 6))
    plt.plot(cut_values, label="Cut Value Over Iterations", color="black")
    plt.axhline(y=max_cut_value, color='red', linestyle='--', linewidth=1, label="Optimal Cut Value")
    plt.xlabel("Iterations")
    plt.ylabel("Cut Value")
    plt.title("Cut Value Over Iterations", fontsize=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.savefig("maxcut_cut.png", format='png')
    plt.show()
