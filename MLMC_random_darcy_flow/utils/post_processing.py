import re
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_from_file(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Did you run the C++ code first or is it perhaps in the wrong folder?")
        return

    levels, means, variances, samples = [], [], [], []
    
    pattern = re.compile(r"FINAL_STATS Level:(\d+) Mean:([\d\.e+-]+) Var:([\d\.e+-]+) Samples:(\d+)")

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                levels.append(int(match.group(1)))
                means.append(abs(float(match.group(2)))) # Absolute for log-plot
                variances.append(float(match.group(3)))
                samples.append(int(match.group(4)))

    if not levels:
        print("No valid MLMC data found in the file.")
        return

    # Convert to arrays for plotting
    levels = np.array(levels)
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Variance Decay
    axs[0].plot(levels, variances, 'o-', color='firebrick', label=r'Var[$P_l - P_{l-1}$]')
    axs[0].set_yscale('log')
    axs[0].set_title('Variance Decay', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Level')
    axs[0].grid(True, which='both', alpha=0.3)
    axs[0].legend()

    # 2. Mean Difference (Bias)
    axs[1].plot(levels, means, 's-', color='royalblue', label=r'|$E[P_l - P_{l-1}]$|')
    axs[1].set_yscale('log')
    axs[1].set_title('Mean Difference (Bias)', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Level')
    axs[1].grid(True, which='both', alpha=0.3)
    axs[1].legend()

    # 3. Samples per Level
    axs[2].bar(levels, samples, color='seagreen', alpha=0.7)
    axs[2].set_yscale('log')
    axs[2].set_title('Samples per Level (Workload)', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Level')
    axs[2].set_ylabel('$N_l$')
    axs[2].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('mlmc_plots.png', dpi=150) 
    print("Plot saved as 'mlmc_plots.png'")
    plt.show()

if __name__ == "__main__":
    plot_from_file('mlmc_results.txt')

    