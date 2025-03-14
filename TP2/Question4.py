import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# i2 (lignes) et i1 (colonnes)
i2_labels = ["Kodim01", "Kodim02", "Kodim05", "Kodim13", "Kodim23"]
i1_labels = ["Kodim01", "Kodim02", "Kodim05", "Kodim13", "Kodim23"]

psnr_values = [
    [33.39, 32.87, 31.77, 32.99, 32.36],  # i2: Kodim01
    [40.25, 44.05, 26.70, 18.50, 23.32],  # i2: Kodim02
    [24.66, 24.22, 24.09, 26.84, 22.50],  # i2: Kodim05
    [24.28, 23.89, 27.45, 28.37, 27.44],  # i2: Kodim13
    [25.34, 24.43, 25.67, 25.16, 25.74]   # i2: Kodim23
]

ssim_values = [
    [0.9665, 0.9711, 0.9687, 0.9687, 0.9658],  # i2: Kodim01
    [0.9931, 0.9961, 0.9416, 0.8872, 0.8865],  # i2: Kodim02
    [0.8967, 0.9030, 0.9223, 0.9349, 0.8897],  # i2: Kodim05
    [0.8923, 0.8495, 0.9597, 0.9584, 0.9636],  # i2: Kodim13
    [0.8742, 0.8677, 0.8789, 0.8874, 0.8865]   # i2: Kodim23
]

score_values = [
    [0.2615, 0.2616, 0.2562, 0.2610, 0.2571],  # i2: Kodim01
    [0.2071, 0.2256, 0.1525, 0.1312, 0.1321],  # i2: Kodim02
    [0.2044, 0.2066, 0.2153, 0.2258, 0.1989],  # i2: Kodim05
    [0.2019, 0.1828, 0.2389, 0.2406, 0.2408],  # i2: Kodim13
    [0.1955, 0.1913, 0.1981, 0.2010, 0.2016]   # i2: Kodim23
]

df_psnr = pd.DataFrame(psnr_values, index=i2_labels, columns=i1_labels)
df_ssim = pd.DataFrame(ssim_values, index=i2_labels, columns=i1_labels)
df_score = pd.DataFrame(score_values, index=i2_labels, columns=i1_labels)

def plot_heatmap_seaborn(df, title):
    plt.figure(figsize=(12,6))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", cbar=True)
    ax.set_title(title)
    ax.set_xlabel("i1")
    ax.set_ylabel("i2")
    plt.tight_layout()
    plt.show()

plot_heatmap_seaborn(df_psnr, "Heatmap PSNR (dB)")
plot_heatmap_seaborn(df_ssim, "Heatmap SSIM")
plot_heatmap_seaborn(df_score, "Heatmap Score")
