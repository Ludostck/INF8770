import pandas as pd
import matplotlib.pyplot as plt

data = [
    # Kodim01
    {"Image": "Kodim01", "Espace": "RGB", "Quant": "8/8/8", "PSNR": 58.52, "SSIM": 0.9997, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim01", "Espace": "RGB", "Quant": "8/8/4", "PSNR": 52.24, "SSIM": 0.9987, "Compression": 16.67, "Score": 0.1372},
    {"Image": "Kodim01", "Espace": "RGB", "Quant": "8/8/0", "PSNR": 41.61, "SSIM": 0.9972, "Compression": 33.33, "Score": 0.2144},
    {"Image": "Kodim01", "Espace": "RGB", "Quant": "8/2/2", "PSNR": 33.39, "SSIM": 0.9665, "Compression": 50.00, "Score": 0.2615},
    {"Image": "Kodim01", "Espace": "YUV", "Quant": "8/8/8", "PSNR": 59.28, "SSIM": 0.9997, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim01", "Espace": "YUV", "Quant": "8/8/4", "PSNR": 51.26, "SSIM": 0.9982, "Compression": 16.67, "Score": 0.1339},
    {"Image": "Kodim01", "Espace": "YUV", "Quant": "8/8/0", "PSNR": 40.18, "SSIM": 0.9961, "Compression": 33.33, "Score": 0.2078},
    {"Image": "Kodim01", "Espace": "YUV", "Quant": "8/2/2", "PSNR": 32.93, "SSIM": 0.9641, "Compression": 50.00, "Score": 0.2585},
    # Kodim02
    {"Image": "Kodim02", "Espace": "RGB", "Quant": "8/8/8", "PSNR": 58.48, "SSIM": 0.9992, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim02", "Espace": "RGB", "Quant": "8/8/4", "PSNR": 52.96, "SSIM": 0.9972, "Compression": 16.67, "Score": 0.1395},
    {"Image": "Kodim02", "Espace": "RGB", "Quant": "8/8/0", "PSNR": 44.05, "SSIM": 0.9961, "Compression": 33.33, "Score": 0.2256},
    {"Image": "Kodim02", "Espace": "RGB", "Quant": "8/2/2", "PSNR": 27.31, "SSIM": 0.7908, "Compression": 50.00, "Score": 0.1647},
    {"Image": "Kodim02", "Espace": "YUV", "Quant": "8/8/8", "PSNR": 58.34, "SSIM": 0.9992, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim02", "Espace": "YUV", "Quant": "8/8/4", "PSNR": 51.63, "SSIM": 0.9960, "Compression": 16.67, "Score": 0.1348},
    {"Image": "Kodim02", "Espace": "YUV", "Quant": "8/8/0", "PSNR": 42.73, "SSIM": 0.9946, "Compression": 33.33, "Score": 0.2187},
    {"Image": "Kodim02", "Espace": "YUV", "Quant": "8/2/2", "PSNR": 26.98, "SSIM": 0.7653, "Compression": 50.00, "Score": 0.1540},
    # Kodim05
    {"Image": "Kodim05", "Espace": "RGB", "Quant": "8/8/8", "PSNR": 57.31, "SSIM": 0.9996, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim05", "Espace": "RGB", "Quant": "8/8/4", "PSNR": 39.66, "SSIM": 0.9855, "Compression": 16.67, "Score": 0.1011},
    {"Image": "Kodim05", "Espace": "RGB", "Quant": "8/8/0", "PSNR": 30.46, "SSIM": 0.9810, "Compression": 33.33, "Score": 0.1718},
    {"Image": "Kodim05", "Espace": "RGB", "Quant": "8/2/2", "PSNR": 24.09, "SSIM": 0.9223, "Compression": 50.00, "Score": 0.2153},
    {"Image": "Kodim05", "Espace": "YUV", "Quant": "8/8/8", "PSNR": 57.46, "SSIM": 0.9996, "Compression": 0.00, "Score": 0.0000},
    {"Image": "Kodim05", "Espace": "YUV", "Quant": "8/8/4", "PSNR": 40.00, "SSIM": 0.9847, "Compression": 16.67, "Score": 0.1016},
    {"Image": "Kodim05", "Espace": "YUV", "Quant": "8/8/0", "PSNR": 28.66, "SSIM": 0.9730, "Compression": 33.33, "Score": 0.1656},
    {"Image": "Kodim05", "Espace": "YUV", "Quant": "8/2/2", "PSNR": 23.08, "SSIM": 0.8395, "Compression": 50.00, "Score": 0.1777},
    # kodim13
    {"Image": "kodim13", "Espace": "RGB", "Quant": "8/8/8", "PSNR": 57.64, "SSIM": 0.9998, "Compression": 0.00, "Score": 0.0000},
    {"Image": "kodim13", "Espace": "RGB", "Quant": "8/8/4", "PSNR": 50.68, "SSIM": 0.9988, "Compression": 16.67, "Score": 0.1322},
    {"Image": "kodim13", "Espace": "RGB", "Quant": "8/8/0", "PSNR": 36.47, "SSIM": 0.9963, "Compression": 33.33, "Score": 0.1937},
    {"Image": "kodim13", "Espace": "RGB", "Quant": "8/2/2", "PSNR": 26.03, "SSIM": 0.9330, "Compression": 50.00, "Score": 0.2233},
    {"Image": "kodim13", "Espace": "YUV", "Quant": "8/8/8", "PSNR": 57.76, "SSIM": 0.9997, "Compression": 0.00, "Score": 0.0000},
    {"Image": "kodim13", "Espace": "YUV", "Quant": "8/8/4", "PSNR": 46.81, "SSIM": 0.9970, "Compression": 16.67, "Score": 0.1203},
    {"Image": "kodim13", "Espace": "YUV", "Quant": "8/8/0", "PSNR": 34.27, "SSIM": 0.9914, "Compression": 33.33, "Score": 0.1850},
    {"Image": "kodim13", "Espace": "YUV", "Quant": "8/2/2", "PSNR": 28.37, "SSIM": 0.9584, "Compression": 50.00, "Score": 0.2406},
    # kodim23
    {"Image": "kodim23", "Espace": "RGB", "Quant": "8/8/8", "PSNR": 57.56, "SSIM": 0.9989, "Compression": 0.00, "Score": 0.0000},
    {"Image": "kodim23", "Espace": "RGB", "Quant": "8/8/4", "PSNR": 43.29, "SSIM": 0.9797, "Compression": 16.67, "Score": 0.1082},
    {"Image": "kodim23", "Espace": "RGB", "Quant": "8/8/0", "PSNR": 23.84, "SSIM": 0.9497, "Compression": 33.33, "Score": 0.1518},
    {"Image": "kodim23", "Espace": "RGB", "Quant": "8/2/2", "PSNR": 25.74, "SSIM": 0.8865, "Compression": 50.00, "Score": 0.2016},
    {"Image": "kodim23", "Espace": "YUV", "Quant": "8/8/8", "PSNR": 57.70, "SSIM": 0.9989, "Compression": 0.00, "Score": 0.0000},
    {"Image": "kodim23", "Espace": "YUV", "Quant": "8/8/4", "PSNR": 41.72, "SSIM": 0.9742, "Compression": 16.67, "Score": 0.1036},
    {"Image": "kodim23", "Espace": "YUV", "Quant": "8/8/0", "PSNR": 21.84, "SSIM": 0.9329, "Compression": 33.33, "Score": 0.1454},
    {"Image": "kodim23", "Espace": "YUV", "Quant": "8/2/2", "PSNR": 25.58, "SSIM": 0.8793, "Compression": 50.00, "Score": 0.1982},
]

df = pd.DataFrame(data)

# Calcul des moyennes
avg_df = df.groupby(["Image", "Espace"])[["PSNR", "SSIM", "Score"]].mean().reset_index()

images = ["Kodim01", "Kodim02", "Kodim05", "kodim13", "kodim23"]

# graphique 
def plot_metric(metric, y_label):
    plt.figure(figsize=(15, 4.5))
    for espace, color in zip(["RGB", "YUV"], ["violet", "green"]):
        sub = avg_df[avg_df["Espace"] == espace].set_index("Image").reindex(images).reset_index()
        plt.plot(sub["Image"], sub[metric], marker="o", label=espace, color=color)
    plt.xlabel("Image")
    plt.ylabel(y_label)
    plt.title("Moyenne de " + metric + " par image")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_metric("PSNR", "PSNR (dB)")
plot_metric("SSIM", "SSIM")
plot_metric("Score", "Score")
