import numpy as np
import os
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def rgb_to_yuv(image_rgb):
    
    transform_matrix = np.array([[0.299,     0.587,     0.114],
                                 [-0.14713,  -0.28886,   0.436],
                                 [0.615,     -0.51499,  -0.10001]])
    H, W, C = image_rgb.shape
    flat_rgb = image_rgb.reshape(-1, 3)
    flat_yuv = flat_rgb @ transform_matrix.T
    return flat_yuv.reshape(H, W, C)

def yuv_to_rgb(image_yuv):
    inv_matrix = np.array([[ 1.0,       0.0,       1.13983],
                           [ 1.0,      -0.39465,  -0.58060],
                           [ 1.0,       2.03211,   0.0     ]])
    H, W, C = image_yuv.shape
    flat_yuv = image_yuv.reshape(-1, 3)
    flat_rgb = flat_yuv @ inv_matrix.T
    return flat_rgb.reshape(H, W, C)

def compute_kl_transform_basis(image):
    """
    Calcule la base KL 
    Retourne le vecteur moyen et la matrice de base
    """
    H, W, C = image.shape
    flat_data = image.reshape(-1, C)
    mean_vec = np.mean(flat_data, axis=0)
    centered_data = flat_data - mean_vec
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    return mean_vec, eigen_vectors

def apply_kl_transform(image, mean_vec, kl_basis):
    "Projette l'image dans la nouvelle base KL."
    H, W, C = image.shape
    flat_data = image.reshape(-1, C)
    centered_data = flat_data - mean_vec
    transformed = centered_data @ kl_basis
    return transformed.reshape(H, W, C)

def apply_inverse_kl_transform(transformed_image, mean_vec, kl_basis):
    "Reconstruit l'image"
    H, W, C = transformed_image.shape
    flat_data = transformed_image.reshape(-1, C)
    original = flat_data @ kl_basis.T + mean_vec
    return original.reshape(H, W, C)

def quantize_image(image, bits_per_channel):
    """
    Quantifie chaque canal de l'image selon le nombre de bits spécifié.
    Pour chaque canal :
     - on normalise
     - on arrondit sur le nombre de niveaux
     - on remappe à l'intervalle d'origine
    """
    quantized = np.zeros_like(image)
    C = image.shape[-1]
    for c in range(C):
        nb_bits = bits_per_channel[c]
        channel = image[..., c]
        if nb_bits <= 0:
            quantized[..., c] = 0
            continue
        c_min, c_max = channel.min(), channel.max()
        if np.isclose(c_min, c_max):
            quantized[..., c] = channel
            continue
        levels = 2 ** nb_bits - 1
        norm_channel = (channel - c_min) / (c_max - c_min)
        q_channel = np.round(norm_channel * levels)
        quantized[..., c] = (q_channel / levels) * (c_max - c_min) + c_min
    return quantized

def compute_psnr(original, reconstructed):
    return peak_signal_noise_ratio(original, reconstructed, data_range=original.max()-original.min())

def compute_ssim(original, reconstructed, win_size=7):
    H, W = original.shape[:2]
    adapted_win_size = min(win_size, H, W)
    if adapted_win_size % 2 == 0:
        adapted_win_size -= 1
    return structural_similarity(original, reconstructed,
                                 data_range=original.max()-original.min(),
                                 channel_axis=-1,
                                 win_size=adapted_win_size)

def compute_compression(original_bits_per_pixel, quant_bits_per_channel, image_shape):
    """
    (1 - (Tc/To)) * 100, où Tc = somme des bits après quantification,
    To = bits par pixel originaux (ici 24).
    """
    H, W = image_shape
    total_pixels = H * W
    original_size_bits = total_pixels * original_bits_per_pixel
    new_size_bits = total_pixels * sum(quant_bits_per_channel)
    if new_size_bits == 0:
        return 100.0
    ratio = original_size_bits / new_size_bits
    compression_pct = (1.0 - 1.0 / ratio) * 100.0
    return max(0.0, min(compression_pct, 100.0))

def main():

    path_img = "data/kodim23.png"  # Chemin de l'image à traiter
    quant_schemes = {
        "8/8/8": [8, 8, 8],
        "8/8/4": [8, 8, 4],
        "8/8/0": [8, 8, 0],
        "8/2/2": [8, 2, 2]
    }
    original_bits_per_pixel = 24  # 8 bits par canal sur 3 canaux
    
    
    img = io.imread(path_img)
    img_float = img_as_float(img).astype(np.float32)
    H, W = img.shape[:2]
    
    # Dossier de sortie pour les images reconstruites
    output_folder = "data_q1/kodim23"
    os.makedirs(output_folder, exist_ok=True)
    
    results = []
    
    # Espaces colorimétriques
    color_spaces = {
        "RGB": lambda x: x,
        "YUV": rgb_to_yuv
    }
    
    # Boucle sur chaque espace colorimétrique
    for cs_name, cs_func in color_spaces.items():
        # Conversion éventuelle
        img_cs = cs_func(img_float)
        # Calcul de la base KL 
        mean_vec, kl_basis = compute_kl_transform_basis(img_cs)
        transformed = apply_kl_transform(img_cs, mean_vec, kl_basis)
        
        # Boucle sur quantification
        for qs_name, bits in quant_schemes.items():
            quantized = quantize_image(transformed, bits)
            recon_kl = apply_inverse_kl_transform(quantized, mean_vec, kl_basis)
            if cs_name == "YUV":
                reconstructed = yuv_to_rgb(recon_kl)
            else:
                reconstructed = recon_kl
            reconstructed = np.clip(reconstructed, 0, 1)
            
            output_filename = f"image_rec_{cs_name}_{qs_name.replace('/', '_')}.png"
            output_path = os.path.join(output_folder, output_filename)
            io.imsave(output_path, img_as_ubyte(reconstructed))
            
            # Calcul des métriques
            psnr_val = compute_psnr(img_float, reconstructed)
            ssim_val = compute_ssim(img_float, reconstructed, win_size=5)
            comp_pct = compute_compression(original_bits_per_pixel, bits, (H, W))

            # Score 
            psnr_norm = min(1,max(0,(psnr_val-20)/40))
            score = comp_pct/100 * (ssim_val**2+psnr_norm**2)/2
            
            results.append({
                "Color Space": cs_name,
                "Quant Scheme": qs_name,
                "PSNR (dB)": psnr_val,
                "SSIM": ssim_val,
                "Compression (%)": comp_pct,
                "Score": score,
                "Output Image": output_path
            })
            print(f"{cs_name} {qs_name}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}, Compression = {comp_pct:.2f} %, Score = {score:.4f}")
    
    # Graphiques 
    labels = [f"{r['Color Space']}\n{r['Quant Scheme']}" for r in results]
    psnr_values = [r["PSNR (dB)"] for r in results]
    ssim_values = [r["SSIM"] for r in results]
    comp_values = [r["Compression (%)"] for r in results]
    
    rgb_colors = ["#6600DB", "#8921FF", "#A95CFF", "#CC9EFF"]
    yuv_colors = ["#006E12", "#15952A", "#4AC95F", "#88FD9C"]
    rgb_counter = 0
    yuv_counter = 0
    bar_colors = []

    for r in results:
        if r["Color Space"] == "RGB":
            color = rgb_colors[rgb_counter % len(rgb_colors)]
            rgb_counter += 1
        else:
            color = yuv_colors[yuv_counter % len(yuv_colors)]
            yuv_counter += 1
        bar_colors.append(color)
    
    x = np.arange(len(labels))
    width = 0.6
    
    #  PSNR
    plt.figure(figsize=(6,8))
    bars = plt.bar(x, psnr_values, width, color=bar_colors)
    plt.xticks(x, labels)
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR pour chaque compression")
    plt.grid(axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        quant_text = results[i]["Quant Scheme"]
        plt.text(bar.get_x() + bar.get_width()/2, height, quant_text,
                 ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    #  SSIM
    plt.figure(figsize=(6,8))
    bars = plt.bar(x, ssim_values, width, color=bar_colors)
    plt.xticks(x, labels)
    plt.ylabel("SSIM")
    plt.title("SSIM pour chaque compression")
    plt.grid(axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        quant_text = results[i]["Quant Scheme"]
        plt.text(bar.get_x() + bar.get_width()/2, height, quant_text,
                 ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Compression (%)
    plt.figure(figsize=(6,8))
    bars = plt.bar(x, comp_values, width, color=bar_colors)
    plt.xticks(x, labels)
    plt.ylabel("Compression (%)")
    plt.title("Taux de Compression pour chaque compression")
    plt.grid(axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        quant_text = results[i]["Quant Scheme"]
        plt.text(bar.get_x() + bar.get_width()/2, height, quant_text,
                 ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.show()
    
   
    # meilleure configuration
    best_config = max(results, key=lambda x: x["Score"])
    print("\nMeilleure configuration (Score = (1 - sum(bits)/24)*SSIM) :")
    print(f"Color Space: {best_config['Color Space']}, Quant Scheme: {best_config['Quant Scheme']}")
    print(f"Score = {best_config['Score']:.4f}, PSNR = {best_config['PSNR (dB)']:.2f} dB, SSIM = {best_config['SSIM']:.4f}, Compression = {best_config['Compression (%)']:.2f} %")

if __name__ == "__main__":
    main()
