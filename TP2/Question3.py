import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rgb_to_yuv(image_rgb):

    transform_matrix = np.array([[0.299,     0.587,     0.114],
                                 [-0.14713,  -0.28886,   0.436],
                                 [0.615,     -0.51499,  -0.10001]])
    shape = image_rgb.shape
    flat_rgb = image_rgb.reshape(-1, 3)
    flat_yuv = flat_rgb @ transform_matrix.T
    image_yuv = flat_yuv.reshape(shape)
    return image_yuv

def yuv_to_rgb(image_yuv):

    inv_matrix = np.array([[ 1.0,  0.0,       1.13983 ],
                           [ 1.0, -0.39465,  -0.58060 ],
                           [ 1.0,  2.03211,   0.0      ]])
    shape = image_yuv.shape
    flat_yuv = image_yuv.reshape(-1, 3)
    flat_rgb = flat_yuv @ inv_matrix.T
    image_rgb = flat_rgb.reshape(shape)
    return image_rgb

def compute_kl_transform_basis(image):

    H, W, C = image.shape
    flat_data = image.reshape(-1, C)
    mean_vec = np.mean(flat_data, axis=0)
    centered_data = flat_data - mean_vec
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    kl_basis = eigen_vectors
    return mean_vec, kl_basis

def apply_kl_transform(image, mean_vec, kl_basis):

    H, W, C = image.shape
    flat_data = image.reshape(-1, C)
    centered_data = flat_data - mean_vec
    transformed_data = centered_data @ kl_basis
    return transformed_data.reshape(H, W, C)

def apply_inverse_kl_transform(transformed_image, mean_vec, kl_basis):

    H, W, C = transformed_image.shape
    flat_data = transformed_image.reshape(-1, C)
    original_data = flat_data @ kl_basis.T + mean_vec
    return original_data.reshape(H, W, C)

def quantize_image(image, bits_per_channel):

    quantized = np.zeros_like(image)
    C = image.shape[-1]
    
    for c in range(C):
        nb_bits = bits_per_channel[c]
        if nb_bits <= 0:
            quantized[..., c] = 0
            continue
        
        channel_data = image[..., c]
        c_min, c_max = channel_data.min(), channel_data.max()
        if np.isclose(c_min, c_max):
            quantized[..., c] = channel_data
            continue
        
        levels = 2 ** nb_bits - 1
        norm_data = (channel_data - c_min) / (c_max - c_min)
        q_data = np.round(norm_data * levels)
        quant_channel = (q_data / levels) * (c_max - c_min) + c_min
        quantized[..., c] = quant_channel
    
    return quantized

def compute_psnr(original, reconstructed):

    return peak_signal_noise_ratio(original, reconstructed, data_range=original.max() - original.min())

def compute_ssim(original, reconstructed, win_size=7):

    H, W = original.shape[:2]
    adapted_win_size = min(win_size, H, W)
    if adapted_win_size % 2 == 0:
        adapted_win_size -= 1
    return structural_similarity(
        original,
        reconstructed,
        data_range=original.max() - original.min(),
        channel_axis=-1,
        win_size=adapted_win_size
    )

def compute_compression(original_bits_per_pixel, quant_bits_per_channel, image_shape):
 
    H, W = image_shape
    total_pixels = H * W
    original_size_bits = total_pixels * original_bits_per_pixel
    new_bits_per_pixel = sum(quant_bits_per_channel)
    new_size_bits = total_pixels * new_bits_per_pixel
    if new_size_bits == 0:
        return 100.0
    ratio = original_size_bits / new_size_bits
    compression_pct = (1.0 - 1.0 / ratio) * 100.0
    return max(0.0, min(compression_pct, 100.0))


def main_modified():
    # Image cible i2
    i2_path = "data/kodim23.png"
    # Dossier contenant les images i1 
    i1_folder = "data"  

    # Choix de l'espace colorimétrique 
    use_yuv = False 
    # quantification
    bits_per_channel = [8, 2, 2]

    original_bits_per_pixel = 24

    i2_rgb = io.imread(i2_path)
    H, W = i2_rgb.shape[:2]
    i2_rgb_float = img_as_float(i2_rgb).astype(np.float32)
    if use_yuv:
        i2_color = rgb_to_yuv(i2_rgb_float)
    else:
        i2_color = i2_rgb_float

    output_folder = "data_q3/kodim23"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i1_paths = sorted(glob.glob(os.path.join(i1_folder, "*.png")))
    if not i1_paths:
        print(f"Aucune image trouvée dans le dossier {i1_folder}")
        return

    labels = []
    psnr_list = []
    ssim_list = []
    comp_list = []
    score_list = []

    # --- Traitement pour chaque image i1 ---
    for i1_path in i1_paths:
        
        label = os.path.splitext(os.path.basename(i1_path))[0]
        labels.append(label)
        
        # Lecture et conversion de i1
        i1_rgb = io.imread(i1_path)
        i1_rgb_float = img_as_float(i1_rgb).astype(np.float32)
        if use_yuv:
            i1_color = rgb_to_yuv(i1_rgb_float)
        else:
            i1_color = i1_rgb_float

        # Calcul de la base KL à partir de i1
        mean_vec, kl_basis = compute_kl_transform_basis(i1_color)

        # Application de la transformation KL sur i2
        i2_transformed = apply_kl_transform(i2_color, mean_vec, kl_basis)
        # Quantification
        i2_quantized = quantize_image(i2_transformed, bits_per_channel)
        # Reconstruction par transformation inverse KL
        i2_inv_kl = apply_inverse_kl_transform(i2_quantized, mean_vec, kl_basis)

        if use_yuv:
            i2_reconstructed = yuv_to_rgb(i2_inv_kl)
        else:
            i2_reconstructed = i2_inv_kl
        i2_reconstructed = np.clip(i2_reconstructed, 0.0, 1.0)

        rec_filename = f"image_rec_{label}_to_i2.png"
        output_path = os.path.join(output_folder, rec_filename)
        io.imsave(output_path, img_as_ubyte(i2_reconstructed))

        # --- Calcul des métriques ---
        psnr_val = compute_psnr(i2_rgb_float, i2_reconstructed)
        ssim_val = compute_ssim(i2_rgb_float, i2_reconstructed, win_size=5)
        comp_pct = compute_compression(original_bits_per_pixel, bits_per_channel, (H, W))
        psnr_norm = min(1, max(0, (psnr_val - 20) / 40))
        score = (comp_pct / 100) * ((ssim_val**2 + psnr_norm**2) / 2)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        comp_list.append(comp_pct)
        score_list.append(score)

        print(f"{label} -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, Compression: {comp_pct:.2f} %, Score: {score:.4f}")

    
    if use_yuv:
        colors = ["#006E12", "#15952A", "#4AC95F", "#88FD9C", "#C3FFCE"]
    else:
        colors = ["#6600DB", "#8921FF", "#A95CFF", "#CC9EFF", "#E0C4FF"]
    if len(labels) > len(colors):
        colors = colors * ((len(labels) // len(colors)) + 1)

    

if __name__ == "__main__":
    main_modified()
