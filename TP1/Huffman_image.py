import time
from collections import Counter
import heapq
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Node:
    def __init__(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Fonction pour construire l'arbre de Huffman
def build_huffman_tree(frequencies):
    heap = [Node(pixel, freq) for pixel, freq in frequencies.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        single_node = heapq.heappop(heap)
        root = Node(None, single_node.freq)  
        root.left = single_node  
        heapq.heappush(heap, root)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

# Fonction pour générer les codes Huffman
def generate_huffman_codes(root, current_code="", codes={}):
    if root is None:
        return

    if root.pixel is not None:
        codes[root.pixel] = current_code if current_code else "0"  # Gérer un seul pixel

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

    return codes


# Modifier l'affichage de l'histogramme
def plot_histogram(sorted_pixel_values, sorted_pixel_counts):
    plt.figure(figsize=(15, 6))

    plt.bar(range(len(sorted_pixel_counts)), sorted_pixel_counts)

    for i, (pixel, count) in enumerate(zip(sorted_pixel_values, sorted_pixel_counts)):
        color = eval(pixel)  

        plt.gca().add_patch(
            plt.Rectangle(
                (i - 0.4, -max(sorted_pixel_counts) * 0.1),  
                0.8,  
                max(sorted_pixel_counts) * 0.05,  
                color=np.array(color) / 255,  
                clip_on=False,  
            )
        )

        plt.text(
            i, 
            -max(sorted_pixel_counts) * 0.15,  
            f"{color}",
            ha="center",  
            va="center",  
            fontsize=7,  
            rotation=0  
        )

        plt.text(
            i,  
            count + max(sorted_pixel_counts) * 0.02,  
            str(count), 
            ha="center",  
            va="bottom",  
            fontsize=8,  
        )

    plt.title("Histogramme des fréquences des pixels avec légendes colorées et codes RGB")
    plt.xlabel("Pixels (affichés comme couleurs et codes RGB)")
    plt.ylabel("Nombre d'apparitions")
    plt.xticks(range(len(sorted_pixel_counts)), [""] * len(sorted_pixel_counts))  
    plt.ylim(-max(sorted_pixel_counts) * 0.2, max(sorted_pixel_counts) * 1.1)  
    plt.show()





# Fonction principale
def huffman_compression_image(image_path):
    

    #convertir en tableau numpy
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)

    print(pixels)

    #les pixels uniques et leurs fréquences
    pixels_tuple = [tuple(pixel) for pixel in pixels]
    frequencies = Counter(pixels_tuple)

    # Nombre de pixels et pixels uniques
    total_pixels = len(pixels_tuple)
    unique_pixels = len(frequencies)

    # Histogramme
    pixel_values = [f"{pixel}" for pixel in frequencies.keys()]
    pixel_counts = list(frequencies.values())
    sorted_data = sorted(zip(pixel_values, pixel_counts), key=lambda x: x[1], reverse=True)
    sorted_pixel_values = [item[0] for item in sorted_data[:15]]
    sorted_pixel_counts = [item[1] for item in sorted_data[:15]]

    plot_histogram(sorted_pixel_values, sorted_pixel_counts)


    # Nombre de bits pour coder chaque pixel (non compressé)
    bits_per_pixel = len(bin(unique_pixels - 1)[2:])
    total_bits_uncompressed = total_pixels * bits_per_pixel

    # Génération des codes Huffman
    start_time = time.time()
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(huffman_tree)
    compression_time = time.time() - start_time


    # Nombre total de bits après compression
    total_bits_compressed = sum(frequencies[pixel] * len(huffman_codes[pixel]) for pixel in frequencies)

    

    # Taux de compression
    compression_rate = (1 - total_bits_compressed / total_bits_uncompressed) * 100

    # Pixels le plus et le moins fréquents
    most_frequent = max(frequencies.items(), key=lambda x: x[1])
    least_frequent = min(frequencies.items(), key=lambda x: x[1])

    print(f"Nombre de pixels : {total_pixels}")
    print(f"Nombre de pixels différents : {unique_pixels}")
    print(f"Nombre de bits pour coder un pixel (non compressé) : {bits_per_pixel}")
    print(f"Nombre total de bits pour coder l'image (non compressé) : {total_bits_uncompressed}")
    print(f"Nombre total de bits après codage Huffman : {total_bits_compressed}")
    print(f"Pixel le plus fréquent : {most_frequent[0]} avec code {huffman_codes[most_frequent[0]]} ({frequencies[most_frequent[0]]} occurrences)")
    print(f"Pixel le moins fréquent : {least_frequent[0]} avec code {huffman_codes[least_frequent[0]]} ({frequencies[least_frequent[0]]} occurrence)")
    print(f"Temps pour compresser (codage Huffman) : {compression_time:.6f} secondes")
    print(f"Taux de compression : {compression_rate:.2f}%")

if __name__ == "__main__":
    image_path = "data TP1/images/image_1.png"
    huffman_compression_image(image_path)
