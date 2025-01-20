import time
from collections import Counter
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Fonction pour construire l'arbre de Huffman
def build_huffman_tree(frequencies):
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)

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

    if root.char is not None:
        codes[root.char] = current_code

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

    return codes

# Fonction principale
def huffman_compression(text):
    

    # Nombre de caractères et caractères uniques
    total_chars = len(text)
    frequencies = Counter(text)
    unique_chars = len(frequencies)
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    # Histogramme
    plt.bar(sorted_frequencies.keys(),sorted_frequencies.values())
    plt.title("Histogramme des apparitions des caractères")
    plt.xlabel("Caractères")
    plt.ylabel("Nombre d'apparitions")
    plt.show()

    # Nombre de bits pour coder chaque caractère (non compressé)
    bits_per_char = len(bin(unique_chars - 1)[2:])
    total_bits_uncompressed = total_chars * bits_per_char
    
    # Génération des codes Huffman
    start_time = time.time()
    huffman_tree = build_huffman_tree(frequencies)

    huffman_codes = generate_huffman_codes(huffman_tree)
    compression_time = time.time() - start_time
    # Nombre total de bits après compression
    total_bits_compressed = sum(frequencies[char] * len(huffman_codes[char]) for char in frequencies)

    

    # Taux de compression
    compression_rate = (1 - total_bits_compressed / total_bits_uncompressed) * 100

    # Caractères le plus et le moins fréquents
    most_frequent = max(frequencies.items(), key=lambda x: x[1])
    least_frequent = min(frequencies.items(), key=lambda x: x[1])

    print(f"Nombre de caractères : {total_chars}")
    print(f"Nombre de caractères différents : {unique_chars}")
    print(f"Nombre de bits pour coder un caractère (non compressé) : {bits_per_char}")
    print(f"Nombre total de bits pour coder le texte (non compressé) : {total_bits_uncompressed}")
    print(f"Nombre total de bits après codage Huffman : {total_bits_compressed}")
    print(f"Caractère le plus fréquent : '{most_frequent[0]}' avec code {huffman_codes[most_frequent[0]]} ({frequencies[most_frequent[0]]} occurrences)")
    print(f"Caractère le moins fréquent : '{least_frequent[0]}' avec code {huffman_codes[least_frequent[0]]} ({frequencies[least_frequent[0]]} occurrence)")
    print(f"Temps pour compresser (codage Huffman) : {compression_time:.6f} secondes")
    print(f"Taux de compression : {compression_rate:.2f}%")

if __name__ == "__main__":
    with open("data TP1/textes/texte_5.txt", "r", encoding="utf-8") as file:
        texte = file.read()

    huffman_compression(texte)
