import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar
def show_image(image, title, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')

# Fungsi untuk ekstrak channel warna
def extract_channel(image, channel_index):
    channel_image = np.zeros_like(image)
    channel_image[:, :, channel_index] = image[:, :, channel_index]
    return channel_image

# Fungsi konversi ke grayscale
def to_grayscale(image):
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale_image

# Fungsi konversi ke citra biner (thresholding)
def to_binary(image, threshold=128):
    grayscale_image = to_grayscale(image)
    binary_image = (grayscale_image > threshold) * 255
    return binary_image

# Load gambar daun (ubah nama file jika perlu)
images = {
    "Daun Pepaya": iio.imread('daunpepaya.jpg'),
    "Singkong": iio.imread('daunsingkong.jpg'),
    "Kenikir": iio.imread('daunkenikir.jpg')
}

# Iterasi untuk setiap gambar
for name, image in images.items():
    plt.figure(figsize=(10, 6))
    plt.suptitle(name, fontsize=16)

    # a) Channel warna R (Red)
    red_channel = extract_channel(image, 0)
    plt.subplot(2, 3, 1)
    show_image(red_channel, 'Red Channel')

    # b) Channel warna G (Green)
    green_channel = extract_channel(image, 1)
    plt.subplot(2, 3, 2)
    show_image(green_channel, 'Green Channel')

    # c) Channel warna B (Blue)
    blue_channel = extract_channel(image, 2)
    plt.subplot(2, 3, 3)
    show_image(blue_channel, 'Blue Channel')

    # d) Konversi warna Grayscale
    grayscale = to_grayscale(image)
    plt.subplot(2, 3, 4)
    show_image(grayscale, 'Grayscale', cmap='gray')

    # e) Konversi warna Threshold (Biner)
    binary_image = to_binary(image)
    plt.subplot(2, 3, 5)
    show_image(binary_image, 'Binary (Threshold)', cmap='gray')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
