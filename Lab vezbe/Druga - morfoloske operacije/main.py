import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == '__main__':
    img = cv2.imread("../MaterijalLV2/coins.png")
    plt.figure(figsize=(12, 10))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.title("Original Image")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prvi deo zadatka
    # _, mask_coins = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_coins = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_filtered = cv2.morphologyEx(mask_coins, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    plt.subplot(222)
    plt.imshow(mask_coins, cmap='gray')
    plt.title("Mask Coins")

    # Drugi deo zadatka
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]  # saturacioni kanal
    _, copper_coin_lower = cv2.threshold(img_sat, 150, 255, cv2.THRESH_BINARY)
    _, copper_coin_upper = cv2.threshold(img_sat, 250, 255, cv2.THRESH_BINARY_INV)
    copper_coin = cv2.bitwise_and(copper_coin_upper, copper_coin_lower)
    copper_coin_close = cv2.morphologyEx(copper_coin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marker = cv2.bitwise_and(copper_coin_close, mask_filtered)
    plt.subplot(223)
    plt.imshow(marker)
    plt.title("Marker")

    # Treci deo zadatka
    reconstructed = morphological_reconstruction(marker, mask_filtered)
    plt.subplot(224)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.show()

