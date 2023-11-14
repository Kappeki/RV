import numpy as np
import cv2
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
    img = cv2.imread("Circles.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask_circles = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_circles = cv2.threshold(img_gray, 2, 255, cv2.THRESH_BINARY)
    mask_filtered = cv2.morphologyEx(mask_circles, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    plt.imshow(mask_filtered)
    plt.show()
    img_blue = img[:, :, 2]
    plt.imshow(img_blue)
    plt.show()
    _, blue_circles_lower = cv2.threshold(img_blue, 129, 255, cv2.THRESH_BINARY)
    _, blue_circles_upper = cv2.threshold(img_blue, 250, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(blue_circles_lower)
    plt.show()
    plt.imshow(blue_circles_upper)
    plt.show()
    blue_circles = cv2.bitwise_and(blue_circles_upper, blue_circles_lower)
    plt.imshow(blue_circles)
    plt.show()
    blue_circles_open = cv2.morphologyEx(blue_circles, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marker = cv2.bitwise_and(blue_circles_open, mask_filtered)
    plt.imshow(marker)
    plt.show()
    reconstructed = morphological_reconstruction(marker, mask_filtered)
    plt.imshow(reconstructed)
    plt.show()
