import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def inverse_fft(magnitued_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitued_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered

def low_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius:
                img_fft_log[x,y] = 0

    plt.imshow(img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

def high_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) < radius*radius: # jednacina kruga, zelimo da sve unutar kruga bude nula sto predstavlja idealni high pass filter
                img_fft_log[x,y] = 0

    plt.imshow(img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

def band_stop_filter(img, center, low_cutoff, high_cutoff):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    plt.subplot(223)
    plt.imshow(img_fft_log, cmap='gray')
    plt.title("Before noise removal")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if ((x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > low_cutoff*low_cutoff and
                (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) < high_cutoff*high_cutoff):
                    img_fft_log[x,y] = 0

    plt.subplot(224)
    plt.imshow(img_fft_log, cmap="gray")
    plt.title("After noise removal")

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered


def filtering_in_spatial_domain(img):
    gaus_kernel = cv2.getGaussianKernel(ksize=21, sigma=7) # Gausov kernel velicine 21x21 sa standardnom devijacijom 7
    kernel = np.zeros((3, 3), dtype=np.int8) # custom 3x3 kernel popunjem svim nulama
    kernel[1, 2] = 2 # dodaje se jedinica na u srednjem redu desno, pa kernel postaje: [[0 0 0] [0 0 1] [0 0 0]]
    img_gauss_blur = cv2.filter2D(img, -1, gaus_kernel) # gausov kernel
    img_filter_custom = cv2.filter2D(img, -1, kernel) # custom kernel

    return img_gauss_blur, img_filter_custom

if __name__ == '__main__':
    img = cv2.imread("../MaterijalLV1/slika_4.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    center = (256, 256)
    radius = 70

    low_cutoff = 70
    high_cutoff = 71

    plt.figure(figsize=(10, 8))

    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title("Original image")

    #img_gauss, img_custom = filtering_in_spatial_domain(img)


    # img_low_pass = low_pass_filter(img, center, radius)
    # plt.imshow(img_low_pass, cmap='gray')
    # plt.show()
    #
    # img_high_pass = high_pass_filter(img, center, radius)
    # plt.imshow(img_high_pass, cmap='gray')
    # plt.show()

    img_band_stop = band_stop_filter(img, center, low_cutoff, high_cutoff)
    plt.subplot(222)
    plt.imshow(img_band_stop, cmap='gray')
    plt.title("Image after noise removal")

    plt.show()




