import cv2
import numpy as np

KERNEL_SIZE = 9
CONTOUR_MAX = 4  # 4 for 4 sides

def preprocess(img):
    # N.B. img arg is in greyscale
    # Gaussian Blur + Adaptive Thresholding
    improc = cv2.GaussianBlur(img.copy(), (KERNEL_SIZE, KERNEL_SIZE), 0)
    improc = cv2.adaptiveThreshold(improc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Color inversion and dilation to remove noise
    improc = cv2.bitwise_not(improc, improc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    improc = cv2.dilate(improc, kernel)
    return improc



